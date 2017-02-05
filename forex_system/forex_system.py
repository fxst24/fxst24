# coding: utf-8

import configparser
import cvxopt as opt
import numpy as np
import oandapy
import os
import pandas as pd
import requests
import shutil
import smtplib
import threading
import time
from cvxopt import blas, solvers
from datetime import datetime
from datetime import timedelta
from email.mime.text import MIMEText
from numba import float64, int64, jit
#from pykalman import KalmanFilter
from scipy import optimize
from sklearn import linear_model
from sklearn import tree
from sklearn.externals import joblib

# threading.Lockオブジェクトを生成する。
LOCK = threading.Lock()
# OANDA API用に設定する。
OANDA = None
ENVIRONMENT = None
ACCESS_TOKEN = None
ACCOUNT_ID = None
COUNT = 500
# 許容誤差を設定する。
EPS = 1.0e-5
## Spyderのバグ（？）で警告が出ることがあるので無視する。
#import warnings
#warnings.simplefilter(action = 'ignore', category = RuntimeWarning)

def adjust_summer_time(data, timeframe):
    '''夏時間を大まかに調整して返す（1時間足以下のみ）。
    Args:
        data: データ。
        timeframe: 足の種類。
    Returns:
        調整後のデータ（3-10月を夏時間と見なす）。
    '''
    adjusted_data = data.copy()
    shift = int(60 / timeframe)
    adjusted_data[(data.index.month>=3) & ((data.index.month<11))] = (
        data.shift(-shift))
    # NAを後のデータで補間する。
    data = data.fillna(method='ffill')
    return adjusted_data

def ask(instrument):
    '''買値を得る。
    Args:
        instrument: OANDA APIでの通貨ペア名。
    Returns:
        買値。
    '''
    instruments = OANDA.get_prices(instruments=instrument)
    ask = instruments['prices'][0]['ask']
    return ask

def bid(instrument):
    '''売値を得る。
    Args:
        instrument: OANDA APIでの通貨ペア名。
    Returns:
        売値。
    '''
    global OANDA
    instruments = OANDA.get_prices(instruments=instrument)
    bid = instruments['prices'][0]['bid']
    return bid

def calc_apr(ret, start, end):
    '''年率を計算する。
    Args:
        ret: リターン。
        start: 開始日。
        end: 終了日。
    Returns:
        年率。
    '''
    rate = (ret + 1.0).prod() - 1.0
    years = (end - start).total_seconds() / 60.0 / 60.0 / 24.0 / 365.0
    apr = rate / years

    return apr

def calc_drawdowns(ret):
    '''最大ドローダウン（％）を計算する。
    Args:
        ret: リターン。
    Returns:
        最大ドローダウン（％）。
    '''
    # 累積リターンを計算する。
    cum_ret = (ret + 1.0).cumprod() - 1.0
    cum_ret = np.array(cum_ret)
    # 最大ドローダウンを計算する。
    @jit(float64(float64[:]), nopython=True, cache=True)
    def func(cum_ret):
        length = len(cum_ret)
        high_watermark = np.zeros(length)
        drawdown = np.zeros(length)
        for i in range(length):
            if i == 0:
                high_watermark[0] = cum_ret[0]
            else:
                if high_watermark[i - 1] >= cum_ret[i]:
                    high_watermark[i] = high_watermark[i - 1]
                else:
                    high_watermark[i] = cum_ret[i];
            if np.abs(1.0 + cum_ret[i]) < EPS:
                drawdown[i] = drawdown[i - 1]
            else:
                drawdown[i] = ((1.0 + high_watermark[i]) /
                (1.0 + cum_ret[i]) - 1.0)
        for i in range(length):
            if i == 0:
                drawdowns = drawdown[0];
            else:
                if drawdown[i] > drawdowns:
                    drawdowns = drawdown[i];
        return drawdowns
    drawdowns = func(cum_ret)

    return drawdowns
 
def calc_durations(ret, timeframe):
    '''最大ドローダウン期間を計算する。
    Args:
        ret: リターン。
        timeframe: 期間。
    Returns:
        最大ドローダウン期間。
    '''
    # 累積リターンを計算する。
    cum_ret = (ret + 1.0).cumprod() - 1.0
    cum_ret = np.array(cum_ret)
    # 最大ドローダウン期間を計算する。
    @jit(float64(float64[:]), nopython=True, cache=True)
    def func(cum_ret):
        length = len(cum_ret)
        high_watermark = np.zeros(length)
        drawdown = np.zeros(length)
        duration = np.zeros(length)
        for i in range(length):
            if i == 0:
                high_watermark[0] = cum_ret[0]
            else:
                if high_watermark[i - 1] >= cum_ret[i]:
                    high_watermark[i] = high_watermark[i - 1]
                else:
                    high_watermark[i] = cum_ret[i];
            if np.abs(1.0 + cum_ret[i]) < EPS:
                drawdown[i] = drawdown[i - 1]
            else:
                drawdown[i] = ((1.0 + high_watermark[i]) /
                (1.0 + cum_ret[i]) - 1.0)
        for i in range(length):
            if i == 0:
                duration[0] = drawdown[0]
            else:
                if drawdown[i] == 0.0:
                    duration[i] = 0
                else:
                    duration[i] = duration[i - 1] + 1;
        for i in range(length):
            if i == 0:
                durations = duration[0];
            else:
                if duration[i] > durations:
                    durations = duration[i]
        durations = durations / (1440 / timeframe)  # 営業日単位に変換
        return durations
    durations = int(func(cum_ret))

    return durations
 
def calc_kelly(ret):
    '''ケリー基準による最適レバレッジを計算する。
    Args:
        ret: リターン。
    Returns:
        ケリー基準による最適レバレッジ。
    '''
    mean = ret.mean()
    std = ret.std()
    # 標準偏差が0であった場合、とりあえず最適レバレッジは0ということにしておく。
    if np.abs(std) < EPS:  # 標準偏差がマイナスになるはずはないが一応。
        kelly = 0.0
    else:
        kelly = mean / (std * std)

    return kelly
 
def calc_ret(symbol, timeframe, signal, spread, position, start, end):
    '''リターンを計算する。
    Args:
        symbol: 通貨ペア。
        timeframe: 期間。
        signal: シグナル。
        spread: スプレッド。
        position: ポジションの設定。
            0: 買いのみ。
            1: 売りのみ。
            2: 売買両方。
        start: 開始日。
        end: 終了日。
    Returns:
        リターン。
    '''
    # スプレッドの単位を調整する。
    if (symbol == 'AUDJPY' or symbol == 'CADJPY' or symbol == 'CHFJPY' or
        symbol == 'EURJPY' or symbol == 'GBPJPY' or symbol == 'NZDJPY' or
        symbol == 'USDJPY'):
        adjusted_spread = spread / 1000.0
    else:
        adjusted_spread = spread / 100000.0
    # シグナルを調整する。
    adjusted_signal = signal.copy()
    if position == 0:
        adjusted_signal[adjusted_signal==-1] = 0
    elif position == 1:
        adjusted_signal[adjusted_signal==1] = 0
    # コストを計算する。
    cost_buy_entry = (((adjusted_signal == 1) &
                       (adjusted_signal.shift(1) != 1)) * adjusted_spread)
    cost_sell_entry = (((adjusted_signal == -1) &
                       (adjusted_signal.shift(1) != -1)) * adjusted_spread)
    cost = cost_buy_entry + cost_sell_entry
    # リターンを計算する。
    op = i_open(symbol, timeframe, 0)
    ret = ((op.shift(-1) - op) * adjusted_signal - cost) / op
    ret = ret.fillna(0.0)
    ret[(ret==float('inf')) | (ret==float('-inf'))] = 0.0
    ret = ret[start:end]

    return ret
 
def calc_sharpe(ret, start, end):
    '''シャープレシオを計算する。
    Args:
        ret: リターン。
        start: 開始日。
        end: 終了日。
    Returns:
        シャープレシオ。
    '''
    bars = len(ret)
    years = (end - start).total_seconds() / 60.0 / 60.0 / 24.0 / 365.0
    num_bar_per_year = bars / years
    mean = ret.mean()
    std = ret.std()
    # 標準偏差が0であった場合、とりあえずシャープレシオは0ということにしておく。
    if np.abs(std) < EPS:  # 標準偏差がマイナスになるはずはないが一応。
        sharpe = 0.0
    else:
        sharpe = np.sqrt(num_bar_per_year) * mean / std
    return sharpe

def calc_signal(buy_entry, buy_exit, sell_entry, sell_exit):
    '''シグナルを計算する。
    Args:
        buy_entry: 買いエントリー。
        buy_exit: 買いエグジット。
        sell_entry: 売りエントリー。
        sell_exit: 売りエグジット。
    Returns:
        シグナル。
    '''
    # シグナルを計算する。
    buy_entry = buy_entry.astype(int)
    buy_exit = buy_exit.astype(int)
    sell_entry = sell_entry.astype(int)
    sell_exit = sell_exit.astype(int)
    buy = buy_entry.copy()
    buy[buy==0] = np.nan
    buy[buy_exit==1] = 0
    buy = buy.fillna(method='ffill')
    sell = -sell_entry.copy()
    sell[sell==0] = np.nan
    sell[sell_exit==1] = 0
    sell = sell.fillna(method='ffill')
    signal = buy + sell
    signal = signal.fillna(0)
    signal = signal.astype(int)
    return signal

def calc_trades(signal, position, start, end):
    '''トレード数を計算する。
    Args:
        signal: シグナル。
        position: ポジションの設定。
            0: 買いのみ。
            1: 売りのみ。
            2: 売買両方。
        start: 開始日。
        end: 終了日。
    Returns:
        トレード数。
    '''
    adjusted_signal = signal.copy()
    if position == 0:
        adjusted_signal[adjusted_signal==-1] = 0
    elif position == 1:
        adjusted_signal[adjusted_signal==1] = 0
    temp1 = (((adjusted_signal > 0) &
        (adjusted_signal > adjusted_signal.shift(1))) *
        (adjusted_signal - adjusted_signal.shift(1)))
    temp2 = (((adjusted_signal < 0) &
        (adjusted_signal < adjusted_signal.shift(1))) *
        (adjusted_signal.shift(1) - adjusted_signal))
    trade = temp1 + temp2
    trade = trade.fillna(0)
    trade = trade.astype(int)
    trades = trade[start:end].sum()
    return trades

# そのうちに変数名の変更、コメントの追加を行う。
def calc_weights(ret, portfolio):
    '''ウェイトを計算する。
    Args:
        ret: リターン。
        portfolio: ポートフォリオの最適化の設定。
            0: 最適化を行わない（等ウェイト）。
            1: 最適化を行う。
    Returns:
        ウェイト。
    '''
    ret = ret.T
    n_eas = len(ret)
    if portfolio == 1:
        ret = np.asmatrix(ret)
        n = 100
        solvers.options['show_progress'] = False
        mus = [10**(5.0 * t/n - 1.0) for t in range(n)]
        S = opt.matrix(np.cov(ret))
        pbar = opt.matrix(np.mean(ret, axis=1))
        G = -opt.matrix(np.eye(n_eas))
        h = opt.matrix(0.0, (n_eas ,1))
        A = opt.matrix(1.0, (1, n_eas))
        b = opt.matrix(1.0)
        portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]
        ret = [blas.dot(pbar, x) for x in portfolios]
        risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
        m1 = np.polyfit(ret, risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])
        weights = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
        weights = np.asarray(weights)
        weights = weights.reshape(len(weights),)
    else:
        weights = np.empty(n_eas)
        for i in range(n_eas):
            weights[i] = 1.0 / n_eas
    return weights

def convert_minute2period(minute, timeframe):
    '''分を計算期間に変換する。
    Args:
        minute: 分。
        timeframe: 足の種類。
    Returns:
        計算期間。
    '''
    period = int(minute / timeframe)
    return period
    
def convert_symbol2instrument(symbol):
    '''symbolをinstrumentに変換する。
    Args:
        symbol: 通貨ペア。
    Returns:
        instrument。
    '''
    if symbol == 'AUDCAD':
        instrument = 'AUD_CAD'
    elif symbol == 'AUDCHF':
        instrument = 'AUD_CHF'
    elif symbol == 'AUDJPY':
        instrument = 'AUD_JPY'
    elif symbol == 'AUDNZD':
        instrument = 'AUD_NZD'
    elif symbol == 'AUDUSD':
        instrument = 'AUD_USD'
    elif symbol == 'CADCHF':
        instrument = 'CAD_CHF'
    elif symbol == 'CADJPY':
        instrument = 'CAD_JPY'
    elif symbol == 'CHFJPY':
        instrument = 'CHF_JPY'
    elif symbol == 'EURAUD':
        instrument = 'EUR_AUD'
    elif symbol == 'EURCAD':
        instrument = 'EUR_CAD'
    elif symbol == 'EURCHF':
        instrument = 'EUR_CHF'
    elif symbol == 'EURGBP':
        instrument = 'EUR_GBP'
    elif symbol == 'EURJPY':
        instrument = 'EUR_JPY'
    elif symbol == 'EURNZD':
        instrument = 'EUR_NZD'
    elif symbol == 'EURUSD':
        instrument = 'EUR_USD' 
    elif symbol == 'GBPAUD':
        instrument = 'GBP_AUD'
    elif symbol == 'GBPCAD':
        instrument = 'GBP_CAD'
    elif symbol == 'GBPCHF':
        instrument = 'GBP_CHF'
    elif symbol == 'GBPJPY':
        instrument = 'GBP_JPY'
    elif symbol == 'GBPNZD':
        instrument = 'GBP_NZD'
    elif symbol == 'GBPUSD':
        instrument = 'GBP_USD'
    elif symbol == 'NZDCAD':
        instrument = 'NZD_CAD'
    elif symbol == 'NZDCHF':
        instrument = 'NZD_CHF'
    elif symbol == 'NZDJPY':
        instrument = 'NZD_JPY'
    elif symbol == 'NZDUSD':
        instrument = 'NZD_USD'
    elif symbol == 'USDCAD':
        instrument = 'USD_CAD'
    elif symbol == 'USDCHF':
        instrument = 'USD_CHF'
    else:
        instrument = 'USD_JPY'
    return instrument

def convert_timeframe2granularity(timeframe):
    '''timeframeをgranularityに変換する。
    Args:
        timeframe: 期間。
    Returns:
        granularity。
    '''
    if timeframe == 1:
        granularity = 'M1'
    elif timeframe == 5:
        granularity = 'M5'
    elif timeframe == 15:
        granularity = 'M15'
    elif timeframe == 30:
        granularity = 'M30'
    elif timeframe == 60:
        granularity = 'H1'
    elif timeframe == 240:
        granularity = 'H4'
    else:
        granularity = 'D'
    return granularity

def convert2annual_value_with_root_t_rule(ts, timeframe, period):
    '''時系列データを√Tルールによって1年当たりの値に変換する。
    Args:
        ts: 時系列データ。
        timeframe: 期間。
        period: 計算期間。
    Returns:
        1年当たりの値。
    '''
    annual_value = ts * np.sqrt(60 * 24 * 260 / (timeframe * period))

    return annual_value

def divide_symbol(symbol):
    '''通貨ペアをベース通貨とクウォート通貨に分ける。
    Args:
        symbol: 通貨ペア。
    Returns:
        ベース通貨、クウォート通貨。
    '''
    if symbol == 'AUDCAD':
        base = 'aud'
        quote = 'cad'
    elif symbol == 'AUDCHF':
        base = 'aud'
        quote = 'chf'
    elif symbol == 'AUDJPY':
        base = 'aud'
        quote = 'jpy'
    elif symbol == 'AUDNZD':
        base = 'aud'
        quote = 'nzd'
    elif symbol == 'AUDUSD':
        base = 'aud'
        quote = 'usd'
    elif symbol == 'CADCHF':
        base = 'cad'
        quote = 'chf'
    elif symbol == 'CADJPY':
        base = 'cad'
        quote = 'jpy'
    elif symbol == 'CHFJPY':
        base = 'chf'
        quote = 'jpy'
    elif symbol == 'EURAUD':
        base = 'eur'
        quote = 'aud'
    elif symbol == 'EURCAD':
        base = 'eur'
        quote = 'cad'
    elif symbol == 'EURCHF':
        base = 'eur'
        quote = 'chf'
    elif symbol == 'EURGBP':
        base = 'eur'
        quote = 'gbp'
    elif symbol == 'EURJPY':
        base = 'eur'
        quote = 'jpy'
    elif symbol == 'EURNZD':
        base = 'eur'
        quote = 'nzd'
    elif symbol == 'EURUSD':
        base = 'eur'
        quote = 'usd'
    elif symbol == 'GBPAUD':
        base = 'gbp'
        quote = 'aud'
    elif symbol == 'GBPCAD':
        base = 'gbp'
        quote = 'cad'
    elif symbol == 'GBPCHF':
        base = 'gbp'
        quote = 'chf'
    elif symbol == 'GBPJPY':
        base = 'gbp'
        quote = 'jpy'
    elif symbol == 'GBPNZD':
        base = 'gbp'
        quote = 'nzd'
    elif symbol == 'GBPUSD':
        base = 'gbp'
        quote = 'usd'
    elif symbol == 'NZDCAD':
        base = 'nzd'
        quote = 'cad'
    elif symbol == 'NZDCHF':
        base = 'nzd'
        quote = 'chf'
    elif symbol == 'NZDJPY':
        base = 'nzd'
        quote = 'jpy'
    elif symbol == 'NZDUSD':
        base = 'nzd'
        quote = 'usd'
    elif symbol == 'USDCAD':
        base = 'usd'
        quote = 'cad'
    elif symbol == 'USDCHF':
        base = 'usd'
        quote = 'chf'
    else:
        base = 'usd'
        quote = 'jpy'
    return base, quote

def divide_symbol_time(symbol):
    '''通貨ペアをベース通貨とクウォート通貨それぞれの時間に分ける。
    Args:
        symbol: 通貨ペア。
    Returns:
        ベース通貨、クウォート通貨それぞれの時間。
    '''
    # AUD、JPY、NZDは東京時間、CHF、EUR、GBPはロンドン時間、CAD、USDはNY時間とする。
    if symbol == 'AUDCAD':
        base_time = 'tokyo'
        quote_time = 'ny'
    elif symbol == 'AUDCHF':
        base_time = 'tokyo'
        quote_time = 'ldn'
    elif symbol == 'AUDJPY':
        base_time = 'tokyo'
        quote_time = 'tokyo'
    elif symbol == 'AUDNZD':
        base_time = 'tokyo'
        quote_time = 'tokyo'
    elif symbol == 'AUDUSD':
        base_time = 'tokyo'
        quote_time = 'ny'
    elif symbol == 'CADCHF':
        base_time = 'ny'
        quote_time = 'ldn'
    elif symbol == 'CADJPY':
        base_time = 'ny'
        quote_time = 'tokyo'
    elif symbol == 'CHFJPY':
        base_time = 'ldn'
        quote_time = 'tokyo'
    elif symbol == 'EURAUD':
        base_time = 'ldn'
        quote_time = 'tokyo'
    elif symbol == 'EURCAD':
        base_time = 'ldn'
        quote_time = 'ny'
    elif symbol == 'EURCHF':
        base_time = 'ldn'
        quote_time = 'ldn'
    elif symbol == 'EURGBP':
        base_time = 'ldn'
        quote_time = 'ldn'
    elif symbol == 'EURJPY':
        base_time = 'ldn'
        quote_time = 'tokyo'
    elif symbol == 'EURNZD':
        base_time = 'ldn'
        quote_time = 'tokyo'
    elif symbol == 'EURUSD':
        base_time = 'ldn'
        quote_time = 'ny'
    elif symbol == 'GBPAUD':
        base_time = 'ldn'
        quote_time = 'tokyo'
    elif symbol == 'GBPCAD':
        base_time = 'ldn'
        quote_time = 'ny'
    elif symbol == 'GBPCHF':
        base_time = 'ldn'
        quote_time = 'ldn'
    elif symbol == 'GBPJPY':
        base_time = 'ldn'
        quote_time = 'tokyo'
    elif symbol == 'GBPNZD':
        base_time = 'ldn'
        quote_time = 'tokyo'
    elif symbol == 'GBPUSD':
        base_time = 'ldn'
        quote_time = 'ny'
    elif symbol == 'NZDCAD':
        base_time = 'tokyo'
        quote_time = 'ny'
    elif symbol == 'NZDCHF':
        base_time = 'tokyo'
        quote_time = 'ldn'
    elif symbol == 'NZDJPY':
        base_time = 'tokyo'
        quote_time = 'tokyo'
    elif symbol == 'NZDUSD':
        base_time = 'tokyo'
        quote_time = 'ny'
    elif symbol == 'USDCAD':
        base_time = 'ny'
        quote_time = 'ny'
    elif symbol == 'USDCHF':
        base_time = 'ny'
        quote_time = 'ldn'
    else:
        base_time = 'ny'
        quote_time = 'tokyo'
    return base_time, quote_time

def get_current_filename_no_ext():
    '''拡張子なしの現在のファイル名を返す。
    Returns:
        拡張子なしの現在のファイル名。
    '''
    import inspect
    pathname = os.path.dirname(__file__)
    current_filename = inspect.currentframe().f_back.f_code.co_filename
    current_filename = current_filename.replace(pathname + '/', '') 
    current_filename_no_ext, ext = os.path.splitext(current_filename)
    return current_filename_no_ext

def get_intraday_data_from_google(symbol, timeframe):
    '''日中足のデータをGoogleから取得する。
      Args:
          symbol: 銘柄。
          timeframe: 足の種類。
      Returns:
          日中足のデータ。
    '''

    timeframe = timeframe * 60 + 1  # 秒に変換する（よく分からないが1を加える）。
    google = 'http://www.google.com/finance/'
    url = (google + 'getprices?q={0}'.format(symbol.upper()) +
        "&i={0}&p={1}d&f=d,o,h,l,c,v".format(timeframe, 10))

    # データをスクレイピングする。
    # データは銘柄によって8行目からの場合と9行目からの場合があるようだ。
    # 8行目の列数が少ない場合は9行目からと見なすことにする。
    temp = requests.get(url).text.split()
    if len(temp[7]) < 30:
        row = 8
    else:
        row = 7
    temp = [line.split(',') for line in temp[row:]]
    temp = pd.DataFrame(temp)
    index = temp.iloc[:, 0].apply(lambda x: datetime.fromtimestamp(int(x[1:])))
    index = pd.to_datetime(index)
    timeframe = int((timeframe - 1) / 60)  # 分に戻す。
    index = index - timedelta(minutes=timeframe)
    temp.index = index
    temp = temp.iloc[:, 1:]

    # データを整える。
    data = pd.DataFrame(index=index,
                        columns=['open','high','low','close','volume'])
    data['open'] = temp.iloc[:, 3].astype(float)
    data['high'] = temp.iloc[:, 1].astype(float)
    data['low'] = temp.iloc[:, 2].astype(float)
    data['close'] = temp.iloc[:, 0].astype(float)
    data['volume'] = temp.iloc[:, 4].astype(int)
    data.index.name = None

    return data

def has_event(index, timeframe, nfp=0):
    '''イベントの有無を返す。
    Args:
        index: インデックス。
        timeframe: 足の種類。
    Returns:
        イベントの有無。
    '''
    event = pd.Series(index=index)
    week = time_week(index)
    day_of_week = time_day_of_week(index)
    hour = time_hour(index)
    minute = time_minute(index)
    # 非農業部門雇用者数
    if nfp == 1:
        m = int(np.floor(30 / timeframe) * timeframe)
        h = int(np.floor(15 / (timeframe / 60)) * (timeframe / 60))
        event[(week==1) & (day_of_week==5) & (hour==h) & (minute==m)] = 1
    event = event.fillna(0)
    event = event.astype(int)
    return event

def hour():
    '''現在の時を返す。
    Returns:
        現在の時。
    '''    
    hour = datetime.now().hour
    return hour

def i_adj_kairi(symbol, timeframe, period, shift):
    '''調整済み移動位平均乖離率を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 期間。
        period: 計算期間。
        shift: シフト。
    Returns:
        調整済み移動位平均乖離率。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_adj_kairi_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        adj_kairi = joblib.load(path)
    # さもなければ計算する。
    else:
        close = i_close(symbol, timeframe, shift)
        mean = close.rolling(window=period).mean()
        kairi = (close - mean) / mean * 100.0
        adj_kairi = kairi / np.sqrt(timeframe * period)
        adj_kairi = adj_kairi.fillna(0.0)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(adj_kairi, path)
    return adj_kairi

def i_atr(symbol, timeframe, period, shift):
    '''ATRを返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 期間。
        period: 計算期間。
        shift: シフト。
    Returns:
        ATR。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_atr_' + symbol +
            str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        atr = joblib.load(path)
    # さもなければ計算する。
    else:
        high = i_high(symbol, timeframe, shift)
        low = i_low(symbol, timeframe, shift)
        close = i_close(symbol, timeframe, shift)
        temp = high - low
        temp = pd.concat([temp, high - close.shift(1)], axis=1)
        temp = pd.concat([temp, close.shift(1) - low], axis=1)
        tr = temp.max(axis=1)
        atr = pd.ewma(tr, span=period)
        atr = atr.fillna(method='ffill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(atr, path)
    return atr

def i_bands(symbol, timeframe, period, deviation, shift):
    '''ボリンジャーバンドを返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 期間。
        period: 計算期間。
        deviation: 標準偏差。
        shift: シフト。
    Returns:
        ボリンジャーバンド。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_bands_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(deviation) + '_' +
        str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        bands = joblib.load(path)
    # さもなければ計算する。
    else:
        bands = pd.DataFrame()
        close = i_close(symbol, timeframe, shift)
        mean = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        bands['upper'] = mean + deviation * std
        bands['lower'] = mean - deviation * std
        bands = bands.fillna(0.0)
        bands[(bands==float('inf')) | (bands==float('-inf'))] = 0.0
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(bands, path)
    return bands

def i_bandwalk(symbol, timeframe, period, ma_method, shift):
    '''バンドウォークを返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        ma_method: 移動平均のメソッド。
        shift: シフト。
    Returns:
        バンドウォーク。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_bandwalk_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + ma_method + '_' +
        str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        bandwalk = joblib.load(path)
    # さもなければ計算する。
    else:
        @jit(int64[:](float64[:], float64[:], float64[:], int64[:], int64),
             nopython=True, cache=True)
        def func(high, low, ma, ret, length):
            above = 0
            below = 0
            for i in range(length):
                if (low[i] > ma[i]):
                    above = above + 1
                else:
                    above = 0
                if (high[i] < ma[i]):
                    below = below + 1
                else:
                    below = 0
                ret[i] = above - below
            return ret

        high = i_high(symbol, timeframe, shift)
        low = i_low(symbol, timeframe, shift)
        close = i_close(symbol, timeframe, shift)
        if ma_method == 'MODE_SMA':
            ma = close.rolling(window=period).mean()
        elif ma_method == 'MODE_EMA':
            ma = close.ewm(span=period).mean()
        index = ma.index
        high = np.array(high)
        low = np.array(low)
        ma = np.array(ma)
        length = len(ma)
        ret = np.empty(length)
        ret = ret.astype(np.int64)
        bandwalk = func(high, low, ma, ret, length)
        bandwalk = pd.Series(bandwalk, index=index)
        bandwalk = bandwalk.fillna(0)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(bandwalk, path)
    return bandwalk

def i_bandwalk_cl(symbol, timeframe, period, ma_method, shift):
    '''バンドウォーク（終値ベース）を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        ma_method: 移動平均のメソッド。
        shift: シフト。
    Returns:
        バンドウォーク（終値ベース）。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_bandwalk_cl_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + ma_method + '_' +
        str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        bandwalk_cl = joblib.load(path)
    # さもなければ計算する。
    else:
        @jit(int64[:](float64[:], float64[:], int64[:], int64), nopython=True,
             cache=True)
        def func(log_close, log_ma, ret, length):
            above = 0
            below = 0
            for i in range(length):
                if (log_close[i] > log_ma[i]):
                    above = above + 1
                else:
                    above = 0
                if (log_close[i] < log_ma[i]):
                    below = below + 1
                else:
                    below = 0
                ret[i] = above - below
            return ret

        close = i_close(symbol, timeframe, shift)
        if ma_method == 'MODE_SMA':
            ma = close.rolling(window=period).mean()
        elif ma_method == 'MODE_EMA':
            ma = close.ewm(span=period).mean()
        index = ma.index
        close = np.array(close)
        ma = np.array(ma)
        length = len(ma)
        ret = np.empty(length)
        ret = ret.astype(np.int64)
        bandwalk_cl = func(close, ma, ret, length)
        bandwalk_cl = pd.Series(bandwalk_cl, index=index)
        bandwalk_cl = bandwalk_cl.fillna(0)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(bandwalk_cl, path)
    return bandwalk_cl

def i_opening_range(symbol, timeframe, shift):
    '''日足始値からの（対数変換した）値幅を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 期間。
        shift: シフト。
    Returns:
        日足始値からの（対数変換した）値幅。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_opening_range_' + symbol +
            str(timeframe) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        opening_range = joblib.load(path)
    # さもなければ計算する。
    else:
        log_open = i_log_open(symbol, timeframe, shift)
        log_close = i_log_close(symbol, timeframe, shift)
        index = log_open.index
        log_open[(time_hour(index)!=0) | (time_minute(index)!=0)] = np.nan
        log_open = log_open.fillna(method='ffill')
        opening_range = log_close - log_open
        opening_range = opening_range.fillna(method='ffill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(opening_range, path)
    return opening_range

def i_close(symbol, timeframe, shift):
    '''終値を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        shift: シフト。
    Returns:
        終値。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_close_' + symbol +
        str(timeframe) + '_' + str(shift) + '.pkl')
    # トレードのとき、
    if OANDA is not None:
        instrument = convert_symbol2instrument(symbol)
        granularity = convert_timeframe2granularity(timeframe)
        temp = OANDA.get_history(
            instrument=instrument, granularity=granularity, count=COUNT)
        index = pd.Series(np.zeros(COUNT))
        close = pd.Series(np.zeros(COUNT))
        for i in range(COUNT):
            index[i] = temp['candles'][i]['time']
            close[i] = temp['candles'][i]['closeBid']
            index = pd.to_datetime(index)
            close.index = index
        close = close.shift(shift)
    # バックテスト、またはウォークフォワードテストのとき、
    else:
        # 計算結果が保存されていれば復元する。
        if os.path.exists(path) == True:
            close = joblib.load(path)
        # さもなければ計算する。
        else:
            filename = ('~/historical_data/' + symbol + str(timeframe) +
                '.csv')
            temp = pd.read_csv(filename, index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            close = temp.iloc[:, 3]
            close = close.shift(shift)
            close = close.fillna(method='ffill')
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(close, path)
    return close

def i_diff(symbol, timeframe, shift):
    '''終値の階差を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        shift: シフト。
    Returns:
        終値の階差。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_diff_' + symbol +
        str(timeframe) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        diff = joblib.load(path)
    # さもなければ計算する。
    else:
        close = i_close(symbol, timeframe, shift)
        diff = close - close.shift(1)
        diff = diff.fillna(0.0)
        diff[(diff==float('inf')) | (diff==float('-inf'))] = 0.0
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(diff, path)
    return diff

def i_expansion_duration(symbol, timeframe, period, shift):
    '''エクスパンション期間（マイナスの場合はスクイーズ期間）を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
    Returns:
        エクスパンション期間（マイナスの場合はスクイーズ期間）。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_expansion_duration_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        expansion_duration = joblib.load(path)
    # さもなければ計算する。
    else:
        @jit(float64[:](float64[:]), nopython=True, cache=True)
        def func(std_dev):
            up = 0
            down = 0
            length = len(std_dev)
            ret = np.empty(length)
            ret[0] = 0
            for i in range(1, length):
                #if (std_dev[i] > std_dev[i-1]):
                if (std_dev[i] > 1.0):
                    up = up + 1
                else:
                    up = 0
                #if (std_dev[i] < std_dev[i-1]):
                if (std_dev[i] < 1.0):
                    down = down + 1
                else:
                    down = 0
                ret[i] = up - down
            return ret

        std_dev = i_zscore(symbol, timeframe, period, shift)
        #std_dev = i_std_dev(symbol, timeframe, period, shift)
        index = std_dev.index
        std_dev = np.array(std_dev)
        std_dev = np.abs(std_dev)
        expansion_duration = func(std_dev)
        expansion_duration = pd.Series(expansion_duration, index=index)
        expansion_duration = expansion_duration.fillna(0)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(expansion_duration, path)
    return expansion_duration

def i_high(symbol, timeframe, shift):
    '''高値を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        shift: シフト。
    Returns:
        高値。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_high_' + symbol +
        str(timeframe) + '_' + str(shift) + '.pkl')
    # トレードのとき、
    if OANDA is not None:
        instrument = convert_symbol2instrument(symbol)
        granularity = convert_timeframe2granularity(timeframe)
        temp = OANDA.get_history(
            instrument=instrument, granularity=granularity, count=COUNT)
        index = pd.Series(np.zeros(COUNT))
        high = pd.Series(np.zeros(COUNT))
        for i in range(COUNT):
            index[i] = temp['candles'][i]['time']
            high[i] = temp['candles'][i]['highBid']
            index = pd.to_datetime(index)
            high.index = index
        high = high.shift(shift)
    # バックテスト、またはウォークフォワードテストのとき、
    else:
        # 計算結果が保存されていれば復元する。
        if os.path.exists(path) == True:
            high = joblib.load(path)
        # さもなければ計算する。
        else:
            filename = ('~/historical_data/' + symbol + str(timeframe) +
                '.csv')
            temp = pd.read_csv(filename, index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            high = temp.iloc[:, 1]
            high = high.shift(shift)
            high = high.fillna(method='ffill')
            high = high.fillna(method='bfill')
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(high, path)
    return high

def i_highest(symbol, timeframe, period, shift):
    '''直近高値の位置を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 計算期間。
        shift: シフト。
    Returns:
        直近高値の位置。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_highest_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        highest = joblib.load(path)
    # さもなければ計算する。
    else:
        def func(high):
            period = len(high)
            argmax = high.argmax()
            ret = period - 1 - argmax
            return ret

        high = i_high(symbol, timeframe, shift)
        highest = high.rolling(window=period).apply(func)
        highest = highest.fillna(method='ffill')
        highest = highest.fillna(method='bfill')
        highest = highest.astype(int)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(highest, path)
    return highest

def i_hl_band(symbol, timeframe, period, shift):
    '''直近の高値、安値を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 計算期間。
        shift: シフト。
    Returns:
        直近の高値、安値。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_hl_band_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        hl_band = joblib.load(path)
    # さもなければ計算する。
    else:
        high = i_high(symbol, timeframe, shift)
        low = i_low(symbol, timeframe, shift)
        hl_band = pd.DataFrame()
        hl_band['high'] = high.rolling(window=period).max()
        hl_band['low'] = low.rolling(window=period).min()
        hl_band['middle'] = (hl_band['high'] + hl_band['low']) / 2
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(hl_band, path)
    return hl_band

def i_hurst(symbol, timeframe, period, shift):
    '''ハースト指数を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
    Returns:
        ハースト指数。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_hurst_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        hurst = joblib.load(path)
    # さもなければ計算する。
    else:
        # ハースト指数を計算する関数を定義する。
        def calc_hurst(close): 
            max_lag = int(np.floor(len(close) / 2))
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(close[lag:],
                                              close[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        # ハースト指数を計算する。
        close = i_close(symbol, timeframe, shift)
        hurst = close.rolling(window=period).apply(calc_hurst)
        hurst = hurst.fillna(0.0)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(hurst, path)
    return hurst

## まだ作りかけ。
#def i_kalman_filter(symbol, timeframe, period, shift):
#    '''カルマンフィルターによる予測値を返す。
#    Args:
#        symbol: 通貨ペア名。
#        timeframe: タイムフレーム。
#        period: 期間。
#        shift: シフト。
#    Returns:
#        カルマンフィルターによる予測値。
#    '''
#    # 計算結果の保存先のパスを格納する。
#    path = (os.path.dirname(__file__) + '/temp/i_kalman_filter_' + symbol +
#        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
#    # バックテスト、またはウォークフォワードテストのとき、
#    # 計算結果が保存されていれば復元する。
#    if OANDA is None and os.path.exists(path) == True:
#        kalman_filter = joblib.load(path)
#    # さもなければ計算する。
#    else:
#        # カルマンフィルターを計算する関数を定義する。
#        def calc_kalman_filter(data, period, n_iter):
#            kf = KalmanFilter()
#            kf = kf.em(data, n_iter=n_iter)
#            #smoothed_state_means = kf.smooth(data)[0]
#            #kalman_filter = smoothed_state_means[period-1, 0]
#            filtered_state_means = kf.filter(data)[0]
#            kalman_filter = filtered_state_means[period-1, 0]
#            return kalman_filter
#        # カルマンフィルターを計算する
#        close = i_close(symbol, timeframe, shift)
#        index = close.index
#        close = np.array(close)
#        # リスト内包表記も試したが、特に速度は変わらない。
#        n = len(close)
#        kalman_filter = np.empty(n)
#        for i in range(period, n):
#            data = close[i-period:i]
#            kalman_filter[i] = calc_kalman_filter(data, period, 0)
#        kalman_filter = pd.Series(kalman_filter, index=index)
#        kalman_filter = kalman_filter.fillna(method='ffill')
#        kalman_filter = kalman_filter.fillna(method='bfill')
#        # バックテスト、またはウォークフォワードテストのとき、保存する。
#        if OANDA is None:
#            # 一時フォルダーがなければ作成する。
#            make_temp_folder()
#            joblib.dump(kalman_filter, path)
#    return kalman_filter

def i_kairi(symbol, timeframe, period, shift):
    '''移動平均乖離率を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: 足の種類。
        period: 計算期間。
        shift: シフト。
    Returns:
        移動平均乖離率。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_kairi_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        kairi = joblib.load(path)
    # さもなければ計算する。
    else:
        log_close = i_log_close(symbol, timeframe, shift)
        mean = log_close.rolling(window=period).mean()
        kairi = log_close - mean
        kairi = kairi.fillna(method='ffill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(kairi, path)
    return kairi

def i_ku_bandwalk(timeframe, period, shift, aud=0, cad=0, chf=0, eur=0, gbp=0,
                  jpy=0, nzd=0, usd=0):
    '''Ku-Powerのバンドウォークを返す。
    Args:
        timeframe: 期間。
        period:計算期間。
        shift: シフト。
        aud: 豪ドル。
        cad: カナダドル。
        chf: スイスフラン。
        eur: ユーロ。
        gbp: ポンド。
        jpy: 円。
        nzd: NZドル。
        usd: 米ドル。
    Returns:
        Ku-Powerのバンドウォーク。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_ku_bandwalk_' +
            str(timeframe) + '_' + str(period) + '_' + str(shift) + '_' +
            str(aud) + str(cad) + str(chf) + str(eur) + str(gbp) + str(jpy) +
            str(nzd) + str(usd) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        ku_bandwalk = joblib.load(path)
    # さもなければ計算する。
    else:
        @jit(int64[:, :](float64[:, :], float64[:, :], int64[:, :], int64,
             int64), nopython=True, cache=True)
        def func(ku_close, ku_ma, ret, m, n):
            for j in range(n):
                above = 0
                below = 0
                for i in range(m):
                    if (ku_close[i][j] > ku_ma[i][j]):
                        above = above + 1
                    else:
                        above = 0
                    if (ku_close[i][j] < ku_ma[i][j]):
                        below = below + 1
                    else:
                        below = 0
                    ret[i][j] = above - below
            return ret
        ku_power = i_ku_power(timeframe, shift, aud=aud, cad=cad, chf=chf,
                              eur=eur, gbp=gbp, jpy=jpy, nzd=nzd, usd=usd)
        ku_ma = ku_power.rolling(window=period).mean()
        index = ku_power.index
        columns = ku_power.columns
        ku_close = np.array(ku_power)
        ku_ma = np.array(ku_ma)
        m = ku_close.shape[0]
        n = ku_close.shape[1]
        ret = np.empty([m, n])
        ret = ret.astype(int)
        ku_bandwalk = func(ku_close, ku_ma, ret, m, n)
        ku_bandwalk = pd.DataFrame(ku_bandwalk, index=index, columns=columns)
        ku_bandwalk = ku_bandwalk.fillna(0.0)
        ku_bandwalk[(ku_bandwalk==float('inf')) |
                    (ku_bandwalk==float('-inf'))] = 0.0
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(ku_bandwalk, path)
    return ku_bandwalk

def i_ku_kairi(timeframe, period, shift, aud=0, cad=0, chf=0, eur=0, gbp=0,
                jpy=0, nzd=0, usd=0):
    '''Ku-PowerのZスコアを返す。
    Args:
        timeframe: 期間。
        period:計算期間。
        shift: シフト。
        aud: 豪ドル。
        cad: カナダドル。
        chf: スイスフラン。
        eur: ユーロ。
        gbp: ポンド。
        jpy: 円。
        nzd: NZドル。
        usd: 米ドル。
    Returns:
        Ku-PowerのZスコア。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_ku_kairi_' + str(timeframe) +
        '_' + str(period) + '_' + str(shift) + '_' + str(aud) + str(cad) +
        str(chf) + str(eur) + str(gbp) + str(jpy) + str(nzd) + str(usd) +
        '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        ku_kairi = joblib.load(path)
    # さもなければ計算する。
    else:
        ku_power = i_ku_power(timeframe, shift, aud=aud, cad=cad, chf=chf,
                              eur=eur, gbp=gbp, jpy=jpy, nzd=nzd, usd=usd)
        mean = ku_power.rolling(window=period).mean()
        ku_kairi = ku_power - mean
        ku_kairi = ku_kairi.fillna(method='ffill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(ku_kairi, path)
    return ku_kairi

def i_ku_power(timeframe, shift, aud=0, cad=0, chf=0, eur=0, gbp=0, jpy=0,
               nzd=0, usd=0):
    '''Ku-Powerを返す。
    Args:
        timeframe: 期間。
        shift: シフト。
        aud: 豪ドル。
        cad: カナダドル。
        chf: スイスフラン。
        eur: ユーロ。
        gbp: ポンド。
        jpy: 円。
        nzd: NZドル。
        usd: 米ドル。
    Returns:
        Ku-Power。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_ku_power_' + str(timeframe) +
        '_' + str(shift) + '_' + str(aud) + str(cad) + str(chf) + str(eur) +
        str(gbp) + str(jpy) + str(nzd) + str(usd) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        ku_power = joblib.load(path)
    # さもなければ計算する。
    else:
        # 終値を格納する。
        audusd = 0.0
        cadusd = 0.0
        chfusd = 0.0
        eurusd = 0.0
        gbpusd = 0.0
        jpyusd = 0.0
        nzdusd = 0.0
        if aud == 1:
            audusd = i_log_close('AUDUSD', timeframe, shift)
        if cad == 1:
            cadusd = -i_log_close('USDCAD', timeframe, shift)
        if chf == 1:
            chfusd = -i_log_close('USDCHF', timeframe, shift)
        if eur == 1:
            eurusd = i_log_close('EURUSD', timeframe, shift)
        if gbp == 1:
            gbpusd = i_log_close('GBPUSD', timeframe, shift)
        if jpy == 1:
            jpyusd = -i_log_close('USDJPY', timeframe, shift)
        if nzd == 1:
            nzdusd = i_log_close('NZDUSD', timeframe, shift)
        # Ku-Powerを作成する。
        n = aud + cad + chf + eur + gbp + jpy + nzd + usd
        a = (audusd * aud + cadusd * cad + chfusd * chf + eurusd * eur +
            gbpusd * gbp + jpyusd * jpy + nzdusd * nzd) / n
        ku_power = pd.DataFrame()
        if aud == 1:
            ku_power['AUD'] = audusd - a
        if cad == 1:
            ku_power['CAD'] = cadusd - a
        if chf == 1:
            ku_power['CHF'] = chfusd - a
        if eur == 1:
            ku_power['EUR'] = eurusd - a
        if gbp == 1:
            ku_power['GBP'] = gbpusd - a
        if jpy == 1:
            ku_power['JPY'] = jpyusd - a
        if nzd == 1:
            ku_power['NZD'] = nzdusd - a
        if usd == 1:
            ku_power['USD'] = -a
        ku_power = ku_power.fillna(method='ffill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(ku_power, path)
    return ku_power

def i_ku_return(timeframe, period, shift, aud=0, cad=0, chf=0, eur=0, gbp=0,
                jpy=0, nzd=0, usd=0):
    '''Ku-Powerのリターンを返す。
    Args:
        timeframe: 期間。
        period:計算期間。
        shift: シフト。
        aud: 豪ドル。
        cad: カナダドル。
        chf: スイスフラン。
        eur: ユーロ。
        gbp: ポンド。
        jpy: 円。
        nzd: NZドル。
        usd: 米ドル。
    Returns:
        Ku-Powerのリターン。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_ku_return_' + str(timeframe) +
        '_' + str(period) + '_' + str(shift) + '_' + str(aud) + str(cad) +
        str(chf) + str(eur) + str(gbp) + str(jpy) + str(nzd) + str(usd) +
        '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        ku_return = joblib.load(path)
    # さもなければ計算する。
    else:
        ku_power = i_ku_power(timeframe, shift, aud=aud, cad=cad, chf=chf,
                              eur=eur, gbp=gbp, jpy=jpy, nzd=nzd, usd=usd)
        ku_return = ku_power - ku_power.shift(period)
        ku_return = ku_return.fillna(0.0)
        ku_return[(ku_return==float('inf')) | (ku_return==float('-inf'))] = 0.0
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(ku_return, path)
    return ku_return

def i_ku_zscore(timeframe, period, shift, aud=0, cad=0, chf=0, eur=0, gbp=0,
                jpy=0, nzd=0, usd=0):
    '''Ku-PowerのZスコアを返す。
    Args:
        timeframe: 期間。
        period:計算期間。
        shift: シフト。
        aud: 豪ドル。
        cad: カナダドル。
        chf: スイスフラン。
        eur: ユーロ。
        gbp: ポンド。
        jpy: 円。
        nzd: NZドル。
        usd: 米ドル。
    Returns:
        Ku-PowerのZスコア。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_ku_zscore_' + str(timeframe) +
        '_' + str(period) + '_' + str(shift) + '_' + str(aud) + str(cad) +
        str(chf) + str(eur) + str(gbp) + str(jpy) + str(nzd) + str(usd) +
        '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        ku_zscore = joblib.load(path)
    # さもなければ計算する。
    else:
        ku_power = i_ku_power(timeframe, shift, aud=aud, cad=cad, chf=chf,
                              eur=eur, gbp=gbp, jpy=jpy, nzd=nzd, usd=usd)
        mean = ku_power.rolling(window=period).mean()
        std = ku_power.rolling(window=period).std()
        std = std.mean(axis=1)
        ku_zscore = (ku_power - mean).div(std, axis=0)  # メモリーエラー対策。
        ku_zscore = ku_zscore.fillna(method='ffill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(ku_zscore, path)
    return ku_zscore

def i_kurt(symbol, timeframe, period, shift):
    '''対数リターンの尖度を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
    Returns:
        対数リターンの尖度。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_kurt_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        kurt = joblib.load(path)
    # さもなければ計算する。
    else:
        log_return = i_log_return(symbol, timeframe, period, shift)
        kurt = log_return.rolling(window=period).kurt()
        kurt = kurt.fillna(method='ffill')
        kurt = kurt.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(kurt, path)
    return kurt

def i_log_close(symbol, timeframe, shift):
    '''対数変換した終値を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: 足の種類。
        shift: シフト。
    Returns:
        対数変換した終値。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_log_close_' + symbol +
        str(timeframe) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        log_close = joblib.load(path)
    # さもなければ計算する。
    else:
        close = i_close(symbol, timeframe, shift)
        log_close = np.log(close)
        log_close = log_close.fillna(method='ffill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(log_close, path)
    return log_close

def i_log_high(symbol, timeframe, shift):
    '''対数変換した高値を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: 足の種類。
        shift: シフト。
    Returns:
        対数変換した高値。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_log_high_' + symbol +
        str(timeframe) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        log_high = joblib.load(path)
    # さもなければ計算する。
    else:
        high = i_high(symbol, timeframe, shift)
        log_high = np.log(high)
        log_high = log_high.fillna(method='ffill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(log_high, path)
    return log_high

def i_log_low(symbol, timeframe, shift):
    '''対数変換した安値を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: 足の種類。
        shift: シフト。
    Returns:
        対数変換した安値。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_log_low_' + symbol +
        str(timeframe) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        log_low = joblib.load(path)
    # さもなければ計算する。
    else:
        low = i_low(symbol, timeframe, shift)
        log_low = np.log(low)
        log_low = log_low.fillna(method='ffill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(log_low, path)
    return log_low

def i_log_ma(symbol, timeframe, period, shift):
    '''対数変換した移動平均を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: 足の種類。
        period: 計算期間。
        shift: シフト。
    Returns:
        対数変換した移動平均。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_log_ma_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        log_ma = joblib.load(path)
    # さもなければ計算する。
    else:
        log_close = i_log_close(symbol, timeframe, shift)
        log_ma = log_close.rolling(window=period).mean()
        log_ma = log_ma.fillna(method='ffill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(log_ma, path)
    return log_ma

def i_log_open(symbol, timeframe, shift):
    '''対数変換した始値を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: 足の種類。
        shift: シフト。
    Returns:
        対数変換した始値。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_log_open_' + symbol +
        str(timeframe) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        log_open = joblib.load(path)
    # さもなければ計算する。
    else:
        op = i_open(symbol, timeframe, shift)
        log_open = np.log(op)
        log_open = log_open.fillna(method='ffill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(log_open, path)
    return log_open

def i_log_return(symbol, timeframe, period, shift):
    '''対数変換したリターンを返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: 足の種類。
        period: 計算期間。
        shift: シフト。
    Returns:
        対数変換したリターン。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_log_return_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        log_return = joblib.load(path)
    # さもなければ計算する。
    else:
        log_close = i_log_close(symbol, timeframe, shift)
        log_return = log_close - log_close.shift(period)
        log_return = log_return.fillna(method='ffill')
        log_return[(np.isnan(log_return)) | (log_return==float("inf")) |
                (log_return==float("-inf"))] = 0.0
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(log_return, path)
    return log_return

def i_low(symbol, timeframe, shift):
    '''安値を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        shift: シフト。
    Returns:
        安値。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_low_' + symbol +
        str(timeframe) + '_' + str(shift) + '.pkl')
    # トレードのとき、
    if OANDA is not None:
        instrument = convert_symbol2instrument(symbol)
        granularity = convert_timeframe2granularity(timeframe)
        temp = OANDA.get_history(
            instrument=instrument, granularity=granularity, count=COUNT)
        index = pd.Series(np.zeros(COUNT))
        low = pd.Series(np.zeros(COUNT))
        for i in range(COUNT):
            index[i] = temp['candles'][i]['time']
            low[i] = temp['candles'][i]['lowBid']
            index = pd.to_datetime(index)
            low.index = index
        low = low.shift(shift)
    # バックテスト、またはウォークフォワードテストのとき、
    else:
        # 計算結果が保存されていれば復元する。
        if os.path.exists(path) == True:
            low = joblib.load(path)
        # さもなければ計算する。
        else:
            filename = ('~/historical_data/' + symbol + str(timeframe) +
                '.csv')
            temp = pd.read_csv(filename, index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            low = temp.iloc[:, 2]
            low = low.shift(shift)
            low = low.fillna(method='ffill')
            low = low.fillna(method='bfill')
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(low, path)
    return low

def i_lowest(symbol, timeframe, period, shift):
    '''直近安値の位置を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 計算期間。
        shift: シフト。
    Returns:
        直近安値の位置。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_lowest_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        lowest = joblib.load(path)
    # さもなければ計算する。
    else:
        def func(low):
            period = len(low)
            argmin = low.argmin()
            ret = period - 1 - argmin
            return ret

        low = i_low(symbol, timeframe, shift)
        lowest = low.rolling(window=period).apply(func)
        lowest = lowest.fillna(method='ffill')
        lowest = lowest.fillna(method='bfill')
        lowest = lowest.astype(int)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(lowest, path)
    return lowest

def i_ma(symbol, timeframe, period, shift):
    '''移動平均を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
    Returns:
        移動平均。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_ma_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        ma = joblib.load(path)
    # さもなければ計算する。
    else:
        close = i_close(symbol, timeframe, shift)
        ma = close.rolling(window=period).mean()
        ma = ma.fillna(method='ffill')
        ma = ma.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(ma, path)
    return ma

def i_ma_of_ma(symbol, timeframe, period, shift):
    '''移動平均の移動平均を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
    Returns:
        移動平均の移動平均。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_ma_of_ma_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        ma_of_ma = joblib.load(path)
    # さもなければ計算する。
    else:
        ma = i_ma(symbol, timeframe, period, shift)
        ma_of_ma = ma.rolling(window=period).mean()
        ma_of_ma = ma_of_ma.fillna(method='ffill')
        ma_of_ma = ma_of_ma.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(ma_of_ma, path)
    return ma_of_ma

def i_mean(symbol, timeframe, period, shift):
    '''対数リターンの平均を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 期間。
        period: 計算期間。
        shift: シフト。
    Returns:
        対数リターンの平均。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_mean_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        mean = joblib.load(path)
    # さもなければ計算する。
    else:
        log_return = i_log_return(symbol, timeframe, period, shift)
        mean = log_return.rolling(window=period).mean()
        mean = mean.fillna(method='ffill')
        mean = mean.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(mean, path)
    return mean

def i_momentum(symbol, timeframe, period, shift):
    '''モメンタムを返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 期間。
        period: 計算期間。
        shift: シフト。
    Returns:
        モメンタム。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_momentum_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        momentum = joblib.load(path)
    # さもなければ計算する。
    else:
        close = i_close(symbol, timeframe, shift)
        momentum = (close / close.shift(period)) * 100.0
        momentum = momentum.fillna(method='ffill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(momentum, path)
    return momentum

def i_open(symbol, timeframe, shift):
    '''始値を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 期間。
        shift: シフト。
    Returns:
        始値。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_open_' + symbol +
        str(timeframe) + '_' + str(shift) + '.pkl')
    # トレードのとき、
    if OANDA is not None:
        instrument = convert_symbol2instrument(symbol)
        granularity = convert_timeframe2granularity(timeframe)
        temp = OANDA.get_history(
            instrument=instrument, granularity=granularity, count=COUNT)
        index = pd.Series(np.zeros(COUNT))
        op = pd.Series(np.zeros(COUNT))
        for i in range(COUNT):
            index[i] = temp['candles'][i]['time']
            op[i] = temp['candles'][i]['openBid']
            index = pd.to_datetime(index)
            op.index = index
        op = op.shift(shift)
    # バックテスト、またはウォークフォワードテストのとき、
    else:
        # 計算結果が保存されていれば復元する。
        if os.path.exists(path) == True:
            op = joblib.load(path)
        # さもなければ計算する。
        else:
            filename = ('~/historical_data/' + symbol + str(timeframe) +
                '.csv')
            temp = pd.read_csv(filename, index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            op = temp.iloc[:, 0]
            op = op.shift(shift)
            op = op.fillna(method='ffill')
            op = op.fillna(method='bfill')
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(op, path)
    return op

def i_range_ratio(symbol, timeframe, period, shift):
    '''レンジの上、上と中の間、中と下の間、下を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
    Returns:
        レンジの上、上と中の間、中と下の間、下。
        2: 上。　1: 上と中の間。　-1: 中と下の間。　-2: 下。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_range_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        range_ratio = joblib.load(path)
    # さもなければ計算する。
    else:
        a = 1.59091590087
        b = -0.143851282174
        log_return1 = i_log_return(symbol, timeframe, 1, shift)
        std1 = log_return1.rolling(window=period).std()
        range_width = (a * np.sqrt(period) + b) * std1
        hl_band = i_hl_band(symbol, timeframe, period, shift)
        hl_band = np.log(hl_band)
        actual_range_width = hl_band['high'] - hl_band['low']
        range_ratio = actual_range_width / range_width
        range_ratio = range_ratio.fillna(method='ffill')
        range_ratio = range_ratio.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(range_ratio, path)
    return range_ratio

def i_range_duration(symbol, timeframe, period, shift):
    '''正規化したレンジ期間を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
    Returns:
        正規化したレンジ期間。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_range_duration_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        range_duration = joblib.load(path)
    # さもなければ計算する。
    else:
        @jit(float64[:](float64[:]), nopython=True, cache=True)
        def func(rng):
            count = 0
            length = len(rng)
            range_duration = np.zeros(length)
            for i in range(length):
                if (rng[i] < 1.0):
                    count = count + 1
                else:
                    count = 0
                range_duration[i] = count
            return range_duration

        log_return1 = i_log_return(symbol, timeframe, 1, shift)
        std = log_return1.rolling(window=period).std()
        log_return_period = i_log_return(symbol, timeframe, period, shift)
        rng = np.abs(log_return_period) / (std * np.sqrt(period))
        index = rng.index
        rng = np.array(rng)
        range_duration = func(rng)
        range_duration = pd.Series(range_duration, index=index)
        a = 0.530947805031
        b = 1.38462529993
        range_duration = range_duration / (a * np.sqrt(period) + b)
        range_duration = range_duration.fillna(0.0)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(range_duration, path)
    return range_duration

def i_skew(symbol, timeframe, period, shift):
    '''対数リターンの歪度を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
    Returns:
        対数リターンの歪度。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_skew_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        skew = joblib.load(path)
    # さもなければ計算する。
    else:
        log_return = i_log_return(symbol, timeframe, period, shift)
        skew = log_return.rolling(window=period).skew()
        skew = skew.fillna(method='ffill')
        skew = skew.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(skew, path)
    return skew

def i_std(symbol, timeframe, period, shift):
    '''対数リターンの標準偏差を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
    Returns:
        対数リターンの標準偏差。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_std_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        std = joblib.load(path)
    # さもなければ計算する。
    else:
        log_return = i_log_return(symbol, timeframe, period, shift)
        std = log_return.rolling(window=period).std()
        std = std.fillna(method='ffill')
        std = std.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(std, path)
    return std

def i_std_dev(symbol, timeframe, period, shift):
    '''# 標準偏差を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
    Returns:
        # 標準偏差。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_std_dev_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        std_dev = joblib.load(path)
    # さもなければ計算する。
    else:
        close = i_close(symbol, timeframe, shift)
        std_dev = close.rolling(window=period).std()
        std_dev = std_dev.fillna(method='ffill')
        std_dev = std_dev.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(std_dev, path)
    return std_dev

def i_stop_hunting_zone(symbol, timeframe, period, max_iter, pip, shift):
    '''ストップ狩りのゾーンにあるか否かを返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 計算期間。
        max_iter: 最大繰り返し数。
        pip: pip（ただし、MT4に準じて1=0.1pipとする）。
        shift: シフト。
    Returns:
        ストップ狩りのゾーンにあるか否か。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_stop_hunting_zone_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(max_iter) + '_' +
        str(pip) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        stop_hunting_zone = joblib.load(path)
    # さもなければ計算する。
    else:
        if (symbol == 'AUDJPY' or symbol == 'CADJPY' or
            symbol == 'CHFJPY' or symbol == 'EURJPY' or
            symbol == 'GBPJPY' or symbol == 'NZDJPY' or
            symbol == 'USDJPY'):
            multiplier = 0.001
        else:
            multiplier = 0.00001
        close = i_close(symbol, timeframe, shift)
        stop_hunting_zone = pd.DataFrame(index=close.index)
        for i in range(max_iter):
            hl_band = i_hl_band(symbol, timeframe, period*(i+1), shift)
            resistance = pd.Series(index=close.index)
            support = pd.Series(index=close.index)
            resistance[(close >= (hl_band['high'] - (pip * multiplier))) &
                (close <= hl_band['high'])] = 1
            support[(close <= (hl_band['low'] + (pip * multiplier))) &
                (close >= hl_band['low'])] = 1
            resistance = resistance.fillna(0)
            support = support.fillna(0)
            if i == 0:
                stop_hunting_zone['resistance'] = resistance
                stop_hunting_zone['support'] = support
            else:
                stop_hunting_zone['resistance'] += resistance
                stop_hunting_zone['support'] += support
        stop_hunting_zone = stop_hunting_zone.fillna(0)
        stop_hunting_zone[stop_hunting_zone>=1] = 1
        stop_hunting_zone = stop_hunting_zone.astype(bool)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(stop_hunting_zone, path)
    return stop_hunting_zone

def i_strength(timeframe, period, shift, aud=1, cad=0, chf=0, eur=1, gbp=1,
               jpy=1, nzd=0, usd=1):
    '''通貨の強さを返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 期間。
        period: 計算期間。
        shift: シフト。
        aud: 豪ドルの設定。
        cad: カナダドルの設定。
        chf: スイスフランの設定。
        eur: ユーロの設定。
        gbp: ポンドの設定。
        jpy: 円の設定。
        nzd: NZドルの設定。
        usd: 米ドルの設定。
    Returns:
        通貨の強さ。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_strength_' + str(timeframe) +
            '_' + str(period) + '_' + str(shift) + str(aud) + str(cad) +
            str(chf) + str(eur) + str(gbp) + str(jpy) + str(nzd) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        strength = joblib.load(path)
    # さもなければ計算する。
    else:
        temp = pd.DataFrame()
        n = 0
        if aud == 1:
            temp['AUD'] = i_zscore('AUDUSD', timeframe, period, shift)
            n += 1
        if cad == 1:
            temp['CAD'] = -i_zscore('USDCAD', timeframe, period, shift)
            n += 1
        if chf == 1:
            temp['CHF'] = -i_zscore('USDCHF', timeframe, period, shift)
            n += 1
        if eur == 1:
            temp['EUR'] = i_zscore('EURUSD', timeframe, period, shift)
            n += 1
        if gbp == 1:
            temp['GBP'] = i_zscore('GBPUSD', timeframe, period, shift)
            n += 1
        if jpy == 1:
            temp['JPY'] = -i_zscore('USDJPY', timeframe, period, shift)
            n += 1
        if nzd == 1:
            temp['NZD'] = i_zscore('NZDUSD', timeframe, period, shift)
            n += 1
        if usd == 1:
            temp['USD'] = pd.Series(np.zeros(len(temp)), index=temp.index)
            n += 1
        # 同値になることはほとんどないと思うが、その場合は観測順にしている点に注意。
        strength = temp.rank(axis=1, method='first')
        data = np.array(range(1, n+1))
        mean = np.mean(data)
        std = np.std(data)
        strength = (strength - mean) / std
        strength = strength.fillna(method='ffill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(strength, path)
    return strength

def i_trend(symbol, timeframe, period, shift):
    '''トレンドを返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 期間。
        period: 計算期間。
        shift: シフト。
    Returns:
        トレンド。　1:正のトレンド。　0:トレンドなし。　-1:負のトレンド。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_trend_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        trend = joblib.load(path)
    # さもなければ計算する。
    else:
        log_return1 = i_log_return(symbol, timeframe, 1, shift)
        log_return_period = i_log_return(symbol, timeframe, period, shift)
        std = log_return1.rolling(window=period).std()
        trend = log_return_period / (std * np.sqrt(period))
        trend = trend.fillna(0.0)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(trend, path)
    return trend

def i_trend_duration(symbol, timeframe, period, shift):
    '''正規化したトレンド期間を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 期間。
        period: 計算期間。
        shift: シフト。
    Returns:
        正規化したトレンド期間。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_trend_duration_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        trend_duration = joblib.load(path)
    # さもなければ計算する。
    else:
        @jit(float64[:](float64[:]), nopython=True, cache=True)
        def func(rng):
            count = 0
            length = len(rng)
            trend_duration = np.zeros(length)
            for i in range(length):
                if (rng[i] > 1.0):
                    count = count + 1
                else:
                    count = 0
                trend_duration[i] = count
            return trend_duration

        log_return1 = i_log_return(symbol, timeframe, 1, shift)
        std = log_return1.rolling(window=period).std()
        log_return_period = i_log_return(symbol, timeframe, period, shift)
        rng = np.abs(log_return_period) / (std * np.sqrt(period))
        index = rng.index
        rng = np.array(rng)
        trend_duration = func(rng)
        trend_duration = pd.Series(trend_duration, index=index)
        a = 0.225218065625
        b = 0.91475321662
        trend_duration = trend_duration / (a * np.sqrt(period) + b)
        trend_duration = trend_duration.fillna(0.0)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(trend_duration, path)
    return trend_duration

def i_updown(symbol, timeframe, shift):
    '''騰落を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 期間。
        shift: シフト。
    Returns:
        騰落。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_updown_' + symbol +
        str(timeframe) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        updown = joblib.load(path)
    # さもなければ計算する。
    else:
        close = i_close(symbol, timeframe, shift)
        updown = close >= close.shift(1)
        updown = updown.fillna(method='ffill')
        updown = updown.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(updown, path)
    return updown

def i_vix4fx(symbol, timeframe, shift):
    '''1ヶ月当たりのボラティリティの予測値を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 期間。
        shift: シフト。
    Returns:
        1ヶ月当たりのボラティリティの予測値。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_vix4fx_' + symbol +
        str(timeframe) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        vix4fx = joblib.load(path)
    # さもなければ計算する。
    else:
        # 365 / 12 * 5 / 7 * 1440 = 31285.71428571429
        maturity = int(31285.71428571429 / timeframe)
        close = i_close(symbol, timeframe, shift)
        change = np.log(close / close.shift(1))
        vix4fx = (change.rolling(window=maturity).std() *
            np.sqrt(maturity) * 100.0)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(vix4fx, path)
    return vix4fx

def i_volatility(symbol, timeframe, period, shift):
    '''ボラティリティを返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 期間。
        period: 計算期間。
        shift: シフト。
    Returns:
        ボラティリティ。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_volatility_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        volatility = joblib.load(path)
    # さもなければ計算する。
    else:
        log_return = i_log_return(symbol, timeframe, 1, shift)
        mean = log_return.rolling(window=period).mean()
        std = log_return.rolling(window=period).std()
        volatility = (log_return - mean) / std
        volatility = volatility.fillna(method='ffill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(volatility, path)
    return volatility

def i_volume(symbol, timeframe, shift):
    '''出来高を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        shift: シフト。
    Returns:
        出来高。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_volume_' + symbol +
        str(timeframe) + '_' + str(shift) + '.pkl')
    # トレードのとき、
    if OANDA is None:
        instrument = convert_symbol2instrument(symbol)
        granularity = convert_timeframe2granularity(timeframe)
        temp = OANDA.get_history(
            instrument=instrument, granularity=granularity, count=COUNT)
        index = pd.Series(np.zeros(COUNT))
        volume = pd.Series(np.zeros(COUNT))
        for i in range(COUNT):
            index[i] = temp['candles'][i]['time']
            volume[i] = temp['candles'][i]['volumeBid']
            index = pd.to_datetime(index)
            volume.index = index
        volume = volume.shift(shift)
    # バックテスト、またはウォークフォワードテストのとき、
    else:
        # 計算結果が保存されていれば復元する。
        if os.path.exists(path) == True:
            volume = joblib.load(path)
        # さもなければ計算する。
        else:
            filename = ('~/historical_data/' + symbol + str(timeframe) +
                '.csv')
            temp = pd.read_csv( filename, index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            volume = temp.iloc[:, 4]
            volume = volume.shift(shift)
            volume = volume.fillna(method='ffill')
            volume = volume.fillna(method='bfill')
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(volume, path)
    return volume

def i_zscore(symbol, timeframe, period, ma_method, shift):
    '''終値のzスコアを返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: 足の種類。
        period: 計算期間。
        ma_method: 移動平均のメソッド。
        shift: シフト。
    Returns:
        終値のzスコア。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_zscore_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + ma_method + '_' +
        str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        zscore = joblib.load(path)
    # さもなければ計算する。
    else:
        close = i_close(symbol, timeframe, shift)
        if ma_method == 'MODE_SMA':
            mean = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
        elif ma_method == 'MODE_EMA':
            mean = close.ewm(span=period).mean()
            std = close.ewm(span=period).std()
        zscore = (close - mean) / std
        zscore = zscore.fillna(method='ffill')
        zscore[(zscore==float('inf')) | (zscore==float('-inf'))] = 0.0
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(zscore, path)
    return zscore

def i_zresid(symbol, timeframe, period, method, shift, exp1=None, exp2=None,
             exp3=None, exp4=None, exp5=None, exp6=None, exp7=None):
    '''終値とその予測値との標準化残差を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        method: メソッド。
        shift: シフト。
        exp1-exp7: 説明変数となる通貨ペア（必ずexp1から指定する）。
    Returns:
        終値とその予測値との標準化残差。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/temp/i_zresid_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + method + '_' + str(shift) +
        '_' + str(exp1) + '_' + str(exp2) + '_' + str(exp3) + '_' + str(exp4) +
        '_' + str(exp5) + '_' + str(exp6) + '_' + str(exp7) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        zresid = joblib.load(path)
    # さもなければ計算する。
    else:
        def func(x, y, period, clf):
            clf.fit(x, y)
            pred = clf.predict(x)
            se = np.std(y - pred)
            zresid = (y[period-1] - pred[period-1]) / se
            return zresid

        close = i_close(symbol, timeframe, shift)
        index = close.index
        close = np.array(close)
        n_exp = (((exp1 is not None) + (exp2 is not None) + (exp3 is not None) +
            (exp4 is not None) + (exp5 is not None) + (exp6 is not None) +
            (exp7 is not None)) * 1)
        if n_exp == 0:
            exp = np.array(range(len(close)))
            exp = exp.reshape(len(exp), 1)
        else:
            exp = i_close(exp1, timeframe, shift)
            if exp2 is not None:
                temp = i_close(exp2, timeframe, shift)
                exp = pd.concat([exp, temp], axis=1)
            if exp3 is not None:
                temp = i_close(exp3, timeframe, shift)
                exp = pd.concat([exp, temp], axis=1)
            if exp4 is not None:
                temp = i_close(exp4, timeframe, shift)
                exp = pd.concat([exp, temp], axis=1)
            if exp5 is not None:
                temp = i_close(exp5, timeframe, shift)
                exp = pd.concat([exp, temp], axis=1)
            if exp6 is not None:
                temp = i_close(exp6, timeframe, shift)
                exp = pd.concat([exp, temp], axis=1)
            if exp7 is not None:
                temp = i_close(exp7, timeframe, shift)
                exp = pd.concat([exp, temp], axis=1)
            exp = np.array(exp)
            if n_exp == 1:
                exp = exp.reshape(len(exp), 1)
        n = len(close)
        zresid = np.empty(n)
        if method == 'linear':
            clf = linear_model.LinearRegression()
        elif method == 'tree':
            max_depth = 3
            clf = tree.DecisionTreeRegressor(max_depth=max_depth)
        for i in range(period, n):
            x = exp[i-period+1:i+1]
            y = close[i-period+1:i+1]
            zresid[i] = func(x, y, period, clf)
        zresid = pd.Series(zresid, index=index)
        zresid = zresid.fillna(0.0)
        zresid[(zresid==float('inf')) | (zresid==float('-inf'))] = 0.0
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            make_temp_folder()
            joblib.dump(zresid, path)
    return zresid

def make_temp_folder():
    '''一時フォルダを作成する。
    '''
    pathname = os.path.dirname(__file__)
    if os.path.exists(pathname + '/temp') == False:
        os.mkdir(pathname + '/temp')

def minute():
    '''現在の分を返す。
    Returns:
        現在の分。
    '''
    minute = datetime.now().minute
    return minute

def optimize_params(rranges, strategy, symbol, timeframe, start, end, spread,
                    position, min_trade):
    '''パラメーターを最適化する。
    Args:
        rranges: パラメーターのスタート、ストップ、ステップ。
        strategy: 戦略。
        symbol: 通貨ペア。
        timeframe: 足の種類。
        start: 開始日。
        end: 終了日。
        spread: スプレッド。
        position: ポジションの設定。
            0: 買いのみ。
            1: 売りのみ。
            2: 売買両方。
        min_trade: 最低トレード数。
    Returns:
        パラメーター。
    '''
    def func(parameter, strategy, symbol, timeframe, start, end, spread,
             position, min_trade):
        # パフォーマンスを計算する。
        signal = strategy(parameter, symbol, timeframe)
        ret = calc_ret(symbol, timeframe, signal, spread, position, start, end)
        trades = calc_trades(signal, position, start, end)
        sharpe = calc_sharpe(ret, start, end)
        years = (end - start).total_seconds() / 60 / 60 / 24 / 365
        # 1年当たりのトレード数が最低トレード数に満たない場合、
        # 適応度を0にする。
        if trades / years >= min_trade:
            fitness = sharpe
        else:
            fitness = 0.0
        return -fitness

    parameter = optimize.brute(
        func, rranges, args=(strategy, symbol, timeframe, start, end, spread,
        position, min_trade), finish=None)
    return parameter

def order_close(ticket):
    '''決済注文を送信する。
    Args:
        ticket: チケット番号。
    '''
    OANDA.close_trade(ACCOUNT_ID, ticket)
 
def order_send(symbol, lots, side):
    '''新規注文を送信する。
    Args:
        symbol: 通貨ペア名。
        lots: ロット数。
        side: 売買の種別
    Returns:
        チケット番号。
    '''
    # 通貨ペアの名称を変換する。
    instrument = convert_symbol2instrument(symbol)
    response = OANDA.create_order(account_id=ACCOUNT_ID, instrument=instrument,
        units=int(lots*10000), side=side, type='market')
    ticket = response['tradeOpened']['id']
    return ticket

def orders_total():
    '''オーダー総数を計算する。
    Returns:
        オーダー総数。
    '''
    positions = OANDA.get_positions(ACCOUNT_ID)
    total = len(positions['positions'])
    return total

def remove_temp_folder():
    '''一時フォルダを削除する。
    '''
    pathname = os.path.dirname(__file__)
    if os.path.exists(pathname + '/temp') == True:
        shutil.rmtree(pathname + '/temp')

def save_model(model, filename):
    pathname = os.path.dirname(__file__) + '/' + filename
    if os.path.exists(pathname) == False:
        os.mkdir(pathname)
    joblib.dump(model, pathname + '/' + filename + '.pkl') 

def seconds():
    '''現在の秒を返す。
    Returns:
        現在の秒。
    '''    
    seconds = datetime.now().second
    return seconds

def send_mail(subject, some_text, fromaddr, toaddr, host, port, password):
    '''メールを送信する。
    Args:
        subject: タイトル。
        some_text: 文。
    '''
    msg = MIMEText(some_text)
    msg['Subject'] = subject
    msg['From'] = fromaddr
    msg['To'] = toaddr
    s = smtplib.SMTP_SSL(host, port)
    s.login(fromaddr, password)
    s.send_message(msg)
    s.quit()

def send_signal2mt4(filename, signal):
    '''シグナルをMT4に送信する。
    Args:
        filename: ファイル名。
        signal: シグナル。
    '''
    f = open(filename, 'w')
    # 0を使いたくないので2を加える。
    # MT4側で2を減じて調整する。
    f.write(str(int(signal.iloc[len(signal)-1] + 2)))
    f.close()

def time_day(index):
    '''日を返す。
    Args:
        index: インデックス。
    Returns:
        日。
    '''
    time_day = pd.Series(index.day, index=index)
    return time_day

def time_day_of_week(index):
    '''曜日を返す。
    Args:
        index: インデックス。
    Returns:
        曜日（0-6（日-土））。
    '''
    # pandasのdayofweekは月-日、0-6なので、MQL4に合わせて調整する。
    time_day_of_week = pd.Series(index.dayofweek, index=index) + 1
    time_day_of_week[time_day_of_week==7] = 0
    return time_day_of_week

def time_hour(index):
    '''時を返す。
    Args:
        index: インデックス。
    Returns:
        時。
    '''
    time_hour = pd.Series(index.hour, index=index)
    return time_hour
 
def time_minute(index):
    '''分を返す。
    Args:
        index: インデックス。
    Returns:
        分。
    '''
    time_minute = pd.Series(index.minute, index=index)
    return time_minute

def time_month(index):
    '''月を返す。
    Args:
        index: インデックス。
    Returns:
        月。
    '''
    time_month = pd.Series(index.month, index=index)
    return time_month

def time_market_hours(index, market):
    '''指定した市場の時間であるか否かを返す（30分足以下で使用すること）。
    Args:
        index: インデックス。
        market: 市場
    Returns:
        指定した市場の時間であるか否か。
    '''
    # 東京時間の場合。
    if market == 'tokyo':
        market_hours = ((index.hour>=2) & (index.hour<8)) * 1
        market_hours = pd.Series(market_hours, index=index)
        # 夏時間の大まかな調整。
        market_hours[(market_hours.index.month>=3)
            & (market_hours.index.month<11)] = (((index.hour>=3)
            & (index.hour<9))) * 1
    # ロンドン時間の場合。
    elif market == 'london':
        market_hours = ((index.hour>=10) & (index.hour<18)) * 1
        market_hours = pd.Series(market_hours, index=index)
        market_hours[(market_hours.index.hour==18)
            & (market_hours.index.minute<30)] = 1
    # NY時間の場合。
    elif market == 'ny':
        market_hours = ((index.hour>=17) & (index.hour<23)) * 1
        market_hours = pd.Series(market_hours, index=index)
        market_hours[(market_hours.index.hour==16)
            & (market_hours.index.minute>=30)] = 1
    return market_hours

def time_week(index):
    '''週を返す。
       ただし、月の第n週という意味ではなく、月のn番目の曜日という意味。
    Args:
        index: インデックス。
    Returns:
        週（0-4）。
    '''
    day = time_day(index)
    time_week = (np.ceil(day / 7)).astype(int)
    return time_week

def trade(mail, mt4, ea, symbol, timeframe, position, lots, ml, start_train,
          end_train):
    '''トレードを行う。
    Args:
        mail: メールの設定。
            0: メールへのシグナル配信なし。
            1: メールへのシグナル配信あり。
        mt4: MT4の設定。
            0: MT4へのシグナル配信なし。
            1: MT4へのシグナル配信あり。
        ea: EA。
        symbol: 通貨ペア。
        timeframe: 足の種類。
        position: ポジションの設定。
            0: 買いのみ。
            1: 売りのみ。
            2: 売買両方。
        lots: ロット数。
        ml: 機械学習の設定。
            0: 機械学習を使用しない。
            1:機械学習を使用する。
        start_train: 学習期間の開始日。
        end_train: 学習期間の終了日。
    '''
    # グローバル変数を設定する。
    global LOCK
    global OANDA
    global ENVIRONMENT
    global ACCESS_TOKEN
    global ACCOUNT_ID
    # 設定を格納する。
    exec('import ' + ea + ' as ea1')  # 名前が被らないよう、とりあえず「ea1」とする。
    strategy = eval('ea1.strategy')
    parameter = eval('ea1.PARAMETER')
    # 設定を格納する。
    exec('import ' + ea + ' as ea_file')
    strategy = eval('ea_file.strategy')
    parameter = eval('ea_file.PARAMETER')
    # 設定を格納する。
    path = os.path.dirname(__file__)
    config = configparser.ConfigParser()
    config.read(path + '/settings.ini')
    # メールを設定する。
    if mail == 1:
        host = 'smtp.mail.yahoo.co.jp'
        port = 465
        fromaddr = config['DEFAULT']['fromaddr']
        toaddr = config['DEFAULT']['toaddr']
        password = config['DEFAULT']['password']
    # 機械学習を使用する場合、モデル、学習期間での予測値の標準偏差を格納する。
    if ml == 1:
        build_model = eval('ea1.build_model')
        model, pred_train_std = build_model(parameter, symbol, timeframe,
                                            start_train, end_train)
    # OANDA API用に設定する。
    if ENVIRONMENT is None:
        ENVIRONMENT = config['DEFAULT']['environment']
    if ACCESS_TOKEN is None:
        ACCESS_TOKEN = config['DEFAULT']['access_token']
    if ACCOUNT_ID is None:
        ACCOUNT_ID = config['DEFAULT']['account_id']
    if OANDA is None:
        OANDA = oandapy.API(environment=ENVIRONMENT, access_token=ACCESS_TOKEN)
    second_before = 0
    count = COUNT
    pre_history_time = None
    instrument = convert_symbol2instrument(symbol)
    granularity = convert_timeframe2granularity(timeframe)
    # MT4用に設定する。
    if mt4 == 1:
        folder_ea = config['DEFAULT']['folder_ea']
        filename = folder_ea + '/' + ea + '.csv'
        f = open(filename, 'w')
        f.write(str(2))  # EA側で-2とするので2で初期化する。
        f.close()
    # トレードを行う。
    pos = 0
    ticket = 0
    while True:
        # 回線接続を確認してから実行する。
        try:
            # ロックを獲得する。
            LOCK.acquire()
            # 回線が接続していないと約15分でエラーになる。
            second_now = seconds()
            # 毎秒実行する。
            if second_now != second_before:
                second_before = second_now
                history = OANDA.get_history(
                    instrument=instrument, granularity=granularity,
                    count=count)
                history_time = history['candles'][count-1]['time']
                # 過去のデータの更新が完了してから実行する。
                # シグナルの計算は過去のデータを用いているので、過去のデータの
                # 更新が完了してから実行しないと計算がおかしくなる。
                if history_time != pre_history_time:
                    pre_history_time = history_time
                    if ml == 0:
                        signal = strategy(parameter, symbol, timeframe)
                    elif ml == 1:
                        signal = strategy(parameter, symbol, timeframe, model,
                                          pred_train_std)
                    if position == 0:
                        signal[signal==-1] = 0
                    elif position == 1:
                        signal[signal==1] = 0
                    end_row = len(signal) - 1
                    open0 = i_open(symbol, timeframe, 0)
                    price = open0[len(open0)-1]
                    # エグジット→エントリーの順に設定しないとドテンができない。
                    # 買いエグジット。
                    if (pos == 1 and signal.iloc[end_row] != 1):
                        pos = 0
                        order_close(ticket)
                        ticket = 0
                        # メールにシグナルを送信する場合
                        if mail == 1:
                            subject = ea
                            some_text = (symbol + 'を' + str(price) +
                                'で買いエグジットです。')
                            send_mail(subject, some_text, fromaddr, toaddr,
                                      host, port, password)
                        # EAにシグナルを送信する場合
                        if mt4 == 1:
                            send_signal2mt4(filename, signal)
                    # 売りエグジット
                    elif (pos == -1 and signal[end_row] != -1):
                        pos = 0
                        order_close(ticket)
                        ticket = 0
                        # メールにシグナルを送信する場合
                        if mail == 1:
                            subject = ea
                            some_text = (symbol + 'を' + str(price) +
                                '+スプレッドで売りエグジットです。')
                            send_mail(subject, some_text, fromaddr, toaddr,
                                      host, port, password)
                        # EAにシグナルを送信する場合
                        if mt4 == 1:
                            send_signal2mt4(filename, signal)
                    # 買いエントリー
                    elif (pos == 0 and signal[end_row] == 1):
                        pos = 1
                        ticket = order_send(symbol, lots, 'buy')
                        # メールにシグナルを送信する場合
                        if mail == 1:
                            subject = ea
                            some_text = (symbol + 'を' + str(price) +
                                '+スプレッドで買いエントリーです。')
                            send_mail(subject, some_text, fromaddr, toaddr,
                                      host, port, password)
                        # EAにシグナルを送信する場合
                        if mt4 == 1:
                            send_signal2mt4(filename, signal)
                    # 売りエントリー
                    elif (pos == 0 and signal[end_row] == -1):
                        pos = -1
                        ticket = order_send(symbol, lots, 'sell')
                        # メールにシグナルを送信する場合
                        if mail == 1:
                            subject = ea
                            some_text = (symbol + 'を' + str(price) +
                                'で売りエントリーです。')
                            send_mail(subject, some_text, fromaddr, toaddr,
                                      host, port, password)
                        # EAにシグナルを送信する場合
                        if mt4 == 1:
                            send_signal2mt4(filename, signal)
                # シグナルを出力する。
                now = datetime.now()
                print(now.strftime('%Y.%m.%d %H:%M:%S'), ea, symbol,
                      timeframe, signal.iloc[end_row])
            # ロックを解放する。
            LOCK.release()
        # エラーを処理する。
        except Exception as e:
            print('エラーが発生しました。')
            print(e)
            time.sleep(1) # 1秒おきに表示させる。