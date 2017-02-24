# coding: utf-8

import configparser
import cvxopt as opt
import inspect
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

def create_pkl_file_path():
    '''pklファイルのパスを作成する。
    Returns:
        pklファイルのパス。
    '''
    dir_name = os.path.dirname(__file__) + '/temp/'
    framerecords = inspect.stack()
    framerecord = framerecords[1]
    frame = framerecord[0]
    func_name = frame.f_code.co_name
    ls = list(inspect.getargvalues(frame)[3].values())
    size = len(ls)
    arg_values = ''
    for i in range(size):
        if len(str(ls[size-1-i])) < 30:
            arg_values += '_' + str(ls[size-1-i])
        # 名前が長すぎる場合はindexと仮定する。
        else:
            arg_values += '_index'
            
    arg_values += '.pkl'
    pkl_file_path = dir_name + func_name + arg_values
    return pkl_file_path

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

def fill_invalidate_data(data):
    '''無効なデータを埋める。
    Args:
        data: データ。
    '''
    data = data.fillna(method='ffill')
    data[(np.isnan(data)) | (data==float("inf")) | (data==float("-inf"))] = 0.0

def get_current_filename_no_ext():
    '''拡張子なしの現在のファイル名を返す。
    Returns:
        拡張子なしの現在のファイル名。
    '''
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

def hour():
    '''現在の時を返す。
    Returns:
        現在の時。
    '''    
    hour = datetime.now().hour
    return hour

def i_atr(symbol, timeframe, period, shift):
    '''ATRを返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        period: 計算期間。
        shift: シフト。
    Returns:
        ATR。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        high = i_high(symbol, timeframe, shift)
        low = i_low(symbol, timeframe, shift)
        close = i_close(symbol, timeframe, shift)
        temp = high - low
        temp = pd.concat([temp, high - close.shift(1)], axis=1)
        temp = pd.concat([temp, close.shift(1) - low], axis=1)
        tr = temp.max(axis=1)
        ret = tr.rolling(window=period).mean()
        fill_invalidate_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_bandwalk(symbol, timeframe, period, ma_method, shift):
    '''バンドウォークを返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        period: 計算期間。
        ma_method: 移動平均のメソッド。
        shift: シフト。
    Returns:
        バンドウォーク。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
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
        ret = func(high, low, ma, ret, length)
        ret = pd.Series(ret, index=index)
        fill_invalidate_data(ret)
        ret = ret.astype(int)
        save_pkl(ret, pkl_file_path)
    return ret

def i_bandwalk_cl(symbol, timeframe, period, ma_method, shift):
    '''バンドウォーク（終値ベース）を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        period: 計算期間。
        ma_method: 移動平均のメソッド。
        shift: シフト。
    Returns:
        バンドウォーク（終値ベース）。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
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
        ret = func(close, ma, ret, length)
        ret = pd.Series(ret, index=index)
        fill_invalidate_data(ret)
        ret = ret.astype(int)
        save_pkl(ret, pkl_file_path)
    return ret

def i_close(symbol, timeframe, shift):
    '''終値を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        shift: シフト。
    Returns:
        終値。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    # トレードのとき、
    if OANDA is not None:
        instrument = convert_symbol2instrument(symbol)
        granularity = convert_timeframe2granularity(timeframe)
        temp = OANDA.get_history(
            instrument=instrument, granularity=granularity, count=COUNT)
        index = pd.Series(np.zeros(COUNT))
        ret = pd.Series(np.zeros(COUNT))
        for i in range(COUNT):
            index[i] = temp['candles'][i]['time']
            ret[i] = temp['candles'][i]['closeBid']
            index = pd.to_datetime(index)
            ret.index = index
        ret = ret.shift(shift)
    # バックテスト、またはウォークフォワードテストのとき、
    else:
        ret = restore_pkl(pkl_file_path)
        if ret is None:
            filename = ('~/historical_data/' + symbol + str(timeframe) +
                '.csv')
            temp = pd.read_csv(filename, index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            ret = temp.iloc[:, 3]
            ret = ret.shift(shift)
            fill_invalidate_data(ret)
            save_pkl(ret, pkl_file_path)
    return ret

def i_event(index, timeframe, before, after):
    '''イベントの有無を返す（1時間未満の足でのみ使用する）。
    Args:
        index: インデックス。
        timeframe: 足の種類。
        before: イベントの前の時間（分単位）。
        after: イベントの後の時間（分単位）。
    Returns:
        イベントの有無。
            1: 有。
            0:無。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        temp = pd.Series(index=index)
        day_of_week = time_day_of_week(index)
        hour = time_hour(index)
        minute = time_minute(index)
        b = int(before / timeframe)
        a = int(after / timeframe)
        # Mon 00:00
        temp[(day_of_week==1) & (hour==0) & (minute==0)] = 1
        # Tue 16:00
        temp[(day_of_week==2) & (hour==16) & (minute==0)] = 1
        # Wen 10:30
        temp[(day_of_week==3) & (hour==10) & (minute==30)] = 1
        # Thu 03:30
        temp[(day_of_week==4) & (hour==3) & (minute==30)] = 1
        # Thu 10:30   
        temp[(day_of_week==4) & (hour==10) & (minute==30)] = 1
        # Thu 14:30
        temp[(day_of_week==4) & (hour==14) & (minute==30)] = 1
        # Thu 14:35
        temp[(day_of_week==4) & (hour==14) & (minute==35)] = 1
        # Thu 15:30
        temp[(day_of_week==4) & (hour==15) & (minute==30)] = 1
        # Thu 16:00
        temp[(day_of_week==4) & (hour==16) & (minute==0)] = 1
        # Fri 14:30
        temp[(day_of_week==5) & (hour==14) & (minute==30)] = 1
        # Fri 14:35
        temp[(day_of_week==5) & (hour==14) & (minute==35)] = 1
        # Fri 15:30
        temp[(day_of_week==5) & (hour==15) & (minute==30)] = 1
        # Fri 15:40
        temp[(day_of_week==5) & (hour==15) & (minute==40)] = 1
        # Fri 16:00
        temp[(day_of_week==5) & (hour==16) & (minute==0)] = 1
        # Fri 16:55
        temp[(day_of_week==5) & (hour==16) & (minute==55)] = 1
        temp = temp.fillna(0)
        ret = temp.copy()
        for i in range(b):
            ret += temp.shift(-i-1)
        for i in range(a):
            ret += temp.shift(i+1)    
        ret = ret.fillna(0)
        ret[ret>1] = 1
        fill_invalidate_data(ret)
        ret = ret.astype(int)
        save_pkl(ret, pkl_file_path)
    return ret

def i_high(symbol, timeframe, shift):
    '''高値を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        shift: シフト。
    Returns:
        高値。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    # トレードのとき、
    if OANDA is not None:
        instrument = convert_symbol2instrument(symbol)
        granularity = convert_timeframe2granularity(timeframe)
        temp = OANDA.get_history(
            instrument=instrument, granularity=granularity, count=COUNT)
        index = pd.Series(np.zeros(COUNT))
        ret = pd.Series(np.zeros(COUNT))
        for i in range(COUNT):
            index[i] = temp['candles'][i]['time']
            ret[i] = temp['candles'][i]['highBid']
            index = pd.to_datetime(index)
            ret.index = index
        ret = ret.shift(shift)
    # バックテスト、またはウォークフォワードテストのとき、
    else:
        ret = restore_pkl(pkl_file_path)
        if ret is None:
            filename = ('~/historical_data/' + symbol + str(timeframe) +
                '.csv')
            temp = pd.read_csv(filename, index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            ret = temp.iloc[:, 1]
            ret = ret.shift(shift)
            fill_invalidate_data(ret)
            save_pkl(ret, pkl_file_path)
    return ret

def i_highest(symbol, timeframe, period, shift):
    '''直近高値の位置を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        period: 計算期間。
        shift: シフト。
    Returns:
        直近高値の位置。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        def func(high):
            period = len(high)
            argmax = high.argmax()
            ret = period - 1 - argmax
            return ret

        high = i_high(symbol, timeframe, shift)
        ret = high.rolling(window=period).apply(func)
        fill_invalidate_data(ret)
        ret = ret.astype(int)
        save_pkl(ret, pkl_file_path)
    return ret

def i_hl_band(symbol, timeframe, period, shift):
    '''直近の高値、安値を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        period: 計算期間。
        shift: シフト。
    Returns:
        直近の高値、安値。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        high = i_high(symbol, timeframe, shift)
        low = i_low(symbol, timeframe, shift)
        ret = pd.DataFrame()
        ret['high'] = high.rolling(window=period).max()
        ret['low'] = low.rolling(window=period).min()
        ret['middle'] = (ret['high'] + ret['low']) / 2
        fill_invalidate_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_ku_bandwalk(timeframe, period, shift, aud=0, cad=0, chf=0, eur=0, gbp=0,
                  jpy=0, nzd=0, usd=0):
    '''Ku-Powerのバンドウォークを返す。
    Args:
        timeframe: 足の種類。
        period: 計算期間。
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
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
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
        ret = func(ku_close, ku_ma, ret, m, n)
        ret = pd.DataFrame(ret, index=index, columns=columns)
        fill_invalidate_data(ret)
        ret = ret.astype(int)
        save_pkl(ret, pkl_file_path)
    return ret

def i_ku_power(timeframe, shift, aud=0, cad=0, chf=0, eur=0, gbp=0, jpy=0,
               nzd=0, usd=0):
    '''Ku-Powerを返す。
    Args:
        timeframe: 足の種類。
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
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        # 終値を格納する。
        audusd = 0.0
        cadusd = 0.0
        chfusd = 0.0
        eurusd = 0.0
        gbpusd = 0.0
        jpyusd = 0.0
        nzdusd = 0.0
        if aud == 1:
            audusd = np.log(i_close('AUDUSD', timeframe, shift))
        if cad == 1:
            cadusd = -np.log(i_close('USDCAD', timeframe, shift))
        if chf == 1:
            chfusd = -np.log(i_close('USDCHF', timeframe, shift))
        if eur == 1:
            eurusd = np.log(i_close('EURUSD', timeframe, shift))
        if gbp == 1:
            gbpusd = np.log(i_close('GBPUSD', timeframe, shift))
        if jpy == 1:
            jpyusd = -np.log(i_close('USDJPY', timeframe, shift))
        if nzd == 1:
            nzdusd = np.log(i_close('NZDUSD', timeframe, shift))
        # Ku-Powerを作成する。
        n = aud + cad + chf + eur + gbp + jpy + nzd + usd
        a = (audusd * aud + cadusd * cad + chfusd * chf + eurusd * eur +
            gbpusd * gbp + jpyusd * jpy + nzdusd * nzd) / n
        ret = pd.DataFrame()
        if aud == 1:
            ret['AUD'] = audusd - a
        if cad == 1:
            ret['CAD'] = cadusd - a
        if chf == 1:
            ret['CHF'] = chfusd - a
        if eur == 1:
            ret['EUR'] = eurusd - a
        if gbp == 1:
            ret['GBP'] = gbpusd - a
        if jpy == 1:
            ret['JPY'] = jpyusd - a
        if nzd == 1:
            ret['NZD'] = nzdusd - a
        if usd == 1:
            ret['USD'] = -a
        fill_invalidate_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_ku_roc(timeframe, period, shift, aud=0, cad=0, chf=0, eur=0, gbp=0,
                jpy=0, nzd=0, usd=0):
    '''Ku-Powerの変化率を返す。
    Args:
        timeframe: 足の種類。
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
        Ku-Powerの変化率。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        ku_power = i_ku_power(timeframe, shift, aud=aud, cad=cad, chf=chf,
                              eur=eur, gbp=gbp, jpy=jpy, nzd=nzd, usd=usd)
        ret = ku_power - ku_power.shift(period)
        fill_invalidate_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_ku_zscore(timeframe, period, ma_method, shift, aud=0, cad=0, chf=0,
                eur=0, gbp=0, jpy=0, nzd=0, usd=0):
    '''Ku-PowerのZスコアを返す。
    Args:
        timeframe: 足の種類。
        period:計算期間。
        ma_method: 移動平均のメソッド。
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
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        ku_power = i_ku_power(timeframe, shift, aud=aud, cad=cad, chf=chf,
                              eur=eur, gbp=gbp, jpy=jpy, nzd=nzd, usd=usd)
        if ma_method == 'MODE_SMA':
            mean = ku_power.rolling(window=period).mean()
            std = ku_power.rolling(window=period).std()
        elif ma_method == 'MODE_EMA':
            mean = ku_power.ewm(span=period).mean()
            std = ku_power.ewm(span=period).std()
        std = std.mean(axis=1)
        ret = (ku_power - mean).div(std, axis=0)  # メモリーエラー対策。
        fill_invalidate_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_low(symbol, timeframe, shift):
    '''安値を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        shift: シフト。
    Returns:
        安値。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    # トレードのとき、
    if OANDA is not None:
        instrument = convert_symbol2instrument(symbol)
        granularity = convert_timeframe2granularity(timeframe)
        temp = OANDA.get_history(
            instrument=instrument, granularity=granularity, count=COUNT)
        index = pd.Series(np.zeros(COUNT))
        ret = pd.Series(np.zeros(COUNT))
        for i in range(COUNT):
            index[i] = temp['candles'][i]['time']
            ret[i] = temp['candles'][i]['lowBid']
            index = pd.to_datetime(index)
            ret.index = index
        ret = ret.shift(shift)
    # バックテスト、またはウォークフォワードテストのとき、
    else:
        ret = restore_pkl(pkl_file_path)
        if ret is None:
            filename = ('~/historical_data/' + symbol + str(timeframe) +
                '.csv')
            temp = pd.read_csv(filename, index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            ret = temp.iloc[:, 2]
            ret = ret.shift(shift)
            fill_invalidate_data(ret)
            save_pkl(ret, pkl_file_path)
    return ret

def i_lowest(symbol, timeframe, period, shift):
    '''直近安値の位置を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        period: 計算期間。
        shift: シフト。
    Returns:
        直近安値の位置。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        def func(low):
            period = len(low)
            argmin = low.argmin()
            ret = period - 1 - argmin
            return ret

        low = i_low(symbol, timeframe, shift)
        ret = low.rolling(window=period).apply(func)
        fill_invalidate_data(ret)
        ret = ret.astype(int)
        save_pkl(ret, pkl_file_path)
    return ret

def i_open(symbol, timeframe, shift):
    '''始値を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        shift: シフト。
    Returns:
        始値。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    # トレードのとき、
    if OANDA is not None:
        instrument = convert_symbol2instrument(symbol)
        granularity = convert_timeframe2granularity(timeframe)
        temp = OANDA.get_history(
            instrument=instrument, granularity=granularity, count=COUNT)
        index = pd.Series(np.zeros(COUNT))
        ret = pd.Series(np.zeros(COUNT))
        for i in range(COUNT):
            index[i] = temp['candles'][i]['time']
            ret[i] = temp['candles'][i]['openBid']
            index = pd.to_datetime(index)
            ret.index = index
        ret = ret.shift(shift)
    # バックテスト、またはウォークフォワードテストのとき、
    else:
        ret = restore_pkl(pkl_file_path)
        if ret is None:
            filename = ('~/historical_data/' + symbol + str(timeframe) +
                '.csv')
            temp = pd.read_csv(filename, index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            ret = temp.iloc[:, 0]
            ret = ret.shift(shift)
            fill_invalidate_data(ret)
            save_pkl(ret, pkl_file_path)
    return ret

def i_opening_range(symbol, timeframe, shift):
    '''日足始値からの（対数変換した）値幅を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        shift: シフト。
    Returns:
        日足始値からの（対数変換した）値幅。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        log_open = np.log(i_open(symbol, timeframe, shift))
        log_close = np.log(i_close(symbol, timeframe, shift))
        index = log_open.index
        log_open[(time_hour(index)!=0) | (time_minute(index)!=0)] = np.nan
        log_open = log_open.fillna(method='ffill')
        ret = log_close - log_open
        fill_invalidate_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_roc(symbol, timeframe, period, shift):
    '''変化率を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        period: 計算期間。
        shift: シフト。
    Returns:
        変化率。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        close = i_close(symbol, timeframe, shift)
        ret = (close / close.shift(period) - 1.0) * 100.0
        fill_invalidate_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_strength(timeframe, period, shift, aud=1, cad=0, chf=0, eur=1, gbp=1,
               jpy=1, nzd=0, usd=1):
    '''通貨の強さを返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
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
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
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
        ret = temp.rank(axis=1, method='first')
        data = np.array(range(1, n+1))
        mean = np.mean(data)
        std = np.std(data)
        ret = (ret - mean) / std
        fill_invalidate_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_trange(symbol, timeframe, shift):
    '''TRANGEを返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        shift: シフト。
    Returns:
        TRANGE。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        high = i_high(symbol, timeframe, shift)
        low = i_low(symbol, timeframe, shift)
        close = i_close(symbol, timeframe, shift)
        temp = high - low
        temp = pd.concat([temp, high - close.shift(1)], axis=1)
        temp = pd.concat([temp, close.shift(1) - low], axis=1)
        ret = temp.max(axis=1)
        fill_invalidate_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_volume(symbol, timeframe, shift):
    '''出来高を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        shift: シフト。
    Returns:
        出来高。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    # トレードのとき、
    if OANDA is None:
        instrument = convert_symbol2instrument(symbol)
        granularity = convert_timeframe2granularity(timeframe)
        temp = OANDA.get_history(
            instrument=instrument, granularity=granularity, count=COUNT)
        index = pd.Series(np.zeros(COUNT))
        ret = pd.Series(np.zeros(COUNT))
        for i in range(COUNT):
            index[i] = temp['candles'][i]['time']
            ret[i] = temp['candles'][i]['volumeBid']
            index = pd.to_datetime(index)
            ret.index = index
        ret = ret.shift(shift)
    # バックテスト、またはウォークフォワードテストのとき、
    else:
        ret = restore_pkl(pkl_file_path)
        if ret is None:
            filename = ('~/historical_data/' + symbol + str(timeframe) +
                '.csv')
            temp = pd.read_csv( filename, index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            ret = temp.iloc[:, 4]
            ret = ret.shift(shift)
            fill_invalidate_data(ret)
            save_pkl(ret, pkl_file_path)
    return ret

def i_zscore(symbol, timeframe, period, ma_method, shift):
    '''終値のzスコアを返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        period: 計算期間。
        ma_method: 移動平均のメソッド。
        shift: シフト。
    Returns:
        終値のzスコア。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        close = i_close(symbol, timeframe, shift)
        if ma_method == 'MODE_SMA':
            mean = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
        elif ma_method == 'MODE_EMA':
            mean = close.ewm(span=period).mean()
            std = close.ewm(span=period).std()
        ret = (close - mean) / std
        fill_invalidate_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_zresid(symbol, timeframe, period, method, shift, exp1=None, exp2=None,
             exp3=None, exp4=None, exp5=None, exp6=None, exp7=None):
    '''終値とその予測値との標準化残差を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        period: 計算期間。
        method: メソッド。
        shift: シフト。
        exp1-exp7: 説明変数となる通貨ペア（必ずexp1から指定する）。
    Returns:
        終値とその予測値との標準化残差。
    '''
    pkl_file_path = create_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
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
        ret = np.empty(n)
        if method == 'linear':
            clf = linear_model.LinearRegression()
        elif method == 'tree':
            max_depth = 3
            clf = tree.DecisionTreeRegressor(max_depth=max_depth)
        for i in range(period, n):
            x = exp[i-period+1:i+1]
            y = close[i-period+1:i+1]
            ret[i] = func(x, y, period, clf)
        ret = pd.Series(ret, index=index)
        fill_invalidate_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

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

def restore_pkl(pkl_file_path):
    '''pklファイルを復元する。
    Args:
        pkl_file_path: pklファイルのパス。
    Returns:
        pklファイル。
    '''
    if OANDA is None and os.path.exists(pkl_file_path) == True:
        ret = joblib.load(pkl_file_path)
    else:
        ret = None
    return ret

def save_model(model, filename):
    pathname = os.path.dirname(__file__) + '/' + filename
    if os.path.exists(pathname) == False:
        os.mkdir(pathname)
    joblib.dump(model, pathname + '/' + filename + '.pkl') 

def save_pkl(data, pkl_file_path):
    '''pklファイルを保存する。
    Args:
        data: データ。
        pkl_file_path: pklファイルのパス。
    '''
    if OANDA is None:
        make_temp_folder()
        joblib.dump(data, pkl_file_path)

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