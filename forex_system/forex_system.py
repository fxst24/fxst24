# coding: utf-8
import configparser
import matplotlib.pyplot as plt
import numpy as np
import oandapy
import os
import pandas as pd
import shutil
import smtplib
import threading
import time
from datetime import datetime
from datetime import timedelta
from email.mime.text import MIMEText
from numba import float64, int64, jit
from pykalman import KalmanFilter
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
# Spyderのバグ（？）で警告が出るので無視する。
import warnings
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)

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

def backtest(args):
    '''バックテストを行う。
    Args:
        args: 引数。
    Returns:
        リターン、トレード数、パラメータ、タイムフレーム、開始日、終了日。
    '''
    # 設定を格納する。
    n = len(args)
    ea = args[0]
    symbol = args[1]
    timeframe = int(args[2])
    start = datetime.strptime(args[3], '%Y.%m.%d')
    end = datetime.strptime(args[4], '%Y.%m.%d')
    spread = int(args[5])
    position = 2  # デフォルト値
    min_trade = 260  # デフォルト値
    optimization = 0  # デフォルト値
    if n >= 7:
        position = int(args[6])
    if n >= 8:
        min_trade = int(args[7])
    if n == 9:
        optimization = int(args[8])
    exec('import ' + ea + ' as ea_file')
    strategy = eval('ea_file.strategy')
    parameter = eval('ea_file.PARAMETER')
    rranges = eval('ea_file.RRANGES')
    # パフォーマンスを計算する関数を定義する。
    def calc_performance(parameter, strategy, symbol, timeframe, start, end,
                         spread, position, min_trade, optimization):
        '''パフォーマンスを計算する。
        Args:
            parameter: 最適化するパラメータ。
            strategy: 戦略を記述した関数。
            symbol: 通貨ペア名。
            timeframe: 足の種類。
            start: 開始日。
            end: 終了日。
            spread: スプレッド。
            position: ポジションの設定。  0: 買いのみ。  1: 売りのみ。  2: 売買両方。
            min_trade: 最低トレード数。
            optimization: 最適化の設定。  0: 最適化なし。  1: 最適化あり。
        Returns:
            最適化ありの場合は適応度、最適化なしの場合はリターン, トレード数, 年率,
            シャープレシオ, 最適レバレッジ, ドローダウン, ドローダウン期間。
        '''
        # パフォーマンスを計算する。
        signal = strategy(parameter, symbol, timeframe, position)
        ret = calc_ret(symbol, timeframe, signal, spread, start, end)
        trades = calc_trades(signal, start, end)
        sharpe = calc_sharpe(ret, start, end)
        # 最適化する場合、適応度の符号を逆にして返す（最適値=最小値のため）。
        if optimization == 1:
            years = (end - start).total_seconds() / 60 / 60 / 24 / 365
            # 1年当たりのトレード数が最低トレード数に満たない場合、
            # 適応度を0にする。
            if trades / years >= min_trade:
                fitness = sharpe
            else:
                fitness = 0.0
            return -fitness
        # 最適化しない場合、各パフォーマンスを返す。
        else:
            return ret, trades
    # バックテストを行う。
    if optimization == 1:
        result = optimize.brute(calc_performance, rranges,
                                args=(strategy, symbol, timeframe, start, end,
                                      spread, position, min_trade, 1),
                                      finish=None)
        parameter = result
    ret, trades = calc_performance(parameter, strategy, symbol, timeframe,
                                   start, end, spread, position, min_trade, 0)
    return ret, trades, parameter, timeframe, start, end

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
        start: 開始年月日。
        end: 終了年月日。
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
        timeframe: タイムフレーム。
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
 
def calc_ret(symbol, timeframe, signal, spread, start, end):
    '''リターンを計算する。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        signal: シグナル。
        spread: スプレッド。
        start: 開始年月日。
        end: 終了年月日。
    Returns:
        年率。
    '''
    # スプレッドの単位の調整
    if (symbol == 'AUDJPY' or symbol == 'CADJPY' or symbol == 'CHFJPY' or
        symbol == 'EURJPY' or symbol == 'GBPJPY' or symbol == 'NZDJPY' or
        symbol == 'USDJPY'):
        adjusted_spread = spread / 1000.0
    else:
        adjusted_spread = spread / 100000.0
    # コストを計算する。
    temp1 = (adjusted_spread * ((signal > 0) & (signal > signal.shift(1))) *
        (signal - signal.shift(1)))
    temp2 = (adjusted_spread * ((signal < 0) & (signal < signal.shift(1))) *
        (signal.shift(1) - signal))
    cost = temp1 + temp2
    # リターンを計算する。
    op = i_open(symbol, timeframe, 0)
    ret = ((op.shift(-1) - op) * signal - cost) / op
    ret = ret.fillna(0.0)
    ret[(ret==float('inf')) | (ret==float('-inf'))] = 0.0
    ret = ret[start:end]
    return ret
 
def calc_sharpe(ret, start, end):
    '''シャープレシオを計算する。
    Args:
        ret: リターン。
        start: 開始年月日。
        end: 終了年月日。
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

def calc_signal(buy_entry, buy_exit, sell_entry, sell_exit, position):
    '''シグナルを計算する。
    Args:
        buy_entry: 買いエントリー。
        buy_exit: 買いエグジット。
        sell_entry: 売りエントリー。
        sell_exit: 売りエグジット。
        position: ポジションの設定。  0: 買いのみ。  1: 売りのみ。  2: 売買両方。
    Returns:
        シグナル。
    '''
    # シグナルを計算する。
    buy = buy_entry.copy()
    buy[buy==0] = np.nan
    buy[buy_exit==1] = 0
    buy = buy.fillna(method='ffill')
    sell = -sell_entry.copy()
    sell[sell==0] = np.nan
    sell[sell_exit==1] = 0
    sell = sell.fillna(method='ffill')
    if position == 0:
        signal = buy
    elif position == 1:
        signal = sell
    else:
        signal = buy + sell
    signal = signal.fillna(0)
    signal = signal.astype(int)
    return signal

def calc_trades(signal, start, end):
    '''トレード数を計算する。
    Args:
        signal: シグナル。
        start: 開始年月日。
        end: 終了年月日。
    Returns:
        トレード数。
    '''
    temp1 = (((signal > 0) & (signal > signal.shift(1))) *
        (signal - signal.shift(1)))
    temp2 = (((signal < 0) & (signal < signal.shift(1))) *
        (signal.shift(1) - signal))
    trade = temp1 + temp2
    trade = trade.fillna(0)
    trade = trade.astype(int)
    trades = trade[start:end].sum()
    return trades

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
        timeframe: タイムフレーム。
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

def i_bands(symbol, timeframe, period, deviation, shift):
    '''ボリンジャーバンドを返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 計算期間。
        deviation: 標準偏差。
        shift: シフト。
    Returns:
        ボリンジャーバンド。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_bands_' + symbol +
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
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(bands, path)
    return bands

def i_bandwalk(symbol, timeframe, period, shift):
    '''バンドウォーク（移動平均）を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
    Returns:
        バンドウォーク（移動平均）。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_bandwalk_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        bandwalk = joblib.load(path)
    # さもなければ計算する。
    else:
        # バンドウォークを計算する関数を定義する。
        @jit(float64[:](float64[:], float64[:], float64[:]),
            nopython=True, cache=True)
        def calc_bandwalk(high, low, ma):
            up = 0
            down = 0
            length = len(ma)
            bandwalk = np.empty(length)
            for i in range(length):
                if (low[i] > ma[i]):
                    up = up + 1
                else:
                    up = 0
                if (high[i] < ma[i]):
                    down = down + 1
                else:
                    down = 0
                bandwalk[i] = up - down

            return bandwalk

        # バンドウォークを計算する。
        high = i_high(symbol, timeframe, shift)
        low = i_low(symbol, timeframe, shift)
        ma = i_ma(symbol, timeframe, period, shift)
        index = ma.index
        high = np.array(high)
        low = np.array(low)
        ma = np.array(ma)
        bandwalk = calc_bandwalk(high, low, ma)
        a = 0.903  # 指数（正規化するために経験的に導き出した数値）
        b = 0.393  # 切片（同上）
        bandwalk = bandwalk / (float(period) ** a + b)
        bandwalk = pd.Series(bandwalk, index=index)
        bandwalk = bandwalk.fillna(0)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(bandwalk, path)
    return bandwalk

def i_bandwalk_linear_regression(symbol, timeframe, period, shift, aud=0.0,
                                 cad=0.0, chf=0.0, eur=0.0, gbp=0.0, jpy=0.0,
                                 nzd=0.0, usd=0.0):
    '''バンドウォーク（線形回帰）を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
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
        バンドウォーク（線形回帰）。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_bandwalk_linear_regression_' +
        symbol + str(timeframe) + '_' + str(period) + '_' + str(shift) + '_' +
        str(aud) + str(cad) + str(chf) + str(eur) + str(gbp) + str(jpy) +
        str(nzd) + str(usd) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        bandwalk_linear_regression = joblib.load(path)
    # さもなければ計算する。
    else:
        # バンドウォークを計算する関数を定義する。
        @jit(float64[:](float64[:], float64[:], float64[:]),
            nopython=True, cache=True)
        def calc_bandwalk_linear_regression(high, low, pred):
            up = 0
            down = 0
            length = len(pred)
            bandwalk_linear_regression = np.empty(length)
            for i in range(length):
                if (low[i] > pred[i]):
                    up = up + 1
                else:
                    up = 0
                if (high[i] < pred[i]):
                    down = down + 1
                else:
                    down = 0
                bandwalk_linear_regression[i] = up - down
            return bandwalk_linear_regression

        # バンドウォークを計算する。
        high = i_high(symbol, timeframe, shift)
        low = i_low(symbol, timeframe, shift)
        pred = i_linear_regression(symbol, timeframe, period, shift, aud=aud,
                                   cad=cad, chf=chf, eur=eur, gbp=gbp, jpy=jpy,
                                   nzd=nzd, usd=usd)
        index = pred.index
        high = np.array(high)
        low = np.array(low)
        pred = np.array(pred)
        bandwalk_linear_regression = calc_bandwalk_linear_regression(high, low,
                                                                     pred)
        a = 0.903  # 指数（正規化するために経験的に導き出した数値）
        b = 0.393  # 切片（同上）
        bandwalk_linear_regression = (bandwalk_linear_regression /
            (float(period) ** a + b))
        bandwalk_linear_regression = pd.Series(bandwalk_linear_regression,
                                               index=index)
        bandwalk_linear_regression = bandwalk_linear_regression.fillna(0)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(bandwalk_linear_regression, path)
    return bandwalk_linear_regression

def i_bandwalk_rank(timeframe, period, shift, aud=0.0, cad=0.0, chf=0.0,
                    eur=0.0, gbp=0.0, jpy=0.0, nzd=0.0, usd=0.0):
    '''バンドウォーク（通貨の強弱の順位）を返す。
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
        バンドウォーク（通貨の強弱の順位）。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_bandwalk_rank_' +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '_' +
        str(aud) + str(cad) + str(chf) + str(eur) + str(gbp) + str(jpy) +
        str(nzd) + str(usd) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        bandwalk_rank = joblib.load(path)
    # さもなければ計算する。
    else:
        # バンドウォークを計算する関数を定義する。
        @jit(float64[:, :](float64[:, :], float64[:, :], int64, int64),
             nopython=True, cache=True)
        def calc_bandwalk_rank(rank, bandwalk_rank, m, n):
            for j in range(n):
                up = 0
                down = 0
                for i in range(m):
                    if (rank[i][j] > 0.0):
                        up = up + 1
                    else:
                        up = 0
                    if (rank[i][j] < 0.0):
                        down = down + 1
                    else:
                        down = 0
                    bandwalk_rank[i][j] = up - down
            return bandwalk_rank

        # バンドウォークを計算する。
        rank =i_rank(timeframe, period, shift, aud=aud, cad=cad, chf=chf,
                     eur=eur, gbp=gbp, jpy=jpy, nzd=nzd, usd=usd)
        index = rank.index
        columns = rank.columns
        rank = np.array(rank)
        m = len(rank)
        n = len(rank[0])
        bandwalk_rank = np.empty([m, n])  # numbaは関数内部で2次元配列を作れない。
        bandwalk_rank = calc_bandwalk_rank(rank, bandwalk_rank, m, n)
        a = 0.903  # 指数（正規化するために経験的に導き出した数値）
        b = 0.393  # 切片（同上）
        bandwalk_rank = bandwalk_rank / (float(period) ** a + b)
        bandwalk_rank = pd.DataFrame(bandwalk_rank, index=index,
                                     columns=columns)
        bandwalk_rank = bandwalk_rank.fillna(0)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(bandwalk_rank, path)
    return bandwalk_rank

def i_bandwalk_tree_regression(symbol, timeframe, period, shift, aud=0, cad=0,
                               chf=0, eur=0, gbp=0, jpy=0, nzd=0, usd=0.0,
                               max_depth=3):
    '''バンドウォーク（決定木）を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
        aud: 豪ドル。
        cad: カナダドル。
        chf: スイスフラン。
        eur: ユーロ。
        gbp: ポンド。
        jpy: 円。
        nzd: NZドル。
        usd: 米ドル。
        max_depth: 最大の深さ。
    Returns:
        バンドウォーク（決定木）。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_bandwalk_tree_regression_' +
        symbol + str(timeframe) + '_' + str(period) + '_' + str(shift) + '_' +
        str(aud) + str(cad) + str(chf) + str(eur) + str(gbp) + str(jpy) +
        str(nzd) + str(usd) + '_' + str(max_depth) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        bandwalk_tree_regression = joblib.load(path)
    # さもなければ計算する。
    else:
        # バンドウォークを計算する関数を定義する。
        @jit(float64[:](float64[:], float64[:], float64[:]),
            nopython=True, cache=True)
        def calc_bandwalk_tree_regression(high, low, pred):
            up = 0
            down = 0
            length = len(pred)
            bandwalk_linear_regression = np.empty(length)
            for i in range(length):
                if (low[i] > pred[i]):
                    up = up + 1
                else:
                    up = 0
                if (high[i] < pred[i]):
                    down = down + 1
                else:
                    down = 0
                bandwalk_linear_regression[i] = up - down
            return bandwalk_linear_regression

        # バンドウォークを計算する。
        high = i_high(symbol, timeframe, shift)
        low = i_low(symbol, timeframe, shift)
        pred = i_tree_regression(symbol, timeframe, period, shift, aud=aud,
                                 cad=cad, chf=chf, eur=eur, gbp=gbp, jpy=jpy,
                                 nzd=nzd, usd=usd, max_depth=max_depth)
        index = pred.index
        high = np.array(high)
        low = np.array(low)
        pred = np.array(pred)
        bandwalk_tree_regression = calc_bandwalk_tree_regression(high, low,
                                                                 pred)
        a = 0.903  # 指数（正規化するために経験的に導き出した数値）
        b = 0.393  # 切片（同上）
        bandwalk_tree_regression = (bandwalk_tree_regression /
            (float(period) ** a + b))
        bandwalk_tree_regression = pd.Series(bandwalk_tree_regression,
                                             index=index)
        bandwalk_tree_regression = bandwalk_tree_regression.fillna(0)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(bandwalk_tree_regression, path)
    return bandwalk_tree_regression

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
    path = (os.path.dirname(__file__) + '/tmp/i_close_' + symbol +
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
            close = close.fillna(method='bfill')
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(close, path)
    return close

def i_coef(symbol, timeframe, period, shift, aud=0.0, cad=0.0, chf=0.0, eur=0.0,
           gbp=0.0, jpy=0.0, nzd=0.0, usd=0.0):
    '''線形回帰による係数を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
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
        線形回帰による係数。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_coef_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + str(aud) +
        str(cad) + str(chf) + str(eur) + str(gbp) + str(jpy) + str(nzd) +
        str(usd) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        coef = joblib.load(path)
    # さもなければ計算する。
    else:
        def calc_coef(x, y, period):
            clf = linear_model.LinearRegression()
            clf.fit(x, y)
            coef = clf.coef_
            return coef

        close = i_close(symbol, timeframe, shift)
        index = close.index
        close = np.array(close)
        n_currency = ((
            (aud > 0.0) + (cad > 0.0) + (chf > 0.0) + (eur > 0.0) +
            (gbp > 0.0) + (jpy > 0.0) + (nzd > 0.0) + (usd > 0.0)) * 1)
        if n_currency == 0:
            ku_close = np.array(range(len(close)))
        else:
            ku_close = i_ku_close(timeframe, period, shift, aud=aud, cad=cad,
                                  chf=chf, eur=eur, gbp=gbp, jpy=jpy, nzd=nzd,
                                  usd=usd)
            ku_close = np.array(ku_close)
            if n_currency == 1:
                ku_close = ku_close.reshape(len(ku_close), 1)
        # リスト内包表記も試したが、特に速度は変わらない。
        n = len(close)
        if n_currency == 1:
            coef = np.empty(n)
        else:
            coef = np.empty([n, n_currency])
        for i in range(period, n):
            x = ku_close[i-period:i]
            y = close[i-period:i]
            coef[i] = calc_coef(x, y, period)
        if n == 1:
            coef = pd.Series(coef, index=index)
        else:
            coef = pd.DataFrame(coef, index=index)
        coef = coef.fillna(method='ffill')
        coef = coef.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(coef, path)
    return coef

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
    path = (os.path.dirname(__file__) + '/tmp/i_diff_' + symbol +
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
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(diff, path)
    return diff

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
    path = (os.path.dirname(__file__) + '/tmp/i_high_' + symbol +
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
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(high, path)
    return high

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
    path = (os.path.dirname(__file__) + '/tmp/i_hl_band_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        hl_band = joblib.load(path)
    # さもなければ計算する。
    else:
        close = i_close(symbol, timeframe, shift)
        hl_band = pd.DataFrame()
        hl_band['high'] = close.rolling(window=period).max()
        hl_band['low'] = close.rolling(window=period).min()
        hl_band['middle'] = (hl_band['high'] + hl_band['low']) / 2
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
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
    path = (os.path.dirname(__file__) + '/tmp/i_hurst_' + symbol +
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
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(hurst, path)
    return hurst

# まだ作りかけ。
def i_kalman_filter(symbol, timeframe, period, shift):
    '''カルマンフィルターによる予測値を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
    Returns:
        カルマンフィルターによる予測値。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_kalman_filter_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        kalman_filter = joblib.load(path)
    # さもなければ計算する。
    else:
        # カルマンフィルターを計算する関数を定義する。
        def calc_kalman_filter(data, period, n_iter):
            kf = KalmanFilter()
            kf = kf.em(data, n_iter=n_iter)
            #smoothed_state_means = kf.smooth(data)[0]
            #kalman_filter = smoothed_state_means[period-1, 0]
            filtered_state_means = kf.filter(data)[0]
            kalman_filter = filtered_state_means[period-1, 0]
            return kalman_filter
        # カルマンフィルターを計算する
        close = i_close(symbol, timeframe, shift)
        index = close.index
        close = np.array(close)
        # リスト内包表記も試したが、特に速度は変わらない。
        n = len(close)
        kalman_filter = np.empty(n)
        for i in range(period, n):
            data = close[i-period:i]
            kalman_filter[i] = calc_kalman_filter(data, period, 0)
        kalman_filter = pd.Series(kalman_filter, index=index)
        kalman_filter = kalman_filter.fillna(method='ffill')
        kalman_filter = kalman_filter.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(kalman_filter, path)
    return kalman_filter

def i_ku_bandwalk(timeframe, period, shift, aud=0.0, cad=0.0, chf=0.0, eur=0.0,
                  gbp=0.0, jpy=0.0, nzd=0.0, usd=0.0):
    '''Ku-Chartによるバンドウォークを返す。
    Args:
        timeframe: 足の種類。
        period: 期間。
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
        Ku-Chartによるバンドウォーク。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_ku_bandwalk_' + 
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '_' +
        str(aud) + str(cad) + str(chf) + str(eur) + str(gbp) + str(jpy) +
        str(nzd) + str(usd) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        ku_bandwalk = joblib.load(path)
    # さもなければ計算する。
    else:
        # Ku-Chartによるバンドウォークを計算する関数を定義する。
        @jit(float64[:, :](float64[:, :], float64[:, :], int64),
             nopython=True, cache=True)   
        def calc_ku_bandwalk(ku_close, median, n):
            length = len(ku_close)
            ku_bandwalk = np.empty((length, n))
            for j in range(n):
                up = 0
                down = 0
                for i in range(length):
                    if (ku_close[i][j] > median[i][j]):
                        up = up + 1
                    else:
                        up = 0
                    if (ku_close[i][j] < median[i][j]):
                        down = down + 1
                    else:
                        down = 0
                    ku_bandwalk[i][j] = up - down
            return ku_bandwalk
        # Ku-Chartによるバンドウォークを計算する。
        ku_close = i_ku_close(timeframe, shift, aud, cad, chf, eur, gbp, jpy,
                              nzd, usd)
        index = ku_close.index
        columns = ku_close.columns
        n = ((aud > EPS) + (cad > EPS) + (chf > EPS) + (eur > EPS) +
            (gbp > EPS) + (jpy > EPS) + (nzd > EPS) + (usd > EPS))
        median = ku_close.rolling(window=period).median()
        ku_close = ku_close.as_matrix()
        median = median.as_matrix()
        ku_bandwalk = calc_ku_bandwalk(ku_close, median, n)
        ku_bandwalk = pd.DataFrame(
            ku_bandwalk, index=index, columns=columns)
        ku_bandwalk = ku_bandwalk.fillna(0)
        a = 0.903  # 指数（正規化するために経験的に導き出した数値）
        b = 0.393  # 切片（同上）
        ku_bandwalk = ku_bandwalk / (float(period) ** a + b)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(ku_bandwalk, path)
    return ku_bandwalk

def i_ku_close(timeframe, shift, aud=0.0, cad=0.0, chf=0.0, eur=0.0, gbp=0.0,
               jpy=0.0, nzd=0.0, usd=0.0):
    '''Ku-Chartによる終値を返す。
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
        Ku-Chartによる終値。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_ku_close_' + str(timeframe) +
        '_' + str(shift) + '_' + str(aud) + str(cad) + str(chf) + str(eur) +
        str(gbp) + str(jpy) + str(nzd) + str(usd) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        ku_close = joblib.load(path)
    # さもなければ計算する。
    else:
        ku_close = pd.DataFrame()
        # 重みの合計を計算する。
        weight_sum = aud + cad + chf + eur + gbp + jpy + nzd + usd
        # 終値を格納する。
        audusd = 0.0
        cadusd = 0.0
        chfusd = 0.0
        eurusd = 0.0
        gbpusd = 0.0
        jpyusd = 0.0
        nzdusd = 0.0
        if aud > EPS:
            audusd = i_close('AUDUSD', timeframe, shift)
            audusd = audusd.apply(np.log)
        if cad > EPS:
            cadusd = 1 / i_close('USDCAD', timeframe, shift)
            cadusd = cadusd.apply(np.log)
        if chf > EPS:
            chfusd = 1 / i_close('USDCHF', timeframe, shift)
            chfusd = chfusd.apply(np.log)
        if eur > EPS:
            eurusd = i_close('EURUSD', timeframe, shift)
            eurusd = eurusd.apply(np.log)
        if gbp > EPS:
            gbpusd = i_close('GBPUSD', timeframe, shift)
            gbpusd = gbpusd.apply(np.log)
        if jpy > EPS:
            jpyusd = 1 / i_close('USDJPY', timeframe, shift)
            jpyusd = jpyusd.apply(np.log)
        if nzd > EPS:
            nzdusd = i_close('NZDUSD', timeframe, shift)
            nzdusd = nzdusd.apply(np.log)
        # KU-AUDを作成する。
        if aud > EPS:
            audcad = audusd - cadusd
            audchf = audusd - chfusd
            audeur = audusd - eurusd
            audgbp = audusd - gbpusd
            audjpy = audusd - jpyusd
            audnzd = audusd - nzdusd
            ku_close['aud'] = (((audcad * aud * cad) +
                (audchf * aud * chf) + (audeur * aud * eur) +
                (audgbp * aud * gbp) + (audjpy * aud * jpy) +
                (audnzd * aud * nzd) + (audusd * aud * usd)) /
                weight_sum * 10000.0)
        # KU-CADを作成する。
        if cad > EPS:
            cadaud = cadusd - audusd
            cadchf = cadusd - chfusd
            cadeur = cadusd - eurusd
            cadgbp = cadusd - gbpusd
            cadjpy = cadusd - jpyusd
            cadnzd = cadusd - nzdusd
            ku_close['cad'] = (((cadaud * cad * aud) +
                (cadchf * cad * chf) + (cadeur * cad * eur) +
                (cadgbp * cad * gbp) + (cadjpy * cad * jpy) +
                (cadnzd * cad * nzd) + (cadusd * cad * usd)) /
                weight_sum * 10000.0)
        # KU-CHFを作成する。
        if chf > EPS:
            chfaud = chfusd - audusd
            chfcad = chfusd - cadusd
            chfeur = chfusd - eurusd
            chfgbp = chfusd - gbpusd
            chfjpy = chfusd - jpyusd
            chfnzd = chfusd - nzdusd
            ku_close['chf'] = (((chfaud * chf * aud) +
                (chfcad * chf * cad) + (chfeur * chf * eur) +
                (chfgbp * chf * gbp) + (chfjpy * chf * jpy) +
                (chfnzd * chf * nzd) + (chfusd * chf * usd)) /
                weight_sum * 10000.0)
        # KU-EURを作成する。
        if eur > EPS:
            euraud = eurusd - audusd
            eurcad = eurusd - cadusd
            eurchf = eurusd - chfusd
            eurgbp = eurusd - gbpusd
            eurjpy = eurusd - jpyusd
            eurnzd = eurusd - nzdusd
            ku_close['eur'] = (((euraud * eur * aud) +
                (eurcad * eur * cad) + (eurchf * eur * chf) +
                (eurgbp * eur * gbp) + (eurjpy * eur * jpy) +
                (eurnzd * eur * nzd) + (eurusd * eur * usd)) /
                weight_sum * 10000.0)
        # KU-GBPを作成する。
        if gbp > EPS:
            gbpaud = gbpusd - audusd
            gbpcad = gbpusd - cadusd
            gbpchf = gbpusd - chfusd
            gbpeur = gbpusd - eurusd
            gbpjpy = gbpusd - jpyusd
            gbpnzd = gbpusd - nzdusd
            ku_close['gbp'] = (((gbpaud * gbp * aud) +
                (gbpcad * gbp * cad) + (gbpchf * gbp * chf) +
                (gbpeur * gbp * eur) + (gbpjpy * gbp * jpy) +
                (gbpnzd * gbp * nzd) + (gbpusd * gbp * usd)) /
                weight_sum * 10000.0)
        # KU-JPYを作成する。
        if jpy > EPS:
            jpyaud = jpyusd - audusd
            jpycad = jpyusd - cadusd
            jpychf = jpyusd - chfusd
            jpyeur = jpyusd - eurusd
            jpygbp = jpyusd - gbpusd
            jpynzd = jpyusd - nzdusd
            ku_close['jpy'] = (((jpyaud * jpy * aud) +
                (jpycad * jpy * cad) + (jpychf * jpy * chf) +
                (jpyeur * jpy * eur) + (jpygbp * jpy * gbp) +
                (jpynzd * jpy * nzd) + (jpyusd * jpy * usd)) /
                weight_sum * 10000.0)
        # KU-NZDを作成する。
        if nzd > EPS:
            nzdaud = nzdusd - audusd
            nzdcad = nzdusd - cadusd
            nzdchf = nzdusd - chfusd
            nzdeur = nzdusd - eurusd
            nzdgbp = nzdusd - gbpusd
            nzdjpy = nzdusd - jpyusd
            ku_close['nzd'] = (((nzdaud * nzd * aud) +
                (nzdcad * nzd * cad) + (nzdchf * nzd * chf) +
                (nzdeur * nzd * eur) + (nzdgbp * nzd * gbp) +
                (nzdjpy * nzd * jpy) + (nzdusd * nzd * usd)) /
                weight_sum * 10000.0)
        # KU-USDを作成する。
        if usd > EPS:
            usdaud = -audusd
            usdcad = -cadusd
            usdchf = -chfusd
            usdeur = -eurusd
            usdgbp = -gbpusd
            usdjpy = -jpyusd
            usdnzd = -nzdusd
            ku_close['usd'] = (((usdaud * usd * aud) +
                (usdcad * usd * cad) + (usdchf * usd * chf) +
                (usdeur * usd * eur) + (usdgbp * usd * gbp) +
                (usdjpy * usd * jpy) + (usdnzd * usd * nzd)) /
                weight_sum * 10000.0)
        ku_close = ku_close.fillna(method='ffill')
        ku_close = ku_close.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(ku_close, path)
    return ku_close

def i_ku_zresid(timeframe, period, shift, aud=0.0, cad=0.0, chf=0.0, eur=0.0,
                 gbp=0.0, jpy=0.0, nzd=0.0, usd=0.0):
    '''Ku-Chartによる終値とその予測値との標準化残差を返す。
    Args:
        timeframe: 足の種類。
        period: 期間。
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
        Ku-Chartによる終値とその予測値との標準化残差を返す。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_ku_zresid_' + str(timeframe) +
        '_' + str(period) + '_' + str(shift) + '_' + str(aud) + str(cad) +
        str(chf) + str(eur) + str(gbp) + str(jpy) + str(nzd) + str(usd) +
        '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        ku_zresid = joblib.load(path)
    # さもなければ計算する。
    else:
        # zスコアを計算する関数を定義する。
        @jit(float64(float64[:]), nopython=True, cache=True)
        def calc_ku_zresid(ku_close):
            median = np.median(ku_close)
            period = len(ku_close)
            std = 0.0
            for i in range(period):
                std = std + (ku_close[i] - median) * (ku_close[i] - median)
            std = std / period
            std = np.sqrt(std)
            if std < EPS:
                ku_z_score = 0.0
            else:
                ku_z_score = (ku_close[-1] - median) / std
            return ku_z_score
        ku_close = i_ku_close(timeframe, shift, aud, cad, chf, eur, gbp, jpy,
                              nzd, usd)
        ku_zresid = ku_close.rolling(window=period).apply(calc_ku_zresid)
        ku_zresid = ku_zresid.fillna(0)
        ku_zresid[(ku_zresid==float('inf')) | (ku_zresid==float('-inf'))] = 0.0
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(ku_zresid, path)
    return ku_zresid

def i_kurt(symbol, timeframe, period, shift):
    '''リターンの尖度を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
    Returns:
        リターンの尖度。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_kurt_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        kurt = joblib.load(path)
    # さもなければ計算する。
    else:
        ret = i_return(symbol, timeframe, shift)
        kurt = ret.rolling(window=period).kurt()
        kurt = kurt.fillna(method='ffill')
        kurt = kurt.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(kurt, path)
    return kurt

def i_linear_regression(symbol, timeframe, period, shift, aud=0.0, cad=0.0,
                        chf=0.0, eur=0.0, gbp=0.0, jpy=0.0, nzd=0.0, usd=0.0):
    '''線形回帰による予測値を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
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
        線形回帰による予測値。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_linear_regression_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + str(aud) +
        str(cad) + str(chf) + str(eur) + str(gbp) + str(jpy) + str(nzd) +
        str(usd) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        linear_regression = joblib.load(path)
    # さもなければ計算する。
    else:
        def calc_linear_regression(x, y, period):
            clf = linear_model.LinearRegression()
            clf.fit(x, y)
            pred = clf.predict(x)
            linear_regression = pred[period-1]
            return linear_regression

        close = i_close(symbol, timeframe, shift)
        index = close.index
        close = np.array(close)
        n_currency = ((
            (aud > 0.0) + (cad > 0.0) + (chf > 0.0) + (eur > 0.0) +
            (gbp > 0.0) + (jpy > 0.0) + (nzd > 0.0) + (usd > 0.0)) * 1)
        if n_currency == 0:
            ku_close = np.array(range(len(close)))
        else:
            ku_close = i_ku_close(timeframe, shift, aud=aud, cad=cad,
                                  chf=chf, eur=eur, gbp=gbp, jpy=jpy, nzd=nzd,
                                  usd=usd)
            ku_close = np.array(ku_close)
            if n_currency == 1:
                ku_close = ku_close.reshape(len(ku_close), 1)
        # リスト内包表記も試したが、特に速度は変わらない。
        n = len(close)
        linear_regression = np.empty(n)
        for i in range(period, n):
            x = ku_close[i-period:i]
            y = close[i-period:i]
            linear_regression[i] = calc_linear_regression(x, y, period)
        linear_regression = pd.Series(linear_regression, index=index)
        linear_regression = linear_regression.fillna(method='ffill')
        linear_regression = linear_regression.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(linear_regression, path)
    return linear_regression

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
    path = (os.path.dirname(__file__) + '/tmp/i_low_' + symbol +
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
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(low, path)
    return low

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
    path = (os.path.dirname(__file__) + '/tmp/i_ma_' + symbol +
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
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(ma, path)
    return ma

def i_mean(symbol, timeframe, period, shift):
    '''リターンの平均を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
    Returns:
        リターンの平均。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_mean_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        mean = joblib.load(path)
    # さもなければ計算する。
    else:
        ret = i_return(symbol, timeframe, shift)
        mean = ret.rolling(window=period).mean()
        mean = mean.fillna(method='ffill')
        mean = mean.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(mean, path)
    return mean

def i_open(symbol, timeframe, shift):
    '''始値を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        shift: シフト。
    Returns:
        始値。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_open_' + symbol +
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
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(op, path)
    return op

def i_rank(timeframe, period, shift, aud=0.0, cad=0.0, chf=0.0, eur=0.0,
               gbp=0.0, jpy=0.0, nzd=0.0, usd=0.0):
    '''通貨の強弱の順位（降順）を返す。
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
        通貨の強弱の順位（降順）。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_rank_' + str(timeframe) + '_' +
        str(period) + '_' + str(shift) + '_' + str(aud) + str(cad) + str(chf) +
        str(eur) + str(gbp) + str(jpy) + str(nzd) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        rank = joblib.load(path)
    # さもなければ計算する。
    else:
        ku_zresid =i_ku_zresid(timeframe, period, shift, aud=aud, cad=cad,
                              chf=chf, eur=eur, gbp=gbp, jpy=jpy, nzd=nzd,
                              usd=usd)
        rank = ku_zresid.rank(axis=1)
        n = len(rank.iloc[0, :])
        array = np.array(range(1, n + 1))
        mean = np.mean(array)
        std = np.std(array)
        rank = (rank - mean) / std
        rank = rank.fillna(method='ffill')
        rank = rank.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(rank, path)
    return rank

def i_return(symbol, timeframe, shift):
    '''リターンを返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        shift: シフト。
    Returns:
        リターン。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_return_' + symbol +
        str(timeframe) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        ret = joblib.load(path)
    # さもなければ計算する。
    else:
        close = i_close(symbol, timeframe, shift)
        log_close = close.apply(np.log)
        ret = log_close - log_close.shift(1)
        ret = ret.fillna(0.0)
        ret[(ret==float('inf')) | (ret==float('-inf'))] = 0.0
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(ret, path)
    return ret

def i_skew(symbol, timeframe, period, shift):
    '''リターンの歪度を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
    Returns:
        リターンの歪度。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_skew_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        skew = joblib.load(path)
    # さもなければ計算する。
    else:
        ret = i_return(symbol, timeframe, shift)
        skew = ret.rolling(window=period).skew()
        skew = skew.fillna(method='ffill')
        skew = skew.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(skew, path)
    return skew

def i_std(symbol, timeframe, period, shift):
    '''リターンの標準偏差を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
    Returns:
        リターンの標準偏差。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_std_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        std = joblib.load(path)
    # さもなければ計算する。
    else:
        ret = i_return(symbol, timeframe, shift)
        std = ret.rolling(window=period).std()
        std = std.fillna(method='ffill')
        std = std.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
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
    path = (os.path.dirname(__file__) + '/tmp/i_std_dev_' + symbol +
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
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(std_dev, path)
    return std_dev

def i_std_dev_linear_regression(symbol, timeframe, period, shift, aud=0.0,
                                cad=0.0, chf=0.0, eur=0.0, gbp=0.0, jpy=0.0,
                                nzd=0.0, usd=0.0):
    '''線形回帰による予測値との標準偏差を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
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
        線形回帰による予測値との標準偏差。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_std_dev_linear_regression_' +
        symbol + str(timeframe) + '_' + str(period) + '_' + str(shift) +
        str(aud) + str(cad) + str(chf) + str(eur) + str(gbp) + str(jpy) + 
        str(nzd) + str(usd) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        std_dev_linear_regression = joblib.load(path)
    # さもなければ計算する。
    else:
        def calc_std_dev_linear_regression(x, y):
            clf = linear_model.LinearRegression()
            clf.fit(x, y)
            pred = clf.predict(x)
            error = y - pred
            return np.std(error)

        close = i_close(symbol, timeframe, shift)
        index = close.index
        close = np.array(close)
        n_currency = ((
            (aud > 0.0) + (cad > 0.0) + (chf > 0.0) + (eur > 0.0) +
            (gbp > 0.0) + (jpy > 0.0) + (nzd > 0.0) + (usd > 0.0)) * 1)
        if n_currency == 0:
            ku_close = np.array(range(len(close)))
        else:
            ku_close = i_ku_close(timeframe, shift, aud=aud, cad=cad, chf=chf,
                                  eur=eur, gbp=gbp, jpy=jpy, nzd=nzd, usd=usd)
            ku_close = np.array(ku_close)
            if n_currency == 1:
                ku_close = ku_close.reshape(len(ku_close), 1)
        # リスト内包表記も試したが、特に速度は変わらない。
        n = len(close)
        std_dev_linear_regression = np.empty(n)
        for i in range(period, n):
            x = ku_close[i-period:i]
            y = close[i-period:i]
            std_dev_linear_regression[i] = calc_std_dev_linear_regression(x, y)
        std_dev_linear_regression = pd.Series(std_dev_linear_regression,
                                              index=index)
        std_dev_linear_regression = std_dev_linear_regression.fillna(
            method='ffill')
        std_dev_linear_regression = std_dev_linear_regression.fillna(
            method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(std_dev_linear_regression, path)
    return std_dev_linear_regression

def i_std_dev_tree_regression(symbol, timeframe, period, shift, aud=0.0,
                              cad=0.0, chf=0.0, eur=0.0, gbp=0.0, jpy=0.0,
                              nzd=0.0, usd=0.0, max_depth=3):
    '''決定木による予測値との標準偏差を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
        aud: 豪ドル。
        cad: カナダドル。
        chf: スイスフラン。
        eur: ユーロ。
        gbp: ポンド。
        jpy: 円。
        nzd: NZドル。
        usd: 米ドル。
        max_depth: 最大の深さ。
    Returns:
        決定木による予測値との標準偏差。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_std_dev_tree_regression_' +
        symbol + str(timeframe) + '_' + str(period) + '_' + str(shift) + '_' +
        str(aud) + str(cad) + str(chf) + str(eur) + str(gbp) + str(jpy) +
        str(nzd) + str(usd) + '_' + str(max_depth) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        std_dev_tree_regression = joblib.load(path)
    # さもなければ計算する。
    else:
        def calc_std_dev_tree_regression(x, y, max_depth):
            clf = tree.DecisionTreeRegressor(max_depth=max_depth)
            clf.fit(x, y)
            pred = clf.predict(x)
            error = y - pred
            return np.std(error)

        close = i_close(symbol, timeframe, shift)
        index = close.index
        close = np.array(close)
        n_currency = ((
            (aud > 0.0) + (cad > 0.0) + (chf > 0.0) + (eur > 0.0) +
            (gbp > 0.0) + (jpy > 0.0) + (nzd > 0.0) + (usd > 0.0)) * 1)
        if n_currency == 0:
            ku_close = np.array(range(len(close)))
        else:
            ku_close = i_ku_close(timeframe, period, shift, aud=aud, cad=cad,
                                  chf=chf, eur=eur, gbp=gbp, jpy=jpy, nzd=nzd,
                                  usd=usd)
            ku_close = np.array(ku_close)
            if n_currency == 1:
                ku_close = ku_close.reshape(len(ku_close), 1)
        # リスト内包表記も試したが、特に速度は変わらない。
        n = len(close)
        std_dev_tree_regression = np.empty(n)
        for i in range(period, n):
            x = ku_close[i-period:i]
            y = close[i-period:i]
            std_dev_tree_regression[i] = calc_std_dev_tree_regression(x, y,
                max_depth)
        std_dev_tree_regression = pd.Series(std_dev_tree_regression,
                                            index=index)
        std_dev_tree_regression = std_dev_tree_regression.fillna(method='ffill')
        std_dev_tree_regression = std_dev_tree_regression.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(std_dev_tree_regression, path)
    return std_dev_tree_regression

def i_stop_hunting_zone(symbol, timeframe, period, shift):
    '''ストップ狩りのゾーンにあるか否かを返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 計算期間。
        shift: シフト。
    Returns:
        ストップ狩りのゾーンにあるか否か。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_stop_hunting_zone_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
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
            n = 10
        else:
            n = 1000
        close0 = i_close(symbol, timeframe, shift)
        hl_band1 = i_hl_band(symbol, timeframe, period, shift + 1)
        high_ceil = np.ceil(hl_band1['high'] * n) / n
        high_floor = np.floor(hl_band1['high'] * n) / n
        low_ceil = np.ceil(hl_band1['low'] * n) / n
        low_floor = np.floor(hl_band1['low'] * n) / n
        upper = pd.Series(index=close0.index)
        lower = pd.Series(index=close0.index)
        upper[(close0 >= high_floor) & (close0 <= high_ceil)] = 1
        lower[(close0 >= low_floor) & (close0 <= low_ceil)] = 1
        stop_hunting_zone = pd.DataFrame(index=close0.index)
        stop_hunting_zone['upper'] = upper
        stop_hunting_zone['lower'] = lower
        stop_hunting_zone = stop_hunting_zone.fillna(0)
        stop_hunting_zone = stop_hunting_zone.astype(bool)
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(stop_hunting_zone, path)
    return stop_hunting_zone

def i_tree_regression(symbol, timeframe, period, shift, aud=0.0, cad=0.0,
                      chf=0.0, eur=0.0, gbp=0.0, jpy=0.0, nzd=0.0, usd=0.0,
                      max_depth=3):
    '''決定木による予測値を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
        aud: 豪ドル。
        cad: カナダドル。
        chf: スイスフラン。
        eur: ユーロ。
        gbp: ポンド。
        jpy: 円。
        nzd: NZドル。
        usd: 米ドル。
        max_depth: 最大の深さ。
    Returns:
        決定木による予測値。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_tree_regression_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '_' + str(aud) +
        str(cad) + str(chf) + str(eur) + str(gbp) + str(jpy) + str(nzd) +
        str(usd) + '_' + str(max_depth) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        tree_regression = joblib.load(path)
    # さもなければ計算する。
    else:
        def calc_tree_regression(x, y, period, max_depth):
            clf = tree.DecisionTreeRegressor(max_depth=max_depth)
            clf.fit(x, y)
            pred = clf.predict(x)
            return pred[period-1]

        close = i_close(symbol, timeframe, shift)
        index = close.index
        close = np.array(close)
        n_currency = ((
            (aud > 0.0) + (cad > 0.0) + (chf > 0.0) + (eur > 0.0) +
            (gbp > 0.0) + (jpy > 0.0) + (nzd > 0.0) + (usd > 0.0)) * 1)
        if n_currency == 0:
            ku_close = np.array(range(len(close)))
        else:
            ku_close = i_ku_close(timeframe, period, shift, aud=aud, cad=cad,
                                  chf=chf, eur=eur, gbp=gbp, jpy=jpy, nzd=nzd,
                                  usd=usd)
            ku_close = np.array(ku_close)
            if n_currency == 1:
                ku_close = ku_close.reshape(len(ku_close), 1)
        # リスト内包表記も試したが、特に速度は変わらない。
        n = len(close)
        tree_regression = np.empty(n)
        for i in range(period, n):
            x = ku_close[i-period:i]
            y = close[i-period:i]
            tree_regression[i] = calc_tree_regression(x, y, period, max_depth)
        tree_regression = pd.Series(tree_regression, index=index)
        tree_regression = tree_regression.fillna(method='ffill')
        tree_regression = tree_regression.fillna(method='bfill')
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(tree_regression, path)
    return tree_regression

def i_updown(symbol, timeframe, shift):
    '''騰落を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        shift: シフト。
    Returns:
        騰落。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_updown_' + symbol +
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
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(updown, path)
    return updown

def i_vix4fx(symbol, timeframe, shift):
    '''1ヶ月当たりのボラティリティの予測値を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        shift: シフト。
    Returns:
        1ヶ月当たりのボラティリティの予測値。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_vix4fx_' + symbol +
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
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(vix4fx, path)
    return vix4fx

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
    path = (os.path.dirname(__file__) + '/tmp/i_volume_' + symbol +
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
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(volume, path)
    return volume

def i_zresid(symbol, timeframe, period, shift):
    '''終値とその予測値（移動平均）との標準化残差を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
    Returns:
        終値とその予測値（移動平均）との標準化残差。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_zresid_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + str(shift) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        zresid = joblib.load(path)
    # さもなければ計算する。
    else:
        # 終値を格納する。
        close = i_close(symbol, timeframe, shift)
        # 予測値、標準誤差を格納する。
        pred = i_ma(symbol, timeframe, period, shift)
        se = i_std_dev(symbol, timeframe, period, shift)
        # 標準化残差を計算する。
        zresid = (close - pred) / se
        zresid = zresid.fillna(0.0)
        zresid[(zresid==float('inf')) | (zresid==float('-inf'))] = 0.0
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(zresid, path)
    return zresid

def i_zresid_linear_regression(symbol, timeframe, period, shift, aud=0.0,
                               cad=0.0, chf=0.0, eur=0.0, gbp=0.0, jpy=0.0,
                               nzd=0.0, usd=0.0):
    '''終値とその予測値（線形回帰）との標準化残差を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
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
        終値とその予測値（線形回帰）との標準化残差。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_zresid_linear_regression_' +
        symbol + str(timeframe) + '_' + str(period) + '_' + str(shift) + '_' +
        str(aud) + str(cad) + str(chf) + str(eur) + str(gbp) + str(jpy) +
        str(nzd) + str(usd) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        zresid_linear_regression = joblib.load(path)
    # さもなければ計算する。
    else:
        # 終値を格納する。
        close = i_close(symbol, timeframe, shift)
        # 予測値、標準誤差を格納する。
        pred = i_linear_regression(symbol, timeframe, period, shift, aud=aud,
                                   cad=cad, chf=chf, eur=eur, gbp=gbp, jpy=jpy,
                                   nzd=nzd, usd=usd)
        se = i_std_dev_linear_regression(symbol, timeframe, period, shift,
                                         aud=aud, cad=cad, chf=chf, eur=eur,
                                         gbp=gbp, jpy=jpy, nzd=nzd, usd=usd)
        # 標準化残差を計算する。
        zresid_linear_regression = (close - pred) / se
        zresid_linear_regression = zresid_linear_regression.fillna(0.0)
        zresid_linear_regression[(zresid_linear_regression==float('inf')) |
            (zresid_linear_regression==float('-inf'))] = 0.0
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(zresid_linear_regression, path)
    return zresid_linear_regression

def i_zresid_tree_regression(symbol, timeframe, period, shift, aud=0.0, cad=0.0,
                             chf=0.0, eur=0.0, gbp=0.0, jpy=0.0, nzd=0.0,
                             usd=0.0, max_depth=3):
    '''終値とその予測値（決定木）との標準化残差を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
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
        終値とその予測値（決定木）との標準化残差。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_zresid_tree_regression_' +
        symbol + str(timeframe) + '_' + str(period) + '_' + str(shift) + '_' +
        str(aud) + str(cad) + str(chf) + str(eur) + str(gbp) + str(jpy) +
        str(nzd) + str(usd) + '_' + str(max_depth) + '.pkl')
    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if OANDA is None and os.path.exists(path) == True:
        zresid_tree_regression = joblib.load(path)
    # さもなければ計算する。
    else:
        # 終値を格納する。
        close = i_close(symbol, timeframe, shift)
        # 予測値、標準誤差を格納する。
        pred = i_tree_regression(symbol, timeframe, period, shift, aud=aud,
                                 cad=cad, chf=chf, eur=eur, gbp=gbp, jpy=jpy,
                                 nzd=nzd, usd=usd, max_depth=max_depth)
        se = i_std_dev_tree_regression(symbol, timeframe, period, shift,aud=aud,
                                       cad=cad, chf=chf, eur=eur, gbp=gbp,
                                       jpy=jpy, nzd=nzd, usd=usd,
                                       max_depth=max_depth)
        # 標準化残差を計算する。
        zresid_tree_regression = (close - pred) / se
        zresid_tree_regression = zresid_tree_regression.fillna(0.0)
        zresid_tree_regression[(zresid_tree_regression==float('inf')) |
            (zresid_tree_regression==float('-inf'))] = 0.0
        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if OANDA is None:
            # 一時フォルダーがなければ作成する。
            if os.path.exists(os.path.dirname(__file__) + '/tmp') == False:
                os.mkdir(os.path.dirname(__file__) + '/tmp')
            joblib.dump(zresid_tree_regression, path)
    return zresid_tree_regression

def is_trading_hours(index, market):
    '''取引時間であるか否かを返す。
    Args:
        index: インデックス。
        market: 市場
    Returns:
        取引時間であるか否か。
    '''
    # 東京時間の場合。
    if market == 'tokyo':
        trading_hours = ((index.hour>=2) & (index.hour<8)) * 1
        trading_hours = pd.Series(trading_hours, index=index)
        # 夏時間の大まかな調整。
        trading_hours[(trading_hours.index.month>=3)
            & (trading_hours.index.month<11)] = (((index.hour>=3)
            & (index.hour<9))) * 1
    # ロンドン時間の場合。
    elif market == 'ldn':
        trading_hours = ((index.hour>=10) & (index.hour<18)) * 1
        trading_hours = pd.Series(trading_hours, index=index)
        trading_hours[(trading_hours.index.hour==18)
            & (trading_hours.index.minute<30)] = 1
    # NY時間の場合。
    else:
        trading_hours = ((index.hour>=17) & (index.hour<23)) * 1
        trading_hours = pd.Series(trading_hours, index=index)
        trading_hours[(trading_hours.index.hour==16)
            & (trading_hours.index.minute>=30)] = 1
    return trading_hours

def minute():
    '''現在の分を返す。
    Returns:
        現在の分。
    '''    
    minute = datetime.now().minute
    return minute

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

def show_backtest_result(ret, trades, timeframe, start, end, parameter_ea1,
                parameter_ea2, parameter_ea3, parameter_ea4, parameter_ea5):
    '''バックテストの結果を表示する。
    Args:
        ret: リターン。
        trades: トレード数。
        timeframe: タイムフレーム。
        start: 開始日。
        end: 終了日。
        parameter_ea1: EA1のパラメータ。
        parameter_ea2: EA2のパラメータ。
        parameter_ea3: EA3のパラメータ。
        parameter_ea4: EA4のパラメータ。
        parameter_ea5: EA5のパラメータ。
    '''
    apr = calc_apr(ret, start, end)
    sharpe = calc_sharpe(ret, start, end)
    kelly = calc_kelly(ret)
    drawdowns = calc_drawdowns(ret)
    durations = calc_durations(ret, timeframe)
    # グラフを作成する。
    cum_ret = ret.cumsum()
    graph = cum_ret.plot()
    graph.set_xlabel('date')
    graph.set_ylabel('cumulative return')
    # レポートを作成する。
    report =  pd.DataFrame(index=[0])
    report['start'] = start.strftime('%Y.%m.%d')
    report['end'] = end.strftime('%Y.%m.%d')
    report['trades'] = trades
    report['apr'] = apr
    report['sharpe'] = sharpe
    report['kelly'] = kelly
    report['drawdowns'] = drawdowns
    report['durations'] = durations
    if parameter_ea1 is not None:
        report['parameter_ea1'] = str(parameter_ea1)
    if parameter_ea2 is not None:
        report['parameter_ea2'] = str(parameter_ea2)
    if parameter_ea3 is not None:
        report['parameter_ea3'] = str(parameter_ea3)
    if parameter_ea4 is not None:
        report['parameter_ea4'] = str(parameter_ea4)
    if parameter_ea5 is not None:
        report['parameter_ea5'] = str(parameter_ea5)
    # グラフを出力する。
    plt.show(graph)
    plt.close()
    # レポートを出力する。
    pd.set_option('display.width', 1000)
    print(report)

def show_walkforwardtest_result(ret, trades, timeframe, start, end):
    '''ウォークフォワードテストの結果を表示する。
    Args:
        ret: リターン。
        trades: トレード数。
        timeframe: タイムフレーム。
        start: 開始日。
        end: 終了日。
    '''
    apr = calc_apr(ret, start, end)
    sharpe = calc_sharpe(ret, start, end)
    kelly = calc_kelly(ret)
    drawdowns = calc_drawdowns(ret)
    durations = calc_durations(ret, timeframe)
    # グラフを作成する。
    cum_ret = ret.cumsum()
    graph = cum_ret.plot()
    graph.set_xlabel('date')
    graph.set_ylabel('cumulative return')
    # レポートを作成する。
    report = pd.DataFrame(index=[0])
    report['start'] = start.strftime('%Y.%m.%d')
    report['end'] = end.strftime('%Y.%m.%d')
    report['trades'] = trades
    report['apr'] = apr
    report['sharpe'] = sharpe
    report['kelly'] = kelly
    report['drawdowns'] = drawdowns
    report['durations'] = int(durations)
    # グラフを出力する。
    plt.show(graph)
    plt.close()
    # レポートを出力する。
    pd.set_option('display.width', 1000)
    print(report)

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

def trade(*args):
    '''トレードを行う。
    Args:
        calc_signal: シグナルを計算する関数。
        args: 引数。
        parameter: デフォルトのパラメータ。
        strategy: 戦略名。
    '''
    # グローバル変数を設定する。
    global LOCK
    global OANDA
    global ENVIRONMENT
    global ACCESS_TOKEN
    global ACCOUNT_ID
    # 設定を格納する。
    ea = args[0]
    symbol = args[1]
    timeframe = int(args[2])
    lots = float(args[3])
    position = 2
    mail = 0
    mt4 = 0
    n = len(args)
    if n >= 5:
          position = int(args[4])
    if n >= 6:
        mail = int(args[5])
    if n == 7:
        mt4 = int(args[6])
    exec('import ' + ea + ' as ea1')
    strategy = eval('ea1.strategy')
    parameter = eval('ea1.PARAMETER')
    # 設定ファイルを読み込む。
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
                    signal = strategy(parameter, symbol, timeframe, position)
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
                                'で売りエグジットです。')
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
                                'で買いエントリーです。')
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

def walkforwardtest(args):
    '''ウォークフォワードテストテストを行う。
    Args:
        args: 引数。
    Returns:
        リターン、トレード数、パラメータ、タイムフレーム、開始日、終了日。
    '''
    # 一時フォルダが残っていたら削除する。
    path = os.path.dirname(__file__)
    if os.path.exists(path + '/tmp') == True:
        shutil.rmtree(path + '/tmp')
    # 一時フォルダを作成する。
    os.mkdir(path + '/tmp')
    # 設定を格納する。
    n = len(args)
    ea = args[0]
    symbol = args[1]
    timeframe = int(args[2])
    start = datetime.strptime(args[3], '%Y.%m.%d')
    end = datetime.strptime(args[4], '%Y.%m.%d')
    spread = int(args[5])
    position = 2  # デフォルト値
    min_trade = 260  # デフォルト値
    in_sample_period = 360
    out_of_sample_period = 30
    if n >= 7:
          position = int(args[6])
    if n >= 8:
          min_trade = int(args[7])
    if n >= 9:
        in_sample_period = int(args[8])
    if n == 10:
        out_of_sample_period = int(args[9])
    exec('import ' + ea + ' as ea_file')
    strategy = eval('ea_file.strategy')
    rranges = eval('ea_file.RRANGES')
    # パフォーマンスを計算する関数を定義する。
    def calc_performance(parameter, strategy, symbol, timeframe, start, end,
                         spread, position, min_trade, optimization):
        '''パフォーマンスを計算する。
        Args:
            parameter: 最適化するパラメータ。
            strategy: 戦略を記述した関数。
            symbol: 通貨ペア名。
            timeframe: 足の種類。
            start: 開始日。
            end: 終了日。
            spread: スプレッド。
            position: ポジションの設定。  0: 買いのみ。  1: 売りのみ。  2: 売買両方。
            min_trade: 最低トレード数。
            optimization: 最適化の設定。  0: 最適化なし。  1: 最適化あり。
        Returns:
            最適化ありの場合は適応度、なしの場合はリターン, トレード数, 年率,
            シャープレシオ, 最適レバレッジ, ドローダウン, ドローダウン期間。
        '''
        # パフォーマンスを計算する。
        signal = strategy(parameter, symbol, timeframe, position)
        ret = calc_ret(symbol, timeframe, signal, spread, start, end)
        trades = calc_trades(signal, start, end)
        sharpe = calc_sharpe(ret, start, end)
        # 最適化する場合、適応度の符号を逆にして返す（最適値=最小値のため）。
        if optimization == 1:
            years = (end - start).total_seconds() / 60 / 60 / 24 / 365
            # 1年当たりのトレード数が最低トレード数に満たない場合、
            # 適応度を0にする。
            if trades / years >= min_trade:
                fitness = sharpe
            else:
                fitness = 0.0
            return -fitness
        # 最適化しない場合、各パフォーマンスを返す。
        else:
            return ret, trades, signal
    end_test = start
    # ウォークフォワードテストを行う。
    i = 0
    while True:
        start_train = start + timedelta(days=out_of_sample_period*i)
        end_train = (start_train + timedelta(days=in_sample_period)
            - timedelta(minutes=timeframe))
        start_test = end_train + timedelta(minutes=timeframe)
        end_test = (start_test + timedelta(days=out_of_sample_period)
            - timedelta(minutes=timeframe))

        if end_test > end:
            break
        result = optimize.brute(
            calc_performance, rranges, args=(strategy, symbol, timeframe,
            start_train, end_train, spread, position, min_trade, 1),
            finish=None)
        parameter = result
        ret, trades, signal = (
            calc_performance(parameter, strategy, symbol, timeframe,
            start_test, end_test, spread, position, min_trade, 0))

        if i == 0:
            signal_all = signal[start_test:end_test]
            start_all = start_test
        else:
            signal_all = signal_all.append(signal[start_test:end_test])
            end_all = end_test

        i = i + 1
    # 全体のパフォーマンスを計算する。
    ret_all = calc_ret(symbol, timeframe, signal_all, spread,
                            start_all, end_all)
    trades_all = calc_trades(signal_all, start_all, end_all)
    return ret_all, trades_all, timeframe, start_all, end_all