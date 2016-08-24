# coding: utf-8

import configparser
import inspect
import matplotlib.pyplot as plt
import numpy as np
import oandapy
import os
import pandas as pd
import shutil
import smtplib
import time

from datetime import datetime
from datetime import timedelta
from email.mime.text import MIMEText
from numba import float64, int64, jit
from scipy import optimize
from sklearn import linear_model
from sklearn.externals import joblib

# Spyderのバグ（？）で警告が出るので無視する。
import warnings
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)

# モードを設定する。
MODE = None

# OANDA API用に設定する。
OANDA = None
ENVIRONMENT = None
ACCESS_TOKEN = None
ACCOUNT_ID = None
COUNT = 500

# 許容誤差を設定する。
EPS = 1.0e-5

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
    instruments = OANDA.get_prices(instruments=instrument)
    bid = instruments['prices'][0]['bid']

    return bid

def backtest(calc_signal, args, parameter, rranges, strategy):
    '''バックテストを行う。
    Args:
        calc_signal: シグナルを計算する関数。
        args: 引数。
        parameter: パラメータ。
        rranges: パラメータの設定。
        strategy: 戦略名。
    '''
    # 一時フォルダが残っていたら削除する。
    path = os.path.dirname(__file__)
    if os.path.exists(path + '/tmp') == True:
        shutil.rmtree(path + '/tmp')

    # 一時フォルダを作成する。
    os.mkdir(path + '/tmp')

    symbol = args.symbol
    timeframe = args.timeframe
    start = datetime.strptime(args.start, '%Y.%m.%d')
    end = datetime.strptime(args.end, '%Y.%m.%d')
    spread = args.spread
    optimization = args.optimization
    position = args.position
    min_trade = args.min_trade

    # スプレッドの単位の調整
    if (symbol == 'AUDJPY' or symbol == 'CADJPY' or symbol == 'CHFJPY' or
        symbol == 'EURJPY' or symbol == 'GBPJPY' or symbol == 'NZDJPY' or
        symbol == 'USDJPY'):
        spread = spread / 1000.0
    else:
        spread = spread / 100000.0

    def calc_performance(parameter, calc_signal, symbol, timeframe, start, end, 
                         spread, optimization, position, min_trade):
        '''パフォーマンスを計算する。
        Args:
            parameter: 最適化するパラメータ。
            calc_signal: シグナルを計算する関数。
            symbol: 通貨ペア名。
            timeframe: タイムフレーム。
            start: 開始日。
            end: 終了日。
            spread: スプレッド。
            optimization: 最適化の設定。
            position: ポジションの設定。
            min_trade: 最低トレード数。
        Returns:
            最適化の場合は適応度、そうでない場合はリターン, トレード数, 年率,
            シャープレシオ, 最適レバレッジ, ドローダウン, ドローダウン期間。
        '''
        # パフォーマンスを計算する。
        signal = calc_signal(parameter, symbol, timeframe, start, end, spread,
                             optimization, position, min_trade)
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
            apr = calc_apr(ret, start, end)
            kelly = calc_kelly(ret)
            drawdowns = calc_drawdowns(ret)
            durations = calc_durations(ret, timeframe)

            return ret, trades, apr, sharpe, kelly, drawdowns, durations

    # バックテストを行う。
    if optimization == 1:
        result = optimize.brute(
            calc_performance, rranges, args=(calc_signal, symbol, timeframe,
                start, end, spread, 1, position, min_trade), finish=None)
        parameter = result

    ret, trades, apr, sharpe, kelly, drawdowns, durations = (
        calc_performance(parameter, calc_signal, symbol, timeframe,start, end,
        spread, 0, position, min_trade))

    # グラフを作成する。
    cum_ret = ret.cumsum()
    graph = cum_ret.plot()
    graph.set_title(strategy + '(' + symbol + str(timeframe) + ')')
    graph.set_xlabel('date')
    graph.set_ylabel('cumulative return')

    # レポートを作成する。
    report =  pd.DataFrame(
        index=[0], columns=['start','end', 'trades', 'apr', 'sharpe', 'kelly',
        'drawdowns', 'durations', 'parameter'])
    report['start'] = start.strftime('%Y.%m.%d')
    report['end'] = end.strftime('%Y.%m.%d')
    report['trades'] = trades
    report['apr'] = apr
    report['sharpe'] = sharpe
    report['kelly'] = kelly
    report['drawdowns'] = drawdowns
    report['durations'] = durations
    report['parameter'] = str(parameter)

    # グラフを出力する。
    plt.show(graph)
    plt.close()

    # レポートを出力する。
    pd.set_option('line_width', 1000)
    print('strategy = ', strategy)
    print('symbol = ', symbol)
    print('timeframe = ', str(timeframe))
    print(report)
 
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
    op = i_open(symbol, timeframe, 0)

    # コストを計算する。
    temp1 = (spread * ((signal > 0) & (signal > signal.shift(1))) *
        (signal - signal.shift(1)))
    temp2 = (spread * ((signal < 0) & (signal < signal.shift(1))) *
        (signal.shift(1) - signal))
    cost = temp1 + temp2

    # リターンを計算する。
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

def forex_system(calc_signal, parser, parameter, rranges):
    '''各モードのプログラムを行う。
    Args:
        calc_signal: シグナルを計算する関数。
        parser: パーサー。
        parameter: パラメータ。
        rranges: パラメータの設定。
    '''   
    # 開始時間を記録する。
    start_time = time.time()

    global MODE

    parser.add_argument('--mode', type=str)
    parser.add_argument('--symbol', type=str)
    parser.add_argument('--timeframe', type=int)
    parser.add_argument('--start', type=str)
    parser.add_argument('--end', type=str)
    parser.add_argument('--spread', type=int)
    parser.add_argument('--optimization', type=int, default=0)
    parser.add_argument('--position', type=int, default=2)
    parser.add_argument('--min_trade', type=int, default=260)
    parser.add_argument('--in_sample_period', type=int, default=360)
    parser.add_argument('--out_of_sample_period', type=int, default=30)
    parser.add_argument('--lots', type=float, default=0.1)
    parser.add_argument('--mail', type=int, default=0)
    parser.add_argument('--mt4', type=int, default=0)

    args = parser.parse_args()

    # 戦略名を取得する。
    path = os.path.dirname(__file__)
    temp = inspect.currentframe().f_back.f_code.co_filename
    temp = temp.replace(path, '')
    temp = temp.replace('/', '')
    strategy = temp.replace('.py', '')

    if args.mode == 'trade':
        MODE = 'trade'
        trade(calc_signal, args, parameter, strategy)
    elif args.mode == 'backtest':
        MODE = 'backtest'
        backtest(calc_signal, args, parameter, rranges, strategy)
    else:
        MODE = 'walkforwardtest'
        walkforwardtest(calc_signal, args, rranges, strategy)

    # 終了時間を記録する。
    end_time = time.time()

    # 実行時間を出力する。
    if end_time - start_time < 60.0:
        print(
            '実行時間は',
            int(round(end_time - start_time)), '秒です。')
    else:
        print(
            '実行時間は',
            int(round((end_time - start_time) / 60.0)), '分です。')

def hour():
    '''現在の時を返す。
    Returns:
        現在の時。
    '''    
    hour = datetime.now().hour

    return hour

def i_bandwalk(symbol, timeframe, period, method, shift):
    '''バンドウォークを返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        method: メソッド。
        shift: シフト。
    Returns:
        バンドウォーク。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_bandwalk_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + method + '_' + str(shift) +
        '.pkl')

    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if ((MODE == 'backtest' or MODE == 'walkforwardtest') and
        os.path.exists(path) == True):
        bandwalk = joblib.load(path)
    # さもなければ計算する。
    else:
        # バンドウォークを計算する関数を定義する。
        @jit(float64[:](float64[:], float64[:], float64[:]),
            nopython=True, cache=True)
        def calc_bandwalk(high, low, pred):
            up = 0
            down = 0
            length = len(pred)
            bandwalk = np.empty(length)
            for i in range(length):
                if (low[i] > pred[i]):
                    up = up + 1
                else:
                    up = 0
                if (high[i] < pred[i]):
                    down = down + 1
                else:
                    down = 0
                bandwalk[i] = up - down
            return bandwalk

        # 高値、安値を格納する。
        high = i_high(symbol, timeframe, shift)
        low = i_low(symbol, timeframe, shift)

        # 予測値を計算する。
        pred, se = i_pred(symbol, timeframe, period, method, shift)
        index = pred.index

        # 高値、安値、予測値をnumpy配列に変換する。
        high = np.array(high)
        low = np.array(low)
        pred = np.array(pred)

        # バンドウォークを計算する。
        bandwalk = calc_bandwalk(high, low, pred)
        a = 0.903  # 指数（正規化するために経験的に導き出した数値）
        b = 0.393  # 切片（同上）
        bandwalk = bandwalk / (float(period) ** a + b)

        # Seriesに変換する。
        bandwalk = pd.Series(bandwalk, index=index)
        bandwalk = bandwalk.fillna(0)

        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if MODE == 'backtest' or MODE == 'walkforwardtest':
            joblib.dump(bandwalk, path)
 
    return bandwalk

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
    if MODE == 'trade':
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
    path = (os.path.dirname(__file__) + '/tmp/i_diff_' + symbol +
        str(timeframe) + '_' + str(shift) + '.pkl')

    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if ((MODE == 'backtest' or MODE == 'walkforwardtest') and
        os.path.exists(path) == True):
        diff = joblib.load(path)

    # さもなければ計算する。
    else:
        close = i_close(symbol, timeframe, shift)
        diff = close - close.shift(1)
        diff = diff.fillna(0.0)
        diff[(diff==float('inf')) | (diff==float('-inf'))] = 0.0

        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if MODE == 'backtest' or MODE == 'walkforwardtest':
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
    if MODE == 'trade':
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
        str(timeframe) + '_' + str(shift) + '.pkl')

    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if ((MODE == 'backtest' or MODE == 'walkforwardtest') and
        os.path.exists(path) == True):
        hl_band = joblib.load(path)

    # さもなければ計算する。
    else:
        close = i_close(symbol, timeframe, shift)
        hl_band = pd.DataFrame()
        hl_band['high'] = close.rolling(window=period).max()
        hl_band['low'] = close.rolling(window=period).min()
        hl_band['middle'] = (hl_band['high'] + hl_band['low']) / 2

        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if MODE == 'backtest' or MODE == 'walkforwardtest':
            joblib.dump(hl_band, path)

    return hl_band

def i_ku_bandwalk(timeframe, period, shift, aud=0.0, cad=0.0, chf=0.0, eur=1.0,
                  gbp=0.0, jpy=1.0, nzd=0.0, usd=1.0):
    '''Ku-Chartによるバンドウォークを返す。
    Args:
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
        aud: 豪ドル（デフォルト：不使用）。
        cad: カナダドル（デフォルト：不使用）。
        chf: スイスフラン（デフォルト：不使用）。
        eur: ユーロ（デフォルト：使用）。
        gbp: ポンド（デフォルト：不使用）。
        jpy: 円（デフォルト：使用）。
        nzd: NZドル（デフォルト：不使用）。
        usd: 米ドル（デフォルト：使用）。
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
    if ((MODE == 'backtest' or MODE == 'walkforwardtest') and
        os.path.exists(path) == True):
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
        if MODE == 'backtest' or MODE == 'walkforwardtest':
            joblib.dump(ku_bandwalk, path)

    return ku_bandwalk

def i_ku_close(timeframe, shift, aud=0.0, cad=0.0, chf=0.0, eur=1.0, gbp=0.0,
               jpy=1.0, nzd=0.0, usd=1.0):
    '''Ku-Chartによる終値を返す。
    Args:
        timeframe: タイムフレーム。
        shift: シフト。
        aud: 豪ドル（デフォルト：不使用）。
        cad: カナダドル（デフォルト：不使用）。
        chf: スイスフラン（デフォルト：不使用）。
        eur: ユーロ（デフォルト：使用）。
        gbp: ポンド（デフォルト：不使用）。
        jpy: 円（デフォルト：使用）。
        nzd: NZドル（デフォルト：不使用）。
        usd: 米ドル（デフォルト：使用）。
    Returns:
        Ku-Chartによる終値。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_ku_close_' + str(timeframe) +
        '_' + str(shift) + '_' + str(aud) + str(cad) + str(chf) + str(eur) +
        str(gbp) + str(jpy) + str(nzd) + str(usd) + '.pkl')

    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if ((MODE == 'backtest' or MODE == 'walkforwardtest') and
        os.path.exists(path) == True):
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
        if MODE == 'backtest' or MODE == 'walkforwardtest':
            joblib.dump(ku_close, path)

    return ku_close

def i_ku_zresid(timeframe, period, shift, aud=0.0, cad=0.0, chf=0.0, eur=1.0,
                 gbp=0.0, jpy=1.0, nzd=0.0, usd=1.0):
    '''Ku-Chartによる終値とその予測値との標準化残差を返す。
    Args:
        timeframe: タイムフレーム。
        period: 期間。
        shift: シフト。
        aud: 豪ドル（デフォルト：不使用）。
        cad: カナダドル（デフォルト：不使用）。
        chf: スイスフラン（デフォルト：不使用）。
        eur: ユーロ（デフォルト：使用）。
        gbp: ポンド（デフォルト：不使用）。
        jpy: 円（デフォルト：使用）。
        nzd: NZドル（デフォルト：不使用）。
        usd: 米ドル（デフォルト：使用）。
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
    if ((MODE == 'backtest' or MODE == 'walkforwardtest') and
        os.path.exists(path) == True):
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
        if MODE == 'backtest' or MODE == 'walkforwardtest':
            joblib.dump(ku_zresid, path)

    return ku_zresid

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
    if MODE == 'trade':
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
    if ((MODE == 'backtest' or MODE == 'walkforwardtest') and
        os.path.exists(path) == True):
        ma = joblib.load(path)

    # さもなければ計算する。
    else:
        close = i_close(symbol, timeframe, shift)
        ma = close.rolling(window=period).mean()
        ma = ma.fillna(method='ffill')
        ma = ma.fillna(method='bfill')

        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if MODE == 'backtest' or MODE == 'walkforwardtest':
            joblib.dump(ma, path)

    return ma

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
    if MODE == 'trade':
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
            joblib.dump(op, path)

    return op

def i_pred(symbol, timeframe, period, method, shift):
    '''予測値と標準誤差を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 計算期間。
        method: メソッド。
        shift: シフト。
    Returns:
        予測値と標準誤差。
    '''
    # 計算結果の保存先のパスを格納する。
    path1 = (os.path.dirname(__file__) + '/tmp/i_pred1_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + method + '_' + str(shift) +
        '.pkl')
    path2 = (os.path.dirname(__file__) + '/tmp/i_pred2_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + method + '_' + str(shift) +
        '.pkl')

    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if ((MODE == 'backtest' or MODE == 'walkforwardtest') and
        os.path.exists(path1) == True and os.path.exists(path2) == True):
        pred = joblib.load(path1)
        se = joblib.load(path2)

    # さもなければ計算する。
    else:
        close = i_close(symbol, timeframe, shift)
        # メソッドが平均の場合、
        if method == 'mean':
            pred = close.rolling(window=period).mean()
            se = close.rolling(window=period).std()
        # メソッドが中央値の場合、
        elif method == 'median':
            def se_median(close):            
                pred = np.median(close)
                error = close - pred
                se = np.std(error)
                return se

            pred = close.rolling(window=period).median()
            se = close.rolling(window=period).apply(se_median)
        # メソッドが線形回帰の場合、
        elif method == 'linear':
            def pred_linear(close):
                clf = linear_model.LinearRegression()
                period = len(close)
                time = np.array(range(period))
                time = time.reshape(period, 1)
                clf.fit(time, close)
                pred = clf.predict(time)
                return pred[period-1]

            def se_linear(close):
                clf = linear_model.LinearRegression()
                period = len(close)
                time = np.array(range(period))
                time = time.reshape(period, 1)
                clf.fit(time, close)
                pred = clf.predict(time)
                error = close - pred
                se = np.std(error)
                return se

            pred = close.rolling(window=period).apply(pred_linear)
            se = close.rolling(window=period).apply(se_linear)
        # その他
        else:
            pred = pd.Series()
            se = pd.Series()

        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if MODE == 'backtest' or MODE == 'walkforwardtest':
            joblib.dump(pred, path1)
            joblib.dump(se, path2)

    return pred, se

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
    if ((MODE == 'backtest' or MODE == 'walkforwardtest') and
        os.path.exists(path) == True):
        ret = joblib.load(path)

    # さもなければ計算する。
    else:
        close = i_close(symbol, timeframe, shift)
        ret = (close - close.shift(1)) / close.shift(1)
        ret = ret.fillna(0.0)
        ret[(ret==float('inf')) | (ret==float('-inf'))] = 0.0

        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if MODE == 'backtest' or MODE == 'walkforwardtest':
            joblib.dump(ret, path)

    return ret

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
    if ((MODE == 'backtest' or MODE == 'walkforwardtest') and
        os.path.exists(path) == True):
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
        if MODE == 'backtest' or MODE == 'walkforwardtest':
            joblib.dump(stop_hunting_zone, path)

    return stop_hunting_zone

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
    if ((MODE == 'backtest' or MODE == 'walkforwardtest') and
        os.path.exists(path) == True):
        updown = joblib.load(path)

    # さもなければ計算する。
    else:
        close = i_close(symbol, timeframe, shift)
        updown = close >= close.shift(1)
        updown = updown.fillna(method='ffill')
        updown = updown.fillna(method='bfill')

        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if MODE == 'backtest' or MODE == 'walkforwardtest':
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
    if ((MODE == 'backtest' or MODE == 'walkforwardtest') and
        os.path.exists(path) == True):
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
        if MODE == 'backtest' or MODE == 'walkforwardtest':
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
    if MODE == 'trade':
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
            joblib.dump(volume, path)

    return volume

def i_zresid(symbol, timeframe, period, method, shift):
    '''終値とその予測値との標準化残差を返す。
    Args:
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        period: 期間。
        method: メソッド
        shift: シフト。
    Returns:
        終値とその予測値との標準化残差。
    '''
    # 計算結果の保存先のパスを格納する。
    path = (os.path.dirname(__file__) + '/tmp/i_zresid_' + symbol +
        str(timeframe) + '_' + str(period) + '_' + method + '_' + str(shift) +
        '.pkl')

    # バックテスト、またはウォークフォワードテストのとき、
    # 計算結果が保存されていれば復元する。
    if ((MODE == 'backtest' or MODE == 'walkforwardtest') and
        os.path.exists(path) == True):
        zresid = joblib.load(path)

    # さもなければ計算する。
    else:
        # 終値を格納する。
        close = i_close(symbol, timeframe, shift)
        # 予測値を格納する。
        pred, se = i_pred(symbol, timeframe, period, method, shift)
        # 標準化残差を計算する。
        zresid = (close - pred) / se
        zresid = zresid.fillna(0.0)
        zresid[(zresid==float('inf')) | (zresid==float('-inf'))] = 0.0

        # バックテスト、またはウォークフォワードテストのとき、保存する。
        if MODE == 'backtest' or MODE == 'walkforwardtest':
            joblib.dump(zresid, path)

    return zresid

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
        trading_hours = (index.hour>=0) & (index.hour<6)
        trading_hours = pd.Series(trading_hours, index=index)
        '''
        # 夏時間の大まかな調整。
        trading_hours[(trading_hours.index.month>=3)
            & (trading_hours.index.month<11)] = ((index.hour>=0)
            & (index.hour<6))
        '''

    # ロンドン時間の場合。
    elif market == 'ldn':
        trading_hours = (index.hour>=8) & (index.hour<16)
        trading_hours = pd.Series(trading_hours, index=index)
        trading_hours[(trading_hours.index.hour==16)
            & (trading_hours.index.minute<30)] = True

        # 夏時間の大まかな調整。
        trading_hours[(trading_hours.index.month>=3)
            & (trading_hours.index.month<11)] = ((index.hour>=7)
            & (index.hour<15))
        trading_hours[(trading_hours.index.month>=3)
            & (trading_hours.index.month<11)
            & (trading_hours.index.hour==15)
            & (trading_hours.index.minute<30)] = True

    # NY時間の場合。
    else:  # market == 'ny':
        trading_hours = (index.hour>=15) & (index.hour<21)
        trading_hours = pd.Series(trading_hours, index=index)
        trading_hours[(trading_hours.index.hour==14)
            & (trading_hours.index.minute>=30)] = True

        # 夏時間の大まかな調整。
        trading_hours[(trading_hours.index.month>=3)
            & (trading_hours.index.month<11)] = ((index.hour>=14)
            & (index.hour<20))
        trading_hours[(trading_hours.index.month>=3)
            & (trading_hours.index.month<11)
            & (trading_hours.index.hour==13)
            & (trading_hours.index.minute>=30)] = True

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
        曜日（日-土、0-6）。
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

def trade(calc_signal, args, parameter, strategy):
    '''トレードを行う。
    Args:
        calc_signal: シグナルを計算する関数。
        args: 引数。
        parameter: デフォルトのパラメータ。
        strategy: 戦略名。
    '''
    # グローバル変数を設定する。
    global OANDA
    global ENVIRONMENT
    global ACCESS_TOKEN
    global ACCOUNT_ID

    symbol = args.symbol
    timeframe = args.timeframe
    start = datetime.strptime('2001.01.01', '%Y.%m.%d')
    end = datetime.strptime('2100.12.31', '%Y.%m.%d')
    spread = args.spread
    optimization = args.optimization
    position = args.position
    min_trade = args.min_trade
    lots = args.lots
    mail = args.mail
    mt4 = args.mt4

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
    ENVIRONMENT = config['DEFAULT']['environment']
    ACCESS_TOKEN = config['DEFAULT']['access_token']
    ACCOUNT_ID = config['DEFAULT']['account_id']
    OANDA = oandapy.API(environment=ENVIRONMENT, access_token=ACCESS_TOKEN)
    pre_bid = 0.0
    pre_ask = 0.0
    count = COUNT
    pre_history_time = None
    instrument = convert_symbol2instrument(symbol)
    granularity = convert_timeframe2granularity(timeframe)

    # MT4用に設定する。
    if mt4 == 1:
        folder_ea = config['DEFAULT']['folder_ea']
        filename = folder_ea + '/' + strategy + '.csv'
        f = open(filename, 'w')
        f.write(str(2))  # EA側で-2とするので2で初期化する。
        f.close()

    pos = 0
    ticket = 0

    while True:
        # 回線接続を確認してから実行する。
        try:
            # 回線が接続していないと約15分でエラーになる。
            Bid = bid(instrument)  # 関数名とかぶるので先頭を大文字にした。
            Ask = ask(instrument)  # 同上。
            # tickが動いたときのみ実行する。
            if np.abs(Bid - pre_bid) >= EPS or np.abs(Ask - pre_ask) >= EPS:
                pre_bid = Bid
                pre_ask = Ask
                history = OANDA.get_history(
                    instrument=instrument, granularity=granularity,
                    count=count)
                history_time = history['candles'][count-1]['time']
                # 過去のデータの更新が完了してから実行する。
                # シグナルの計算は過去のデータを用いているので、過去のデータの
                # 更新が完了してから実行しないと計算がおかしくなる。
                if history_time != pre_history_time:
                    pre_history_time = history_time
                    signal = calc_signal(parameter, symbol, timeframe, start,
                        end, spread, optimization, position, min_trade)
                    end_row = len(signal) - 1
                    # エグジット→エントリーの順に設定しないとドテンができない。
                    # 買いエグジット。
                    if (pos == 1 and signal.iloc[end_row] != 1):
                        pos = 0
                        order_close(ticket)
                        ticket = 0
                        # メールにシグナルを送信する場合
                        if mail == 1:
                            subject = strategy
                            some_text = (symbol + 'を' +  str(Bid) +
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
                            subject = strategy
                            some_text = (symbol + 'を' +  str(Ask) +
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
                            subject = strategy
                            some_text = (symbol + 'を' +  str(Ask) +
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
                            subject = strategy
                            some_text = (symbol + 'を' +  str(Bid) +
                                'で売りエントリーです。')
                            send_mail(subject, some_text, fromaddr, toaddr,
                                      host, port, password)
                        # EAにシグナルを送信する場合
                        if mt4 == 1:
                            send_signal2mt4(filename, signal)

                # シグナルを出力する。
                now = datetime.now()
                print(now.strftime('%Y.%m.%d %H:%M:%S'), strategy, symbol,
                      timeframe, signal.iloc[end_row])

        except Exception as e:
            print('エラーが発生しました。')
            print(e)
            time.sleep(1) # 1秒おきに表示させる。

def walkforwardtest(calc_signal, args, rranges, strategy):
    '''ウォークフォワードテストを行う。
    Args:
        calc_signal: シグナルを計算する関数。
        args: 引数。
        rranges: パラメータの設定。
        strategy: 戦略名。
    '''
    # 一時フォルダが残っていたら削除する。
    path = os.path.dirname(__file__)
    if os.path.exists(path + '/tmp') == True:
        shutil.rmtree(path + '/tmp')

    # 一時フォルダを作成する。
    os.mkdir(path + '/tmp')

    symbol = args.symbol
    timeframe = args.timeframe
    start = datetime.strptime(args.start, '%Y.%m.%d')
    end = datetime.strptime(args.end, '%Y.%m.%d')
    spread = args.spread
    position = args.position
    min_trade = args.min_trade
    in_sample_period = args.in_sample_period
    out_of_sample_period = args.out_of_sample_period

    # スプレッドの単位の調整
    if (symbol == 'AUDJPY' or symbol == 'CADJPY' or symbol == 'CHFJPY' or
        symbol == 'EURJPY' or symbol == 'GBPJPY' or symbol == 'NZDJPY' or
        symbol == 'USDJPY'):
        spread = spread / 1000.0
    elif (symbol == 'AUDCAD' or symbol == 'AUDCHF' or symbol == 'AUDNZD' or
          symbol == 'AUDUSD' or symbol == 'CADCHF' or symbol == 'EURAUD' or
          symbol == 'EURCAD' or symbol == 'EURCHF' or symbol == 'EURGBP' or
          symbol == 'EURNZD' or symbol == 'EURUSD' or symbol == 'GBPAUD' or
          symbol == 'GBPCAD' or symbol == 'GBPCHF' or symbol == 'GBPNZD' or
          symbol == 'GBPUSD' or symbol == 'NZDCAD' or symbol == 'NZDCHF' or
          symbol == 'NZDUSD' or symbol == 'USDCAD' or symbol == 'USDCHF'):
        spread = spread / 100000.0
    else:
        pass

    end_test = start
    report =  pd.DataFrame(
        index=range(1000),
        columns=['start_train','end_train', 'start_test', 'end_test', 'trades',
                 'apr', 'sharpe', 'kelly', 'drawdowns', 'durations',
                 'parameter'])

    def calc_performance(parameter, calc_signal, symbol, timeframe, start, end,
                         spread, optimization, position, min_trade):
        '''パフォーマンスを計算する。
        Args:
            parameter: 最適化するパラメータ。
            calc_signal: シグナルを計算する関数。
            symbol: 通貨ペア名。
            timeframe: タイムフレーム。
            start: 開始日。
            end: 終了日。
            spread: スプレッド。
            optimization: 最適化の設定。
            position: ポジションの設定。
            min_trade: 最低トレード数。
        Returns:
            最適化の場合は適応度、そうでない場合はリターン, トレード数, 年率,
            シャープレシオ, 最適レバレッジ, ドローダウン, ドローダウン期間、
            シグナル。
        '''
        # パフォーマンスを計算する。
        signal = calc_signal(parameter, symbol, timeframe, start, end, spread,
                             optimization, position, min_trade)
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
        elif optimization == 0:
            apr = calc_apr(ret, start, end)
            kelly = calc_kelly(ret)
            drawdowns = calc_drawdowns(ret)
            durations = calc_durations(ret, timeframe)

            return (ret, trades, apr, sharpe, kelly, drawdowns, durations,
                    signal)

        else:
            pass

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
            calc_performance, rranges, args=(calc_signal, symbol, timeframe,
            start_train, end_train, spread, 1, position, min_trade),
            finish=None)
        parameter = result
        ret, trades, apr, sharpe, kelly, drawdowns, durations, signal = (
            calc_performance(parameter, calc_signal, symbol, timeframe,
            start_test, end_test, spread, 0, position, min_trade))

        report.iloc[i][0] = start_train.strftime('%Y.%m.%d')
        report.iloc[i][1] = end_train.strftime('%Y.%m.%d')
        report.iloc[i][2] = start_test.strftime('%Y.%m.%d')
        report.iloc[i][3] = end_test.strftime('%Y.%m.%d')
        report.iloc[i][4] = trades
        report.iloc[i][5] = apr
        report.iloc[i][6] = sharpe
        report.iloc[i][7] = kelly
        report.iloc[i][8] = drawdowns
        report.iloc[i][9] = durations
        report.iloc[i][10] = str(parameter)

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
    apr_all = calc_apr(ret_all, start_all, end_all)
    sharpe_all = calc_sharpe(ret_all, start_all, end_all)
    kelly_all = calc_kelly(ret_all)
    drawdowns_all = calc_drawdowns(ret_all)
    durations_all = calc_durations(ret_all, timeframe)

    # グラフを作成する。
    cum_ret = ret_all.cumsum()
    graph = cum_ret.plot()
    graph.set_title(strategy + '(' + symbol + str(timeframe) + ')')
    graph.set_xlabel('date')
    graph.set_ylabel('cumulative return')

    # レポートを作成する。
    report.iloc[i][0] = ''
    report.iloc[i][1] = ''
    report.iloc[i][2] = start_all.strftime('%Y.%m.%d')
    report.iloc[i][3] = end_all.strftime('%Y.%m.%d')
    report.iloc[i][4] = trades_all
    report.iloc[i][5] = apr_all
    report.iloc[i][6] = sharpe_all
    report.iloc[i][7] = kelly_all
    report.iloc[i][8] = drawdowns_all
    report.iloc[i][9] = int(durations_all)
    report.iloc[i][10] = ''

    # グラフを出力する。
    plt.show(graph)
    plt.close()

    # レポートを出力する。
    pd.set_option('line_width', 1000)
    print('strategy = ', strategy)
    print('symbol = ', symbol)
    print('timeframe = ', str(timeframe))
    print(report.iloc[:i+1, ])