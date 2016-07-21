# coding: utf-8

import matplotlib.pyplot as plt
import forex_system
import numpy as np
import pandas as pd
import struct
import time

from collections import OrderedDict
from datetime import datetime
from numba import float64, jit
from scipy.optimize import curve_fit

def calc_accuracy4win(*args):
    '''勝つための的中率を計算する。
      Args:
          *args: 可変長引数。
    '''

    width = float(args[0])  # 値幅
    cost = float(args[1])  # コスト
    profit = width - cost  # 的中した場合の利益
    loss = width + cost  # 的中しなかった場合の損失
    n = 100  # 試行回数
    a = 0.0  # 下限（初期値）
    b = 1.0  # 上限（初期値）
    prob = (a + b) / 2 # 的中率

    # 勝つための的中率を二分法で計算する。
    for i in (range(n)):
        if (profit * prob >= loss * (1 - prob)):
            b = prob
            prob = (a + b) / 2
        else:
            a = prob
            prob = (a + b) / 2

    print("勝つための的中率 = ", prob)

def calc_bandwalk_std(period, n):
    '''バンドウォークの標準偏差を計算する。
        Args:
            period: 計算期間。
            n: データ数。
        Returns:
              バンドウォークの標準偏差。
    '''

    temp = np.random.randn(n)
    close = temp.cumsum()
    v = np.ones(period) / period
    mean = np.convolve(close, v, 'valid')
    close = close[period-1:]  
 
    @jit(float64(float64[:], float64[:]), nopython=True, cache=True)
    def func(close, mean):
        bandwalk_mean_std = [0.0, 0.0]
        bandwalk_mean_std = np.array(bandwalk_mean_std)
        up = 0
        down = 0
        length = len(close)
        bandwalk = np.zeros(length)
        for i in range(length):
            if (close[i] > mean[i]):
                up = up + 1
            else:
                up = 0
            if (close[i] < mean[i]):
                down = down + 1
            else:
                down = 0
            bandwalk[i] = up - down
        return bandwalk.std()
 
    bandwalk_std = func(close, mean)

    return bandwalk_std

def calc_trades4win(*args):
    '''勝つためのトレード数を計算する。
      Args:
          *args: 可変長引数。
    '''

    target = float(args[0])  # 目標勝率
    wp = float(args[1])  # 1トレード当たりの想定勝率
    prob = 0.0
    n_trade = -1

    def choose(n, r):
        '''組み合わせの数を返す（PythonにRのchoose()関数に相当するものはないのか？）。
          Args:
              n: 異なるものの数。
              r: 選ぶ数。
          Returns:
              組み合わせの数。
        '''
        if n == 0 or r == 0:
            return 1
        else:
            return choose(n, r - 1) * (n - r + 1) / r

    while (prob < target):
        n_trade = n_trade + 2 # 必ず奇数にする
        prob = 0.0
        for i in range(int((n_trade + 1) / 2), n_trade + 1):
            r = i
            temp = choose(n_trade, r) * wp**r * (1 - wp)**(n_trade - r)
            prob = prob + temp

    print("週単位で勝つための1日当たりのトレード数 = ", n_trade / 5.0)
    print("月単位で勝つための1日当たりのトレード数 = ", n_trade / 20.0)

def calc_volatility4win(*args):
    '''勝つためのボラティリティを計算する。
      Args:
          *args: 可変長引数。
    '''

    prob = float(args[0])  # 的中率
    cost = float(args[1])  # コスト
    n = 100 # 試行回数
    a = 0.0 # 下限（初期値）
    b = 100.0 # 上限（初期値）
    width = (a + b) / 2 # ボラティリティ
    profit = width - cost # 的中した場合の利益
    loss = width + cost # 的中しなかった場合の損失

    # 勝つためのボラティリティを二分法で計算する。
    for i in range(n):
        if (profit * prob >= loss * (1 - prob)):
            b = width
            width = (a + b) / 2
            profit = width - cost
            loss = width + cost
        else:
            a = width
            width = (a + b) / 2
            profit = width - cost
            loss = width + cost

    print("勝つためのボラティリティ = ", width)

def calc_volatilities(*args):
    '''各足のボラティリティを計算する。
      Args:
          *args: 可変長引数。
    '''

    symbol = args[0]  # 通貨ペア
    start_year = int(args[1])  # 開始年
    start_month = int(args[2])  # 開始月
    start_day = int(args[3])  # 開始日
    end_year = int(args[4])  # 終了年
    end_month = int(args[5])  # 終了月
    end_day = int(args[6])  # 終了日

    start = datetime(start_year, start_month, start_day, 0, 0)
    end = datetime(end_year, end_month, end_day, 23, 59)
 
    # ボラティリティを計算する。
    fs = forex_system.ForexSystem()
    if (symbol == 'AUDJPY' or symbol == 'CADJPY' or symbol == 'CHFJPY' or
        symbol == 'EURJPY' or symbol == 'GBPJPY' or symbol == 'NZDJPY' or
        symbol == 'USDJPY'):
        values2pips = 100.0
    else:
    # symbol == 'AUDCAD', 'AUDCHF', 'AUDNZD', 'AUDUSD', 'CADCHF', 'EURAUD',
    #           'EURCAD', 'EURCHF', 'EURGBP', 'EURNZD', 'EURUSD', 'GBPAUD',
    #           'GBPCAD', 'GBPCHF', 'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF',
    #           'NZDUSD', 'USDCAD', 'USDCHF'
        values2pips = 10000.0
    vola1 = fs.i_diff(symbol, 1, 0)[start:end].std() * values2pips
    vola5 = fs.i_diff(symbol, 5, 0)[start:end].std() * values2pips
    vola15 = fs.i_diff(symbol, 15, 0)[start:end].std() * values2pips
    vola30 = fs.i_diff(symbol, 30, 0)[start:end].std() * values2pips
    vola60 = fs.i_diff(symbol, 60, 0)[start:end].std() * values2pips
    vola240 = fs.i_diff(symbol, 240, 0)[start:end].std() * values2pips
    vola1440 = fs.i_diff(symbol, 1440, 0)[start:end].std() * values2pips
 
    # ボラティリティを出力する。
    print("通貨ペア = ", symbol)
    print("1分足のボラティリティ = ", vola1)
    print("5分足のボラティリティ = ", vola5)
    print("15分足のボラティリティ = ", vola15)
    print("30分足のボラティリティ = ", vola30)
    print("1時間足のボラティリティ = ", vola60)
    print("4時間足のボラティリティ = ", vola240)
    print("日足のボラティリティ = ", vola1440)

def convert_hst2csv():
    '''hstファイルをcsvファイルに変換する。
    '''

    for i in range(28):
        if i == 0:
            symbol = 'AUDCAD'
        elif i == 1:
            symbol = 'AUDCHF'
        elif i == 2:
            symbol = 'AUDJPY'
        elif i == 3:
            symbol = 'AUDNZD'
        elif i == 4:
            symbol = 'AUDUSD'
        elif i == 5:
            symbol = 'CADCHF'
        elif i == 6:
            symbol = 'CADJPY'
        elif i == 7:
            symbol = 'CHFJPY'
        elif i == 8:
            symbol = 'EURAUD'
        elif i == 9:
            symbol = 'EURCAD'
        elif i == 10:
            symbol = 'EURCHF'
        elif i == 11:
            symbol = 'EURGBP'
        elif i == 12:
            symbol = 'EURJPY'
        elif i == 13:
            symbol = 'EURNZD'
        elif i == 14:
            symbol = 'EURUSD'
        elif i == 15:
            symbol = 'GBPAUD'
        elif i == 16:
            symbol = 'GBPCAD'
        elif i == 17:
            symbol = 'GBPCHF'
        elif i == 18:
            symbol = 'GBPJPY'
        elif i == 19:
            symbol = 'GBPNZD'
        elif i == 20:
            symbol = 'GBPUSD'
        elif i == 21:
            symbol = 'NZDCAD'
        elif i == 22:
            symbol = 'NZDCHF'
        elif i == 23:
            symbol = 'NZDJPY'
        elif i == 24:
            symbol = 'NZDUSD'
        elif i == 25:
            symbol = 'USDCAD'
        elif i == 26:
            symbol = 'USDCHF'
        else:
            symbol = 'USDJPY'

        # 端末のカレントディレクトリが「~」で、そこからの相対パスであることに注意する。
        filename_hst = '../historical_data/' + symbol + '.hst'
        # 同上
        filename_csv = '../historical_data/' + symbol + '.csv'
        read = 0
        datetime = []
        open_price = []
        low_price = []
        high_price = []
        close_price = []
        volume = []
        with open(filename_hst, 'rb') as f:
            while True:
                if read >= 148:
                    buf = f.read(44)
                    read += 44
                    if not buf:
                        break
                    bar = struct.unpack('< iddddd', buf)
                    datetime.append(
                        time.strftime('%Y-%m-%d %H:%M:%S',
                        time.gmtime(bar[0])))
                    open_price.append(bar[1])
                    high_price.append(bar[3])  # 高値と安値の順序が違うようだ
                    low_price.append(bar[2])  # 同上
                    close_price.append(bar[4])
                    volume.append(bar[5])
                else:
                    buf = f.read(148)
                    read += 148
        data = {'0_datetime':datetime, '1_open_price':open_price,
            '2_high_price':high_price, '3_low_price':low_price,
            '4_close_price':close_price, '5_volume':volume}
        result = pd.DataFrame.from_dict(data)
        result = result.set_index('0_datetime')
        result.to_csv(filename_csv, header = False)

def create_time_series_data(*args):
    '''擬似時系列データを作成する。
      Args:
          *args: 可変長引数。
    '''

    a = args[0]  # 係数。
    t = args[1]  # 時期（データ数と考えていい）。

    e = np.random.randn(t)
    y = np.empty(t)
    y[0] = 0.0

    for i in (range(1, t)):
        y[i] = a * y[i - 1] +  e[i]

    return y
 
def make_historical_data():
    '''ヒストリカルデータを作成する。
    '''

    # csvファイルを読み込む。
    for i in range(28):
        if i == 0:
            symbol = 'AUDCAD'
        elif i == 1:
            symbol = 'AUDCHF'
        elif i == 2:
            symbol = 'AUDJPY'
        elif i == 3:
            symbol = 'AUDNZD'
        elif i == 4:
            symbol = 'AUDUSD'
        elif i == 5:
            symbol = 'CADCHF'
        elif i == 6:
            symbol = 'CADJPY'
        elif i == 7:
            symbol = 'CHFJPY'
        elif i == 8:
            symbol = 'EURAUD'
        elif i == 9:
            symbol = 'EURCAD'
        elif i == 10:
            symbol = 'EURCHF'
        elif i == 11:
            symbol = 'EURGBP'
        elif i == 12:
            symbol = 'EURJPY'
        elif i == 13:
            symbol = 'EURNZD'
        elif i == 14:
            symbol = 'EURUSD'
        elif i == 15:
            symbol = 'GBPAUD'
        elif i == 16:
            symbol = 'GBPCAD'
        elif i == 17:
            symbol = 'GBPCHF'
        elif i == 18:
            symbol = 'GBPJPY'
        elif i == 19:
            symbol = 'GBPNZD'
        elif i == 20:
            symbol = 'GBPUSD'
        elif i == 21:
            symbol = 'NZDCAD'
        elif i == 22:
            symbol = 'NZDCHF'
        elif i == 23:
            symbol = 'NZDJPY'
        elif i == 24:
            symbol = 'NZDUSD'
        elif i == 25:
            symbol = 'USDCAD'
        elif i == 26:
            symbol = 'USDCHF'
        else:
            symbol = 'USDJPY'

        # 1分足の作成
        filename = '~/historical_data/' + symbol + '.csv'
        if i == 0:
            data = pd.read_csv(filename, header=None, index_col=0)
            data.index = pd.to_datetime(data.index)
        else:
            temp = pd.read_csv(filename, header=None, index_col=0)
            temp.index = pd.to_datetime(temp.index)
            data = pd.merge(
                data, temp, left_index=True, right_index=True, how='outer')

    # 列名を変更する。
    label = ['open', 'high', 'low', 'close', 'volume']
    data.columns = label * 28

    # リサンプリングの方法を設定する。
    ohlc_dict = OrderedDict()
    ohlc_dict['open'] = 'first'
    ohlc_dict['high'] = 'max'
    ohlc_dict['low'] = 'min'
    ohlc_dict['close'] = 'last'
    ohlc_dict['volume'] = 'sum'

    # 各足を作成する。
    for i in range(28):
        if i == 0:
            symbol = 'AUDCAD'
        elif i == 1:
            symbol = 'AUDCHF'
        elif i == 2:
            symbol = 'AUDJPY'
        elif i == 3:
            symbol = 'AUDNZD'
        elif i == 4:
            symbol = 'AUDUSD'
        elif i == 5:
            symbol = 'CADCHF'
        elif i == 6:
            symbol = 'CADJPY'
        elif i == 7:
            symbol = 'CHFJPY'
        elif i == 8:
            symbol = 'EURAUD'
        elif i == 9:
            symbol = 'EURCAD'
        elif i == 10:
            symbol = 'EURCHF'
        elif i == 11:
            symbol = 'EURGBP'
        elif i == 12:
            symbol = 'EURJPY'
        elif i == 13:
            symbol = 'EURNZD'
        elif i == 14:
            symbol = 'EURUSD'
        elif i == 15:
            symbol = 'GBPAUD'
        elif i == 16:
            symbol = 'GBPCAD'
        elif i == 17:
            symbol = 'GBPCHF'
        elif i == 18:
            symbol = 'GBPJPY'
        elif i == 19:
            symbol = 'GBPNZD'
        elif i == 20:
            symbol = 'GBPUSD'
        elif i == 21:
            symbol = 'NZDCAD'
        elif i == 22:
            symbol = 'NZDCHF'
        elif i == 23:
            symbol = 'NZDJPY'
        elif i == 24:
            symbol = 'NZDUSD'
        elif i == 25:
            symbol = 'USDCAD'
        elif i == 26:
            symbol = 'USDCHF'
        else:
            symbol = 'USDJPY'
 
        data1 = data.iloc[:, 0+(5*i): 5+(5*i)]

        # 欠損値を補間する。
        data1 = data1.fillna(method='ffill')
        data1 = data1.fillna(method='bfill')

        # 重複行を削除する。
        data1 = data1[~data1.index.duplicated()] 
 
        data5 = data1.resample(
            '5Min', label='left', closed='left').apply(ohlc_dict)
        data15 = data1.resample(
            '15Min', label='left', closed='left').apply(ohlc_dict)
        data30 = data1.resample(
            '30Min', label='left', closed='left').apply(ohlc_dict)
        data60 = data1.resample(
            '60Min', label='left', closed='left').apply(ohlc_dict)
        data240 = data1.resample(
            '240Min', label='left', closed='left').apply(ohlc_dict)
        data1440 = data1.resample(
            '1440Min', label='left', closed='left').apply(ohlc_dict)

        # 欠損値を削除する。
        data5 = data5.dropna()
        data15 = data15.dropna()
        data30 = data30.dropna()
        data60 = data60.dropna()
        data240 = data240.dropna()
        data1440 = data1440.dropna()

        # ファイルを出力する。
        filename1 =  '~/historical_data/' + symbol + '1.csv'
        filename5 =  '~/historical_data/' + symbol + '5.csv'
        filename15 =  '~/historical_data/' + symbol + '15.csv'
        filename30 =  '~/historical_data/' + symbol + '30.csv'
        filename60 =  '~/historical_data/' + symbol + '60.csv'
        filename240 =  '~/historical_data/' + symbol + '240.csv'
        filename1440 =  '~/historical_data/' + symbol + '1440.csv'
        data1.to_csv(filename1)
        data5.to_csv(filename5)
        data15.to_csv(filename15)
        data30.to_csv(filename30)
        data60.to_csv(filename60)
        data240.to_csv(filename240)
        data1440.to_csv(filename1440)

def output_bandwalk_mean_std(*args):
    '''バンドウォークの標準偏差を出力する。
      Args:
          *args: 可変長引数。
    '''
    max_period = args[0]
    n = args[1]

    bandwalk_std = np.empty(max_period)
 
    for i in range(max_period):
        period = i + 1
        bandwalk_std[i] = calc_bandwalk_std(period, n)

    # バンドウォークは計算期間が短いと不規則に見えるのでperiod=5から始める。
    bandwalk_std = bandwalk_std[4:]

    # モデルを作成する。
    def func(x, a, b):
        return x ** a + b

    # period=5から始める。
    x = list(range(5, max_period+1, 1))
    x = np.array(x)
    popt, pcov = curve_fit(func, x, bandwalk_std)
    a = popt[0]
    b = popt[1]
    model = x ** a + b

    # DataFrameに変換する。
    bandwalk_std = pd.Series(bandwalk_std)
    model = pd.Series(model)
    # グラフを出力する。
    result = pd.concat([bandwalk_std, model], axis=1)
    result.columns  = ['bandwalk_std', 'model']
    graph = result.plot()
    graph.set_xlabel('period')
    graph.set_ylabel('bandwalk')
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
               [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.show()
 
    # 傾きと切片を出力する。
    print('指数 = ', a)
    print('切片 = ', b)