import argparse
import configparser
import glob
import inspect
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import oandapy
import os
import pandas as pd
import smtplib
import struct
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from numba import float64, jit
#from pandas_datareader import data as web
from scipy import optimize
from scipy.stats import pearson3
from sklearn.externals import joblib

import pandas.plotting._converter as pandacnv
pandacnv.register()

UNITS = 100000
COUNT = 500
EPS = 1.0e-5

g_oanda = None
g_environment = None
g_access_token = None
g_access_id = None

def backtest(strategy, symbol, timeframe, spread, start, end, parameter):
    empty_folder('temp')
    create_folder('temp')
    start = datetime.strptime(start + ' 00:00', '%Y.%m.%d %H:%M')
    end = datetime.strptime(end + ' 00:00', '%Y.%m.%d %H:%M')
    end -= timedelta(minutes=timeframe)
    report =  pd.DataFrame(
            index=[''], columns=['start', 'end', 'trades', 'apr', 'sharpe',
                  'drawdown', 'parameter'])
    buy_entry, buy_exit, sell_entry, sell_exit = strategy(
            parameter, symbol, timeframe)
    signal = get_signal(buy_entry, buy_exit, sell_entry, sell_exit, symbol,
                        timeframe)
    pnl = get_pnl(signal, symbol, timeframe, spread, start, end)
    trades = get_trades(signal, start, end)
    apr = get_apr(pnl, timeframe)
    sharpe = get_sharpe(pnl, timeframe)
    drawdown = get_drawdown(pnl)
    report.iloc[0, 0] = start.strftime('%Y.%m.%d')
    report.iloc[0, 1] = end.strftime('%Y.%m.%d')
    report.iloc[0, 2] = str(trades)
    report.iloc[0, 3] = str(np.round(apr, 3))
    report.iloc[0, 4] = str(np.round(sharpe, 3))
    report.iloc[0, 5] = str(np.round(drawdown, 3))
    if parameter is not None:
        report.iloc[0, 6] = np.round(parameter, 3)
    report = report.dropna(axis=1)
    pd.set_option('display.max_columns', 100)
    print(report)
    equity = (1.0+pnl).cumprod() - 1.0
    ax=plt.subplot()
    ax.set_xticklabels(equity.index, rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.plot(equity)
    plt.title('Backtest')
    plt.xlabel('Date')
    plt.ylabel('Equith Curve')
    plt.tight_layout()
    plt.savefig('backtest.png', dpi=150)
    plt.show()
    plt.close()
    empty_folder('temp')
    return pnl

def backtest_ml(strategy, symbol, timeframe, spread, start, end, get_model,
                in_sample_period, out_of_sample_period):
    empty_folder('temp')
    create_folder('temp')
    start = datetime.strptime(start + ' 00:00', '%Y.%m.%d %H:%M')
    end = datetime.strptime(end + ' 00:00', '%Y.%m.%d %H:%M')
    end -= timedelta(minutes=timeframe)
    report =  pd.DataFrame(
            index=[['']*1000], columns=['start_test', 'end_test', 'trades',
                  'apr', 'sharpe', 'drawdown'])
    end_test = start
    i = 0
    while True:
        start_train = start + timedelta(days=out_of_sample_period*i)
        end_train = (start_train + timedelta(days=in_sample_period)
            - timedelta(minutes=timeframe))
        start_test = end_train + timedelta(minutes=timeframe)
        if i == 0:
            start_all = start_test
        if (start_test + timedelta(days=out_of_sample_period)
            - timedelta(minutes=timeframe)) > end:
            end_all = end_test
            break
        end_test = (start_test + timedelta(days=out_of_sample_period)
            - timedelta(minutes=timeframe))
        get_model(symbol, timeframe, start_train, end_train)
        buy_entry, buy_exit, sell_entry, sell_exit = strategy(
                None, symbol, timeframe)
        signal = get_signal(buy_entry, buy_exit, sell_entry, sell_exit,
                          symbol, timeframe)
        pnl = get_pnl(signal, symbol, timeframe, spread, start_test,
                      end_test)
        trades = get_trades(signal, start_test, end_test)
        apr = get_apr(pnl, timeframe)
        sharpe = get_sharpe(pnl, timeframe)
        drawdown = get_drawdown(pnl)
        report.iloc[i, 0] = start_test.strftime('%Y.%m.%d')
        report.iloc[i, 1] = end_test.strftime('%Y.%m.%d')
        report.iloc[i, 2] = str(trades)
        report.iloc[i, 3] = str(np.round(apr, 3))
        report.iloc[i, 4] = str(np.round(sharpe, 3))
        report.iloc[i, 5] = str(np.round(drawdown, 3))
        if i == 0:
            temp = signal[start_test:end_test]
        else:
            temp = temp.append(signal[start_test:end_test])
        i += 1
    signal = temp
    pnl = get_pnl(signal, symbol, timeframe, spread, start_all, end_all)
    trades = get_trades(signal, start_all, end_all)
    apr = get_apr(pnl, timeframe)
    sharpe = get_sharpe(pnl, timeframe)
    drawdown = get_drawdown(pnl)
    report.iloc[i, 0] = start_all.strftime('%Y.%m.%d')
    report.iloc[i, 1] = end_all.strftime('%Y.%m.%d')
    report.iloc[i, 2] = str(trades)
    report.iloc[i, 3] = str(np.round(apr, 3))
    report.iloc[i, 4] = str(np.round(sharpe, 3))
    report.iloc[i, 5] = str(np.round(drawdown, 3))
    report = report.iloc[0:i+1, :]
    report = report.dropna(axis=1)
    pd.set_option('display.max_columns', 100)
    print(report)
    equity = (1.0+pnl).cumprod() - 1.0
    ax=plt.subplot()
    ax.set_xticklabels(equity.index, rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.plot(equity)
    plt.title('Backtest')
    plt.xlabel('Date')
    plt.ylabel('Equity Curve')
    plt.tight_layout()
    plt.savefig('backtest.png', dpi=150)
    plt.show()
    plt.close()
    empty_folder('temp')
    return pnl

def backtest_opt(strategy, symbol, timeframe, spread, start, end, rranges,
                 min_trade):
    empty_folder('temp')
    create_folder('temp')
    start = datetime.strptime(start + ' 00:00', '%Y.%m.%d %H:%M')
    end = datetime.strptime(end + ' 00:00', '%Y.%m.%d %H:%M')
    end -= timedelta(minutes=timeframe)
    report =  pd.DataFrame(
            index=[''], columns=['start', 'end', 'trades', 'apr', 'sharpe',
                  'drawdown', 'parameter'])
    parameter = optimize_parameter(
            strategy, symbol, timeframe, spread, start, end, min_trade,
            rranges)
    buy_entry, buy_exit, sell_entry, sell_exit = strategy(
            parameter, symbol, timeframe)
    signal = get_signal(buy_entry, buy_exit, sell_entry, sell_exit, symbol,
                        timeframe)
    pnl = get_pnl(signal, symbol, timeframe, spread, start, end)
    trades = get_trades(signal, start, end)
    apr = get_apr(pnl, timeframe)
    sharpe = get_sharpe(pnl, timeframe)
    drawdown = get_drawdown(pnl)
    report.iloc[0, 0] = start.strftime('%Y.%m.%d')
    report.iloc[0, 1] = end.strftime('%Y.%m.%d')
    report.iloc[0, 2] = str(trades)
    report.iloc[0, 3] = str(np.round(apr, 3))
    report.iloc[0, 4] = str(np.round(sharpe, 3))
    report.iloc[0, 5] = str(np.round(drawdown, 3))
    if parameter is not None:
        report.iloc[0, 6] = np.round(parameter, 3)
    report = report.dropna(axis=1)
    pd.set_option('display.max_columns', 100)
    print(report)
    equity = (1.0+pnl).cumprod() - 1.0
    ax=plt.subplot()
    ax.set_xticklabels(equity.index, rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.plot(equity)
    plt.title('Backtest')
    plt.xlabel('Date')
    plt.ylabel('Equith Curve')
    plt.tight_layout()  #
    plt.savefig('backtest.png', dpi=150)
    plt.show()
    plt.close()
    empty_folder('temp')
    return pnl

def backtest_wft(strategy, symbol, timeframe, spread, start, end, rranges,
                 min_trade, in_sample_period, out_of_sample_period):
    empty_folder('temp')
    create_folder('temp')
    start = datetime.strptime(start + ' 00:00', '%Y.%m.%d %H:%M')
    end = datetime.strptime(end + ' 00:00', '%Y.%m.%d %H:%M')
    end -= timedelta(minutes=timeframe)
    report =  pd.DataFrame(
            index=[['']*1000], columns=['start_test', 'end_test', 'trades',
                  'apr', 'sharpe', 'drawdown', 'parameter'])
    end_test = start
    i = 0
    while True:
        start_train = start + timedelta(days=out_of_sample_period*i)
        end_train = (start_train + timedelta(days=in_sample_period)
            - timedelta(minutes=timeframe))
        start_test = end_train + timedelta(minutes=timeframe)
        if i == 0:
            start_all = start_test
        if (start_test + timedelta(days=out_of_sample_period)
            - timedelta(minutes=timeframe)) > end:
            end_all = end_test
            break
        end_test = (start_test + timedelta(days=out_of_sample_period)
            - timedelta(minutes=timeframe))
        parameter = optimize_parameter(
            strategy, symbol, timeframe, spread, start_train,
            end_train, min_trade, rranges)
        buy_entry, buy_exit, sell_entry, sell_exit = strategy(
                parameter, symbol, timeframe)
        signal = get_signal(buy_entry, buy_exit, sell_entry, sell_exit,
                          symbol, timeframe)
        pnl = get_pnl(signal, symbol, timeframe, spread, start_test,
                      end_test)
        trades = get_trades(signal, start_test, end_test)
        apr = get_apr(pnl, timeframe)
        sharpe = get_sharpe(pnl, timeframe)
        drawdown = get_drawdown(pnl)
        report.iloc[i, 0] = start_test.strftime('%Y.%m.%d')
        report.iloc[i, 1] = end_test.strftime('%Y.%m.%d')
        report.iloc[i, 2] = str(trades)
        report.iloc[i, 3] = str(np.round(apr, 3))
        report.iloc[i, 4] = str(np.round(sharpe, 3))
        report.iloc[i, 5] = str(np.round(drawdown, 3))
        report.iloc[i, 6] = np.round(parameter, 3)
        if i == 0:
            temp = signal[start_test:end_test]
        else:
            temp = temp.append(signal[start_test:end_test])
        i += 1
    signal = temp
    pnl = get_pnl(signal, symbol, timeframe, spread, start_all, end_all)
    trades = get_trades(signal, start_all, end_all)
    apr = get_apr(pnl, timeframe)
    sharpe = get_sharpe(pnl, timeframe)
    drawdown = get_drawdown(pnl)
    report.iloc[i, 0] = start_all.strftime('%Y.%m.%d')
    report.iloc[i, 1] = end_all.strftime('%Y.%m.%d')
    report.iloc[i, 2] = str(trades)
    report.iloc[i, 3] = str(np.round(apr, 3))
    report.iloc[i, 4] = str(np.round(sharpe, 3))
    report.iloc[i, 5] = str(np.round(drawdown, 3))
    report.iloc[i, 6] = ''
    report = report.iloc[0:i+1, :]
    report = report.dropna(axis=1)
    pd.set_option('display.max_columns', 100)
    print(report)
    equity = (1.0+pnl).cumprod() - 1.0
    ax=plt.subplot()
    ax.set_xticklabels(equity.index, rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.plot(equity)
    plt.title('Backtest')
    plt.xlabel('Date')
    plt.ylabel('Equity Curve')
    plt.tight_layout()
    plt.savefig('backtest.png', dpi=150)
    plt.show()
    plt.close()
    empty_folder('temp')
    return pnl

def create_folder(folder):
    pathname = os.path.dirname(__file__)
    if os.path.exists(pathname + '/' + folder) == False:
        os.mkdir(pathname + '/' + folder)

def empty_folder(folder):
    pathname = os.path.dirname(__file__)
    for filename in glob.glob(pathname + '/' + folder + '/*'):
        os.remove(filename)

def fill_data(data):
    filled_data = data.copy()
    filled_data[(filled_data==np.inf) | (filled_data==-np.inf)] = np.nan
    filled_data = filled_data.fillna(method='ffill')
    filled_data = filled_data.fillna(method='bfill')
    return filled_data

def forex_system():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--symbol', type=str)
    parser.add_argument('--timeframe', type=int)
    parser.add_argument('--spread', type=float)
    parser.add_argument('--start', type=str)
    parser.add_argument('--end', type=str)
    parser.add_argument('--min_trade', type=int, default=260)
    parser.add_argument('--in_sample_period', type=int, default=360)
    parser.add_argument('--out_of_sample_period', type=int, default=30)
    parser.add_argument('--start_train', type=str, default='')
    parser.add_argument('--end_train', type=str, default='')
    parser.add_argument('--mail', type=int, default=1)
    parser.add_argument('--mt4', type=int, default=0)
    args = parser.parse_args()
    mode = args.mode
    symbol = args.symbol
    timeframe = args.timeframe
    spread = args.spread
    start = args.start
    end = args.end
    min_trade = args.min_trade
    in_sample_period = args.in_sample_period
    out_of_sample_period = args.out_of_sample_period
    start_train = args.start_train
    end_train = args.end_train
    mail = args.mail
    mt4 = args.mt4
    calling_module = inspect.getmodule(inspect.stack()[1][0])
    strategy = calling_module.strategy
    get_model = calling_module.get_model
    parameter = calling_module.PARAMETER
    rranges = calling_module.RRANGES
    pnl = None
    if mode == 'backtest':
        pnl = backtest(strategy, symbol, timeframe, spread, start, end,
                       parameter=parameter)
    elif mode == 'backtest_opt':
        pnl = backtest_opt(strategy, symbol, timeframe, spread, start, end,
                           rranges=rranges, min_trade=min_trade)
    elif mode == 'backtest_wft':
        pnl = backtest_wft(strategy, symbol, timeframe, spread, start, end,
                           rranges=rranges, min_trade=min_trade,
                           in_sample_period=in_sample_period,
                           out_of_sample_period=out_of_sample_period)
    elif mode == 'backtest_ml':
        pnl = backtest_ml(strategy, symbol, timeframe, spread, start, end,
                           get_model=get_model,
                           in_sample_period=in_sample_period,
                           out_of_sample_period=out_of_sample_period)
    elif mode == 'trade':
        pathname = os.path.dirname(__file__)
        temp = inspect.getmodule(inspect.stack()[1][0]).__file__
        temp = temp.replace(pathname + '/', '') 
        ea, ext = os.path.splitext(temp)
        trade(strategy, symbol, timeframe, ea, parameter=parameter,
              start_train=start_train, end_train=end_train, mail=mail, mt4=mt4)
    elif mode == 'signal':
        pathname = os.path.dirname(__file__)
        temp = inspect.getmodule(inspect.stack()[1][0]).__file__
        temp = temp.replace(pathname + '/', '') 
        ea, ext = os.path.splitext(temp)
        signal(strategy, symbol, timeframe, ea, parameter=parameter,
               start_train=start_train, end_train=end_train)
    return pnl

def get_apr(pnl, timeframe):
    year = (len(pnl)*timeframe) / (60*24*260)
    apr = pnl.sum() / year
    return apr

def get_base_and_quote(symbol):
    if symbol == 'AUDCAD':
        base = 'AUD'
        quote = 'CAD'
    elif symbol == 'AUDCHF':
        base = 'AUD'
        quote = 'CHF'
    elif symbol == 'AUDJPY':
        base = 'AUD'
        quote = 'JPY'
    elif symbol == 'AUDNZD':
        base = 'AUD'
        quote = 'NZD'
    elif symbol == 'AUDUSD':
        base = 'AUD'
        quote = 'USD'
    elif symbol == 'CADCHF':
        base = 'CAD'
        quote = 'CHF'
    elif symbol == 'CADJPY':
        base = 'CAD'
        quote = 'JPY'
    elif symbol == 'CHFJPY':
        base = 'CHF'
        quote = 'JPY'
    elif symbol == 'EURAUD':
        base = 'EUR'
        quote = 'AUD'
    elif symbol == 'EURCAD':
        base = 'EUR'
        quote = 'CAD'
    elif symbol == 'EURCHF':
        base = 'EUR'
        quote = 'CHF'
    elif symbol == 'EURGBP':
        base = 'EUR'
        quote = 'GBP'
    elif symbol == 'EURJPY':
        base = 'EUR'
        quote = 'JPY'
    elif symbol == 'EURNZD':
        base = 'EUR'
        quote = 'NZD'
    elif symbol == 'EURUSD':
        base = 'EUR'
        quote = 'USD'
    elif symbol == 'GBPAUD':
        base = 'GBP'
        quote = 'AUD'
    elif symbol == 'GBPCAD':
        base = 'GBP'
        quote = 'CAD'
    elif symbol == 'GBPCHF':
        base = 'GBP'
        quote = 'CHF'
    elif symbol == 'GBPJPY':
        base = 'GBP'
        quote = 'JPY'
    elif symbol == 'GBPNZD':
        base = 'GBP'
        quote = 'NZD'
    elif symbol == 'GBPUSD':
        base = 'GBP'
        quote = 'USD'
    elif symbol == 'NZDCAD':
        base = 'NZD'
        quote = 'CAD'
    elif symbol == 'NZDCHF':
        base = 'NZD'
        quote = 'CHF'
    elif symbol == 'NZDJPY':
        base = 'NZD'
        quote = 'JPY'
    elif symbol == 'NZDUSD':
        base = 'NZD'
        quote = 'USD'
    elif symbol == 'USDCAD':
        base = 'USD'
        quote = 'CAD'
    elif symbol == 'USDCHF':
        base = 'USD'
        quote = 'CHF'
    elif symbol == 'USDJPY':
        base = 'USD'
        quote = 'JPY'
    else:
        print('error: get_base_and_quote')
    return base, quote

def get_current_filename():
    pathname = os.path.dirname(__file__)
    current_filename = inspect.currentframe().f_back.f_code.co_filename
    current_filename = current_filename.replace(pathname + '/', '') 
    current_filename, ext = os.path.splitext(current_filename)
    return current_filename

def get_drawdown(pnl):
    equity = (1.0+pnl).cumprod() - 1.0
    drawdown = (equity.cummax()-equity).max()
    return drawdown

def get_durations(ret, timeframe):
    cum_ret = (ret + 1.0).cumprod() - 1.0
    cum_ret = np.array(cum_ret)
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
        durations = durations / (1440 / timeframe)
        return durations
    durations = int(func(cum_ret))
    return durations

def get_historical_data(symbol, start, end):
    start = start + ' 00:00'
    end = end + ' 00:00'
    index = pd.date_range(start, end, freq='T')
    data1 = pd.DataFrame(index=index)
    filename = './historical_data/' + symbol + '.csv'
    temp = pd.read_csv(filename, index_col=0)
    temp.index = pd.to_datetime(temp.index)
    temp.index = temp.index + timedelta(hours=2)
    data1 = pd.concat([data1, temp], axis=1)
    label = ['open', 'high', 'low', 'close', 'volume']
    data1.columns = label
    ohlcv_dict = OrderedDict()
    ohlcv_dict['open'] = 'first'
    ohlcv_dict['high'] = 'max'
    ohlcv_dict['low'] = 'min'
    ohlcv_dict['close'] = 'last'
    ohlcv_dict['volume'] = 'sum'
    data1 = data1.fillna(method='ffill')
    data1 = data1[~data1.index.duplicated()]
    for i in [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60, 120, 180, 240, 360,
              480, 720, 1440]:
        if i == 1:
            data = data1.copy()
        else:
            data = data1.resample(
                    str(i)+'T', label='left', closed='left').apply(ohlcv_dict)
        data = data[data.index.dayofweek<5]
        filename =  './historical_data/' + symbol + str(i) + '.csv'
        data.to_csv(filename)

#def get_historical_data_from_yahoo(symbol):
#    start = datetime(2001, 1, 1)
#    end = datetime(2101, 1, 1)
#    data = i_close('USDJPY', 1440, 0)
#    temp = web.DataReader(symbol, 'yahoo', start, end)
#    data = pd.concat([data, temp], axis=1, join_axes=[data.index])
#    data = fill_data(data)
#    data = data[data.index.dayofweek<5]
#    data = data.drop('close', axis=1)
#    data = data.drop('Close', axis=1)
#    data.columns = ['open', 'high', 'low', 'close', 'volume']
#    filename =  './historical_data/' + symbol + '1440.csv'
#    data.to_csv(filename)

def get_model_dir():
    dirname = os.path.dirname(__file__)
    filename = inspect.currentframe().f_back.f_code.co_filename
    filename = filename.replace(dirname + '/', '')
    filename, ext = os.path.splitext(filename)
    model_dir = dirname + '/' + filename
    return model_dir

def get_pkl_file_path():
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
        else:
            arg_values += '_index'
    arg_values += '.pkl'
    pkl_file_path = dir_name + func_name + arg_values
    return pkl_file_path

def get_pnl(signal, symbol, timeframe, spread, start, end):
    op = i_open(symbol, timeframe, 0)
    if op[len(op)-1] >= 50.0:  # e.g. Cross Yen.
        adj_spread = spread / 100.0
    else:
        adj_spread = spread / 10000.0
    cost_buy = ((signal==1) & (signal.shift(1)!=1)) * adj_spread
    cost_sell = ((signal==-1) & (signal.shift(1)!=-1)) * adj_spread
    cost = cost_buy + cost_sell
    pnl = ((op-op.shift(1))*signal.shift(1)-cost) / op.shift(1)
    pnl = pnl[start:end]
    return pnl

def get_randomwalk_data(mean=0.0, std=0.01/np.sqrt(1440), skew=0.0):
    mean = mean / 6
    std = std / np.sqrt(6)
    skew = skew * np.sqrt(6)
    usdjpy = i_close('USDJPY', 1, 0)
    start = usdjpy.index[0]
    end = usdjpy.index[len(usdjpy)-1] + timedelta(seconds=59)
    index = pd.date_range(start, end, freq='10S')
    index = index[index.dayofweek<5]
    n = len(index)
    rnd = pearson3.rvs(skew=skew, loc=mean, scale=std, size=n) 
    randomwalk = rnd.cumsum() + np.log(100)
    randomwalk = np.exp(randomwalk)
    randomwalk = pd.Series(randomwalk, index=index)
    randomwalk1 = randomwalk.resample('T').ohlc()
    volume = pd.DataFrame([6]*len(randomwalk1), index=randomwalk1.index,
                       columns=['volume'])
    randomwalk1 = pd.concat([randomwalk1, volume], axis=1)
    ohlcv_dict = OrderedDict()
    ohlcv_dict['open'] = 'first'
    ohlcv_dict['high'] = 'max'
    ohlcv_dict['low'] = 'min'
    ohlcv_dict['close'] = 'last'
    ohlcv_dict['volume'] = 'sum'
    randomwalk2 = randomwalk1.resample(
            '2T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk3 = randomwalk1.resample(
            '3T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk4 = randomwalk1.resample(
            '4T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk5 = randomwalk1.resample(
            '5T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk6 = randomwalk1.resample(
            '6T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk10 = randomwalk1.resample(
            '10T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk12 = randomwalk1.resample(
            '12T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk15 = randomwalk1.resample(
            '15T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk20 = randomwalk1.resample(
            '20T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk30 = randomwalk1.resample(
            '30T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk60 = randomwalk1.resample(
            '60T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk120 = randomwalk1.resample(
            '120T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk180 = randomwalk1.resample(
            '180T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk240 = randomwalk1.resample(
            '240T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk360 = randomwalk1.resample(
            '360T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk480 = randomwalk1.resample(
            '480T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk720 = randomwalk1.resample(
            '720T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk1440 = randomwalk1.resample(
            '1440T', label='left', closed='left').apply(ohlcv_dict)
    randomwalk1 = randomwalk1[randomwalk1.index.dayofweek<5]
    randomwalk2 = randomwalk2[randomwalk2.index.dayofweek<5]
    randomwalk3 = randomwalk3[randomwalk3.index.dayofweek<5]
    randomwalk4 = randomwalk4[randomwalk4.index.dayofweek<5]
    randomwalk5 = randomwalk5[randomwalk5.index.dayofweek<5]
    randomwalk6 = randomwalk6[randomwalk6.index.dayofweek<5]
    randomwalk10 = randomwalk10[randomwalk10.index.dayofweek<5]
    randomwalk12 = randomwalk12[randomwalk12.index.dayofweek<5]
    randomwalk15 = randomwalk15[randomwalk15.index.dayofweek<5]
    randomwalk20 = randomwalk20[randomwalk20.index.dayofweek<5]
    randomwalk30 = randomwalk30[randomwalk30.index.dayofweek<5]
    randomwalk60 = randomwalk60[randomwalk60.index.dayofweek<5]
    randomwalk120 = randomwalk120[randomwalk120.index.dayofweek<5]
    randomwalk180 = randomwalk180[randomwalk180.index.dayofweek<5]
    randomwalk240 = randomwalk240[randomwalk240.index.dayofweek<5]
    randomwalk360 = randomwalk360[randomwalk360.index.dayofweek<5]
    randomwalk480 = randomwalk480[randomwalk480.index.dayofweek<5]
    randomwalk720 = randomwalk720[randomwalk720.index.dayofweek<5]
    randomwalk1440 = randomwalk1440[randomwalk1440.index.dayofweek<5]
    randomwalk1.to_csv('~/py/historical_data/RANDOM1.csv')
    randomwalk2.to_csv('~/py/historical_data/RANDOM2.csv')
    randomwalk3.to_csv('~/py/historical_data/RANDOM3.csv')
    randomwalk4.to_csv('~/py/historical_data/RANDOM4.csv')
    randomwalk5.to_csv('~/py/historical_data/RANDOM5.csv')
    randomwalk6.to_csv('~/py/historical_data/RANDOM6.csv')
    randomwalk10.to_csv('~/py/historical_data/RANDOM10.csv')
    randomwalk12.to_csv('~/py/historical_data/RANDOM12.csv')
    randomwalk15.to_csv('~/py/historical_data/RANDOM15.csv')
    randomwalk20.to_csv('~/py/historical_data/RANDOM20.csv')
    randomwalk30.to_csv('~/py/historical_data/RANDOM30.csv')
    randomwalk60.to_csv('~/py/historical_data/RANDOM60.csv')
    randomwalk120.to_csv('~/py/historical_data/RANDOM120.csv')
    randomwalk180.to_csv('~/py/historical_data/RANDOM180.csv')
    randomwalk240.to_csv('~/py/historical_data/RANDOM240.csv')
    randomwalk360.to_csv('~/py/historical_data/RANDOM360.csv')
    randomwalk480.to_csv('~/py/historical_data/RANDOM480.csv')
    randomwalk720.to_csv('~/py/historical_data/RANDOM720.csv')
    randomwalk1440.to_csv('~/py/historical_data/RANDOM1440.csv')

def get_sharpe(pnl, timeframe):
    mean = pnl.mean()
    std = pnl.std()
    multiplier = np.sqrt(60*24*260/timeframe)
    if std == 0:
        sharpe = 0.0
    else:
        sharpe = mean / std * multiplier
    return sharpe

def get_signal(buy_entry, buy_exit, sell_entry, sell_exit, symbol, timeframe):
    buy = buy_entry.copy()
    buy[buy_entry==False] = np.nan
    buy[buy_exit==True] = 0.0
    buy.iloc[0] = 0.0
    buy = buy.fillna(method='ffill')
    sell = sell_entry.copy()
    sell[sell_entry==False] = np.nan
    sell[sell_exit==True] = 0.0
    sell.iloc[0] = 0.0
    sell = sell.fillna(method='ffill')
    signal = buy - sell
    signal = signal.fillna(0.0)
    signal = signal.astype(int)
    return signal

def get_signal_of_god(symbol, timeframe, longer_timeframe):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        op = i_open(symbol, timeframe, 0)
        high = op.resample(str(longer_timeframe)+'T').max()
        high = high[high.index.dayofweek<5]
        temp = pd.concat([op, high], axis=1)
        high = temp.iloc[:, 1]
        high = fill_data(high)
        low = op.resample(str(longer_timeframe)+'T').min()
        low = low[low.index.dayofweek<5]
        temp = pd.concat([op, low], axis=1)
        low = temp.iloc[:, 1]
        low = fill_data(low)
        buy = op == low
        buy = fill_data(buy)
        buy = buy.astype(int)
        sell = op == high
        sell = fill_data(sell)
        sell = sell.astype(int)
        ret = buy - sell
        ret[ret==0] = np.nan
        ret = fill_data(ret)
        ret = ret.astype(int)
        save_pkl(ret, pkl_file_path)
    return ret

def get_trades(signal, start, end):
    trade = (signal>signal.shift(1)).astype(int)
    trade = trade[start:end]
    trades = trade.sum()
    return trades

def i_atr(symbol, timeframe, period, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
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
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_close(symbol, timeframe, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    if g_oanda is not None:
        instrument = to_instrument(symbol)
        granularity = to_granularity(timeframe)
        temp = g_oanda.get_history(
            instrument=instrument, granularity=granularity, count=COUNT)
        index = pd.Series(np.zeros(COUNT))
        ret = pd.Series(np.zeros(COUNT))
        for i in range(COUNT):
            index[i] = temp['candles'][i]['time']
            ret[i] = temp['candles'][i]['closeBid']
            index = pd.to_datetime(index)
            ret.index = index
        ret = ret.shift(shift)
    else:
        ret = restore_pkl(pkl_file_path)
        if ret is None:
            filename = ('~/py/historical_data/' + symbol + str(timeframe) +
                '.csv')
            temp = pd.read_csv(filename, index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            ret = temp.iloc[:, 3]
            ret = ret.shift(shift)
            ret = fill_data(ret)
            save_pkl(ret, pkl_file_path)
    return ret

def i_daily_high(symbol, timeframe, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        high = i_high(symbol, timeframe, shift)
        index = high.index
        temp = high.copy()
        temp[(time_hour(index)!=0) | (time_minute(index)!=0)] = np.nan
        temp = temp.fillna(method='ffill')
        ret = high.copy()
        ret[ret<temp] = np.nan
        ret = ret.fillna(method='ffill')
        while(True):
            ret[((time_hour(index)!=0) | (time_minute(index)!=0))
            & (ret < ret.shift(1))] = np.nan
            if ret.isnull().sum()==0:
                break
            ret = ret.fillna(method='ffill')
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_daily_low(symbol, timeframe, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        low = i_low(symbol, timeframe, shift)
        index = low.index
        temp = low.copy()
        temp[(time_hour(index)!=0) | (time_minute(index)!=0)] = np.nan
        temp = temp.fillna(method='ffill')
        ret = low.copy()
        ret[ret>temp] = np.nan
        ret = ret.fillna(method='ffill')
        while(True):
            ret[((time_hour(index)!=0) | (time_minute(index)!=0))
            & (ret > ret.shift(1))] = np.nan
            if ret.isnull().sum()==0:
                break
            ret = ret.fillna(method='ffill')
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_daily_open(symbol, timeframe, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        op = i_open(symbol, timeframe, shift)
        index = op.index
        ret = op.copy()
        ret[(time_hour(index)!=0) | (time_minute(index)!=0)] = np.nan
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_event(symbol, timeframe, before, after):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        close = i_close(symbol, timeframe, 0)
        index = close.index
        temp = pd.Series(index=index)
        week_of_month = time_week_of_month(index)
        day_of_week = time_day_of_week(index)
        hour = time_hour(index)
        minute = time_minute(index)
        b = int(before / timeframe)
        a = int(after / timeframe)
        # US-NFP
        temp[(week_of_month==1) & (day_of_week==5) & (hour==15)
            & (minute==30)] = 1
        temp = temp.fillna(0)
        ret = temp.copy()
        for i in range(b):
            ret += temp.shift(-i-1)
        for i in range(a):
            ret += temp.shift(i+1)    
        ret = ret.fillna(0)
        ret[ret>1] = 1
        ret = fill_data(ret)
        ret = ret.astype(int)
        save_pkl(ret, pkl_file_path)
    return ret

def i_high(symbol, timeframe, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    #
    if g_oanda is not None:
        instrument = to_instrument(symbol)
        granularity = to_granularity(timeframe)
        temp = g_oanda.get_history(
            instrument=instrument, granularity=granularity, count=COUNT)
        index = pd.Series(np.zeros(COUNT))
        ret = pd.Series(np.zeros(COUNT))
        for i in range(COUNT):
            index[i] = temp['candles'][i]['time']
            ret[i] = temp['candles'][i]['highBid']
            index = pd.to_datetime(index)
            ret.index = index
        ret = ret.shift(shift)
    else:
        ret = restore_pkl(pkl_file_path)
        if ret is None:
            filename = ('~/py/historical_data/' + symbol + str(timeframe) +
                '.csv')
            temp = pd.read_csv(filename, index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            ret = temp.iloc[:, 1]
            ret = ret.shift(shift)
            ret = fill_data(ret)
            save_pkl(ret, pkl_file_path)
    return ret

def i_highest(symbol, timeframe, period, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        def func(high):
            period = len(high)
            argmax = high.argmax()
            ret = period - 1 - argmax
            return ret
        high = i_high(symbol, timeframe, shift)
        ret = high.rolling(window=period).apply(func)
        ret = fill_data(ret)
        ret = ret.astype(int)
        save_pkl(ret, pkl_file_path)
    return ret

def i_hl_band(symbol, timeframe, period, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        high = i_high(symbol, timeframe, shift)
        low = i_low(symbol, timeframe, shift)
        ret = pd.DataFrame()
        ret['high'] = high.rolling(window=period).max()
        ret['low'] = low.rolling(window=period).min()
        ret['middle'] = (ret['high'] + ret['low']) / 2
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_kairi(symbol, timeframe, period, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    kairi = restore_pkl(pkl_file_path)
    if kairi is None:
        close = i_close(symbol, timeframe, shift)
        mean = close.rolling(window=period).mean()
        kairi = (close-mean) / mean * 100.0
        kairi = fill_data(kairi)
        save_pkl(kairi, pkl_file_path)
    return kairi

def i_ku_close(timeframe, shift, aud=0, cad=0, chf=0, eur=0, gbp=0, jpy=0,
               nzd=0, usd=0):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
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
        n = aud + cad + chf + eur + gbp + jpy + nzd + usd
        a = (audusd * aud + cadusd * cad + chfusd * chf + eurusd * eur
             + gbpusd * gbp + jpyusd * jpy + nzdusd * nzd) / n
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
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_ku_roc(timeframe, period, shift, aud=0, cad=0, chf=0, eur=0, gbp=0,
             jpy=0, nzd=0, usd=0):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        ku_close = i_ku_close(timeframe, shift, aud=aud, cad=cad, chf=chf,
                              eur=eur, gbp=gbp, jpy=jpy, nzd=nzd, usd=usd)
        ret = ku_close - ku_close.shift(period)
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_ku_trend_duration(timeframe, period, shift, aud=0, cad=0, chf=0, eur=0,
                        gbp=0, jpy=0, nzd=0, usd=0):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        ku_close = i_ku_close(timeframe, shift, aud, cad, chf, eur, gbp, jpy,
                              nzd, usd)
        ku_ma = ku_close.rolling(window=period).mean()
        n_col = len(ku_close.columns)
        above = pd.DataFrame(columns=ku_close.columns)
        below = pd.DataFrame(columns=ku_close.columns)
        for i in range(n_col):
            above.iloc[:, i] = (ku_close > ku_ma).iloc[:, i]
            above.iloc[:, i] = above.iloc[:, i] * (above.iloc[:, i].groupby(
                    (above.iloc[:, i]!=above.iloc[:, i].shift()).cumsum()
                    ).cumcount()+1)
            below.iloc[:, i] = (ku_close < ku_ma).iloc[:, i]
            below.iloc[:, i] = below.iloc[:, i] * (below.iloc[:, i].groupby(
                    (below.iloc[:, i]!=below.iloc[:, i].shift()).cumsum()
                    ).cumcount()+1)
        ret = above - below
        ret = fill_data(ret)
        ret = ret.astype(int)
        save_pkl(ret, pkl_file_path)
    return ret

def i_ku_zscore(timeframe, period, shift, aud=0, cad=0, chf=0, eur=0, gbp=0,
                jpy=0, nzd=0, usd=0):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        ku_close = i_ku_close(timeframe, shift, aud=aud, cad=cad, chf=chf,
                              eur=eur, gbp=gbp, jpy=jpy, nzd=nzd, usd=usd)
        mean = ku_close.rolling(window=period).mean()
        std = ku_close.rolling(window=period).std()
        ret = (ku_close-mean) / std
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_low(symbol, timeframe, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    if g_oanda is not None:
        instrument = to_instrument(symbol)
        granularity = to_granularity(timeframe)
        temp = g_oanda.get_history(
            instrument=instrument, granularity=granularity, count=COUNT)
        index = pd.Series(np.zeros(COUNT))
        ret = pd.Series(np.zeros(COUNT))
        for i in range(COUNT):
            index[i] = temp['candles'][i]['time']
            ret[i] = temp['candles'][i]['lowBid']
            index = pd.to_datetime(index)
            ret.index = index
        ret = ret.shift(shift)
    else:
        ret = restore_pkl(pkl_file_path)
        if ret is None:
            filename = ('~/py/historical_data/' + symbol + str(timeframe) +
                '.csv')
            temp = pd.read_csv(filename, index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            ret = temp.iloc[:, 2]
            ret = ret.shift(shift)
            ret = fill_data(ret)
            save_pkl(ret, pkl_file_path)
    return ret

def i_lowest(symbol, timeframe, period, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        def func(low):
            period = len(low)
            argmin = low.argmin()
            ret = period - 1 - argmin
            return ret
        low = i_low(symbol, timeframe, shift)
        ret = low.rolling(window=period).apply(func)
        ret = fill_data(ret)
        ret = ret.astype(int)
        save_pkl(ret, pkl_file_path)
    return ret

def i_ma(symbol, timeframe, period, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        close = i_close(symbol, timeframe, shift)
        ret = close.rolling(window=period).mean()
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_no_entry(symbol, timeframe, period, pct, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        ret = pd.DataFrame()
        close = i_close(symbol, timeframe, shift)
        hl_band = i_hl_band(symbol, timeframe, period, shift)
        from_upper = (hl_band['high']-close) / close * 100.0
        from_lower = (close-hl_band['low']) / hl_band['low'] * 100.0
        ret['buy'] = from_lower < pct
        ret['sell'] = from_upper < pct
        ret = fill_data(ret)
        ret = ret.astype(int)
        save_pkl(ret, pkl_file_path)
    return ret

def i_open(symbol, timeframe, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    if g_oanda is not None:
        instrument = to_instrument(symbol)
        granularity = to_granularity(timeframe)
        temp = g_oanda.get_history(
            instrument=instrument, granularity=granularity, count=COUNT)
        index = pd.Series(np.zeros(COUNT))
        ret = pd.Series(np.zeros(COUNT))
        for i in range(COUNT):
            index[i] = temp['candles'][i]['time']
            ret[i] = temp['candles'][i]['openBid']
            index = pd.to_datetime(index)
            ret.index = index
        ret = ret.shift(shift)
    else:
        ret = restore_pkl(pkl_file_path)
        if ret is None:
            filename = ('~/py/historical_data/' + symbol + str(timeframe) +
                '.csv')
            temp = pd.read_csv(filename, index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            ret = temp.iloc[:, 0]
            ret = ret.shift(shift)
            ret = fill_data(ret)
            save_pkl(ret, pkl_file_path)
    return ret

def i_percentrank(timeframe, period, shift, aud=0, cad=0, chf=0, eur=0, gbp=0,
                  jpy=0, nzd=0, usd=0):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        temp = i_ku_zscore(
                timeframe, period, shift, aud=aud, cad=cad, chf=chf, eur=eur,
                gbp=gbp, jpy=jpy, nzd=nzd, usd=usd)
        n = aud + cad + chf + eur + gbp + jpy + nzd + usd
        ret = temp.rank(axis=1, method='first')
        ret -= 1
        ret /= (n - 1)
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_roc(symbol, timeframe, period, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        close = i_close(symbol, timeframe, shift)
        ret = (close / close.shift(period) - 1.0) * 100.0
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_trend_duration(symbol, timeframe, period, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        high = i_high(symbol, timeframe, shift)
        low = i_low(symbol, timeframe, shift)
        ma = i_ma(symbol, timeframe, period, shift)
        above = low > ma
        above = above * (
                above.groupby((above!=above.shift()).cumsum()).cumcount()+1)
        below = high < ma
        below = below * (
                below.groupby((below!=below.shift()).cumsum()).cumcount()+1)
        ret= (above - below) / period
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_volume(symbol, timeframe, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    if g_oanda is not None:
        instrument = to_instrument(symbol)
        granularity = to_granularity(timeframe)
        temp = g_oanda.get_history(
            instrument=instrument, granularity=granularity, count=COUNT)
        index = pd.Series(np.zeros(COUNT))
        ret = pd.Series(np.zeros(COUNT))
        for i in range(COUNT):
            index[i] = temp['candles'][i]['time']
            ret[i] = temp['candles'][i]['volumeBid']
            index = pd.to_datetime(index)
            ret.index = index
        ret = ret.shift(shift)
    else:
        ret = restore_pkl(pkl_file_path)
        if ret is None:
            filename = ('~/py/historical_data/' + symbol + str(timeframe) +
                '.csv')
            temp = pd.read_csv( filename, index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            ret = temp.iloc[:, 4]
            ret = ret.shift(shift)
            ret = fill_data(ret)
            save_pkl(ret, pkl_file_path)
    return ret

def i_zscore(symbol, timeframe, period, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        close = i_close(symbol, timeframe, shift)
        mean = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        ret = (close-mean) / std
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def optimize_parameter(strategy, symbol, timeframe, spread, start, end,
                       min_trade, rranges):
    def func(parameter, strategy, symbol, timeframe, spread, start, end,
             min_trade):
        buy_entry, buy_exit, sell_entry, sell_exit = strategy(
                parameter, symbol, timeframe)
        signal = get_signal(buy_entry, buy_exit, sell_entry, sell_exit, symbol,
                            timeframe)
        pnl = get_pnl(signal, symbol, timeframe, spread, start, end)

        trades = get_trades(signal, start, end)
        sharpe = get_sharpe(pnl, timeframe)
        years = (end-start).total_seconds() / (60*60*24*365)
        if trades / years < min_trade:
            sharpe = 0.0
        return -sharpe

    parameter = optimize.brute(
            func, rranges, args=(
                    strategy, symbol, timeframe, spread, start, end,
                    min_trade), finish=None)
    return parameter

def order_close(ticket):
    g_oanda.close_trade(g_access_id, ticket)
 
def order_send(symbol, units, side):
    instrument = to_instrument(symbol)
    response = g_oanda.create_order(
            account_id=g_access_id, instrument=instrument, units=units,
            side=side, type='market')
    ticket = response['tradeOpened']['id']
    return ticket

def rename_historical_data_filename(symbol):
    new_name = './historical_data/' + symbol + '.csv'
    for old_name in glob.glob('./historical_data/' + symbol + '*'):
        os.rename(old_name, new_name)

def restore_model(filename):
    pathname = os.path.dirname(__file__) + '/' + filename
    if os.path.exists(filename) == True:
        ret = joblib.load(pathname + '/' + filename + '.pkl')
    else:
        ret = None
    return ret

def restore_pkl(pkl_file_path):
    if g_oanda is None and os.path.exists(pkl_file_path) == True:
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
    if g_oanda is None:
        create_folder('temp')
        joblib.dump(data, pkl_file_path)

def seconds():
    seconds = datetime.now().second
    return seconds

def send_mail(subject, some_text, fromaddr, toaddr, host, port, password):
    msg = MIMEText(some_text)
    msg['Subject'] = subject
    msg['From'] = fromaddr
    msg['To'] = toaddr
    with smtplib.SMTP_SSL(host, port) as s:
        s.login(fromaddr, password)
        s.send_message(msg)

def send_signal_to_mt4(filename, signal):
    f = open(filename, 'w')
    f.write(str(int(signal.iloc[len(signal)-1] + 2)))
    f.close()

def signal(strategy, symbol, timeframe, ea, parameter, start_train, end_train):
    global g_oanda
    global g_environment
    global g_access_token
    global g_access_id
    path = os.path.dirname(__file__)
    config = configparser.ConfigParser()
    config.read(path + '/settings.ini')
    host = 'smtp.mail.yahoo.co.jp'
    port = 465
    fromaddr = config['DEFAULT']['fromaddr']
    toaddr = config['DEFAULT']['toaddr']
    password = config['DEFAULT']['password']
    if g_environment is None:
        g_environment = config['DEFAULT']['environment']
    if g_access_token is None:
        g_access_token = config['DEFAULT']['access_token']
    if g_access_id is None:
        g_access_id = config['DEFAULT']['account_id']
    if g_oanda is None:
        g_oanda = oandapy.API(environment=g_environment,
                              access_token=g_access_token)
    second_before = 0
    pre_history_time = None
    instrument = to_instrument(symbol)
    granularity = to_granularity(timeframe)
    while True:
        second_now = seconds()
        if second_now != second_before:
            second_before = second_now
            history = None
            history = g_oanda.get_history(
                    instrument=instrument, granularity=granularity,
                    count=COUNT)
            if history is not None:  # Maybe useless.
                history_time = history['candles'][COUNT-1]['time']
                if history_time != pre_history_time:
                    pre_history_time = history_time
                    buy_entry, buy_exit, sell_entry, sell_exit = (
                            strategy(parameter, symbol, timeframe))
                    signal = get_signal(buy_entry, buy_exit, sell_entry,
                                        sell_exit, symbol,timeframe)
                    end_row = len(signal) - 1
                    open0 = i_open(symbol, timeframe, 0)
                    price = open0[len(open0)-1]
                    if signal.iloc[end_row] != signal.iloc[end_row-1]:
                        subject = ea
                        some_text = (
                                'Symbol: ' + symbol + '\n'
                                + 'Signal: ' + str(signal.iloc[end_row]) + '\n'
                                + 'Price: ' + str(price))
                        send_mail(subject, some_text, fromaddr, toaddr,
                                  host, port, password)
                now = datetime.now()
                print(now.strftime('%Y.%m.%d %H:%M:%S'), ea, symbol,
                      timeframe, signal.iloc[end_row])

def time_day(index):
    time_day = pd.Series(index.day, index=index)
    return time_day


def time_day_of_week(index):
    # 0-Sunday,1,2,3,4,5,6
    time_day_of_week = pd.Series(index.dayofweek, index=index) + 1
    time_day_of_week[time_day_of_week==7] = 0
    return time_day_of_week

def time_hour(index):
    time_hour = pd.Series(index.hour, index=index)
    return time_hour

def time_minute(index):
    time_minute = pd.Series(index.minute, index=index)
    return time_minute

def time_month(index):
    time_month = pd.Series(index.month, index=index)
    return time_month

def time_week_of_month(index):
    day = time_day(index)
    time_week_of_month = (np.ceil(day / 7)).astype(int)
    return time_week_of_month

def to_csv_file(symbol):
    filename_hst = './historical_data/' + symbol + '.hst'
    filename_csv = './historical_data/' + symbol + '.csv'
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
                high_price.append(bar[3])  # it's not mistake.
                low_price.append(bar[2])  # it's not mistake too.
                close_price.append(bar[4])
                volume.append(bar[5])
            else:
                buf = f.read(148)
                read += 148
    data = {'0_datetime':datetime, '1_open_price':open_price,
        '2_high_price':high_price, '3_low_price':low_price,
        '4_close_price':close_price, '5_volume':volume}
    result = pd.DataFrame.from_dict(data)
    result.columns = ['Time (UTC)', 'Open', 'High', 'Low', 'Close',
                      'Volume']
    result = result.set_index('Time (UTC)')
    result.to_csv(filename_csv)

def to_instrument(symbol):
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

def to_granularity(timeframe):
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

def to_period(minute, timeframe):
    period = int(minute / timeframe)
    return period

def trade(strategy, symbol, timeframe, ea, parameter, start_train, end_train,
          mail, mt4):
    global g_oanda
    global g_environment
    global g_access_token
    global g_access_id
    path = os.path.dirname(__file__)
    config = configparser.ConfigParser()
    config.read(path + '/settings.ini')
    if mail == 1:
        host = 'smtp.mail.yahoo.co.jp'
        port = 465
        fromaddr = config['DEFAULT']['fromaddr']
        toaddr = config['DEFAULT']['toaddr']
        password = config['DEFAULT']['password']
    if g_environment is None:
        g_environment = config['DEFAULT']['environment']
    if g_access_token is None:
        g_access_token = config['DEFAULT']['access_token']
    if g_access_id is None:
        g_access_id = config['DEFAULT']['account_id']
    if g_oanda is None:
        g_oanda = oandapy.API(environment=g_environment,
                              access_token=g_access_token)
    second_before = 0
    pre_history_time = None
    instrument = to_instrument(symbol)
    granularity = to_granularity(timeframe)
    if mt4 == 1:
        folder_ea = config['DEFAULT']['folder_ea']
        filename = folder_ea + '/' + ea + '.csv'
        f = open(filename, 'w')
        f.write(str(2))
        f.close()
    pos = 0
    ticket = 0
    while True:
        second_now = seconds()
        if second_now != second_before:
            second_before = second_now
            history = None
            history = g_oanda.get_history(
                    instrument=instrument, granularity=granularity,
                    count=COUNT)
            if history is not None:  # Maybe useless.
                history_time = history['candles'][COUNT-1]['time']
                if history_time != pre_history_time:
                    pre_history_time = history_time
                    buy_entry, buy_exit, sell_entry, sell_exit = (
                            strategy(parameter, symbol, timeframe))
                    signal = get_signal(
                            buy_entry, buy_exit, sell_entry, sell_exit, symbol,
                            timeframe)
                    end_row = len(signal) - 1
                    open0 = i_open(symbol, timeframe, 0)
                    price = open0[len(open0)-1]
                    if (pos == 1 and signal.iloc[end_row] != 1):
                        pos = 0
                        order_close(ticket)
                        ticket = 0
                        if mail == 1:
                            subject = ea
                            some_text = (symbol + ' ' +
                                         str(signal.iloc[end_row]) + ' '
                                         + str(price))
                            send_mail(subject, some_text, fromaddr, toaddr,
                                      host, port, password)
                        if mt4 == 1:
                            send_signal_to_mt4(filename, signal)
                    elif (pos == -1 and signal[end_row] != -1):
                        pos = 0
                        order_close(ticket)
                        ticket = 0
                        if mail == 1:
                            subject = ea
                            some_text = (symbol + ' ' +
                                         str(signal.iloc[end_row]) + ' '
                                         + str(price))
                            send_mail(subject, some_text, fromaddr, toaddr,
                                      host, port, password)
                        if mt4 == 1:
                            send_signal_to_mt4(filename, signal)
                    elif (pos == 0 and signal[end_row] == 1):
                        pos = 1
                        ticket = order_send(symbol, np.abs(signal[end_row]),
                                            'buy')
                        if mail == 1:
                            subject = ea
                            some_text = (symbol + ' ' +
                                         str(signal.iloc[end_row]) + ' '
                                         + str(price))
                            send_mail(subject, some_text, fromaddr, toaddr,
                                      host, port, password)
                        if mt4 == 1:
                            send_signal_to_mt4(filename, signal)
                    elif (pos == 0 and signal[end_row] == -1):
                        pos = -1
                        ticket = order_send(symbol, np.abs(signal[end_row]),
                                            'sell')
                        if mail == 1:
                            subject = ea
                            some_text = (symbol + ' ' +
                                         str(signal.iloc[end_row]) + ' '
                                         + str(price))
                            send_mail(subject, some_text, fromaddr, toaddr,
                                      host, port, password)
                        if mt4 == 1:
                            send_signal_to_mt4(filename, signal)
                now = datetime.now()
                print(now.strftime('%Y.%m.%d %H:%M:%S'), ea, symbol,
                      timeframe, signal.iloc[end_row])