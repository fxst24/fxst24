import argparse
import gc
import glob
import inspect
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import struct
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from scipy import optimize
from scipy.stats import pearson3
from sklearn import linear_model
from sklearn.externals import joblib

import pandas.plotting._converter as pandacnv
pandacnv.register()

EPS = 1.0e-5

def backtest(ea, symbol, timeframe, spread, start, end, inputs):
    empty_folder('temp')
    create_folder('temp')
    start = datetime.strptime(start + ' 00:00', '%Y.%m.%d %H:%M')
    end = datetime.strptime(end + ' 00:00', '%Y.%m.%d %H:%M')
    end -= timedelta(minutes=timeframe)
    report =  pd.DataFrame(
            index=[''], columns=['start', 'end', 'trade', 'apr', 'sharpe',
                  'drawdown', 'r2', 'inputs'])
    for i in range(len(symbol)):
        buy_entry, buy_exit, sell_entry, sell_exit = ea(
                inputs, symbol[i], timeframe)
        position = calc_position(
                buy_entry, buy_exit, sell_entry, sell_exit)[start:end]
        trade = calc_trade(position, start, end) 
        pnl = calc_pnl(position, symbol[i], timeframe, spread[i])
        if i == 0:
            trade_all = trade
            pnl_all = pnl
        else:
            trade_all += trade
            pnl_all += pnl
    apr = calc_apr(pnl_all, start, end)
    sharpe = calc_sharpe(pnl_all, start, end)
    drawdown = calc_drawdown(pnl_all, start, end)
    r2 = calc_r2(pnl_all, start, end)
    report.iloc[0, 0] = start.strftime('%Y.%m.%d')
    report.iloc[0, 1] = end.strftime('%Y.%m.%d')
    report.iloc[0, 2] = str(trade_all)
    report.iloc[0, 3] = str(np.round(apr, 2))
    report.iloc[0, 4] = str(np.round(sharpe, 2))
    report.iloc[0, 5] = str(np.round(drawdown, 2))
    report.iloc[0, 6] = str(np.round(r2, 2))
    if inputs is not None:
        report.iloc[0, 7] = np.round(inputs, 2)
    report = report.dropna(axis=1)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 1000)
    print(report)
    equity = (1.0+pnl_all[start:end]).cumprod() - 1.0
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
    return pnl_all

def backtest_ml(ea, symbol, timeframe, spread, start, end, get_model,
                in_sample_period, out_of_sample_period):
    empty_folder('temp')
    create_folder('temp')
    start = datetime.strptime(start + ' 00:00', '%Y.%m.%d %H:%M')
    end = datetime.strptime(end + ' 00:00', '%Y.%m.%d %H:%M')
    end -= timedelta(minutes=timeframe)
    inputs = None
    report =  pd.DataFrame(
            index=[['']*1000], columns=['start_test', 'end_test', 'trade',
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
        for j in range(len(symbol)):
            buy_entry, buy_exit, sell_entry, sell_exit = ea(
                    inputs, symbol[j], timeframe)
            position = calc_position(
                    buy_entry, buy_exit, sell_entry,
                    sell_exit)[start_test:end_test]
            trade = calc_trade(position, start_test, end_test) 
            pnl = calc_pnl(position, symbol[j], timeframe, spread[j])
            if j == 0:
                trade_all = trade
                pnl_all = pnl
            else:
                trade_all += trade
                pnl_all += pnl
        apr = calc_apr(pnl_all, start_test, end_test)
        sharpe = calc_sharpe(pnl_all, start_test, end_test)
        drawdown = calc_drawdown(pnl_all, start_test, end_test)
        report.iloc[i, 0] = start_test.strftime('%Y.%m.%d')
        report.iloc[i, 1] = end_test.strftime('%Y.%m.%d')
        report.iloc[i, 2] = str(trade)
        report.iloc[i, 3] = str(np.round(apr, 2))
        report.iloc[i, 4] = str(np.round(sharpe, 2))
        report.iloc[i, 5] = str(np.round(drawdown, 2))
        if i == 0:
            temp = pnl_all[start_test:end_test]
            trade_all_all = trade_all
        else:
            temp = temp.append(pnl_all[start_test:end_test])
            trade_all_all += trade_all
        i += 1
    pnl_all = temp
    apr = calc_apr(pnl_all, start_all, end_all)
    sharpe = calc_sharpe(pnl_all, start_all, end_all)
    drawdown = calc_drawdown(pnl_all, start_all, end_all)
    report.iloc[i, 0] = start_all.strftime('%Y.%m.%d')
    report.iloc[i, 1] = end_all.strftime('%Y.%m.%d')
    report.iloc[i, 2] = str(trade)
    report.iloc[i, 3] = str(np.round(apr, 2))
    report.iloc[i, 4] = str(np.round(sharpe, 2))
    report.iloc[i, 5] = str(np.round(drawdown, 2))
    report = report.iloc[0:i+1, :]
    report = report.dropna(axis=1)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 1000)
    print(report)
    equity = (1.0+pnl_all).cumprod() - 1.0
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
    return pnl_all

def backtest_opt(ea, symbol, timeframe, spread, start, end, rranges,
                 min_trade, method):
    empty_folder('temp')
    create_folder('temp')
    start = datetime.strptime(start + ' 00:00', '%Y.%m.%d %H:%M')
    end = datetime.strptime(end + ' 00:00', '%Y.%m.%d %H:%M')
    end -= timedelta(minutes=timeframe)
    report =  pd.DataFrame(
            index=[''], columns=['start', 'end', 'trade', 'apr', 'sharpe',
                  'drawdown', 'r2', 'inputs'])
    inputs = optimize_inputs(ea, symbol, timeframe, spread, start, end,
                             min_trade, method, rranges)
    for i in range(len(symbol)):
        buy_entry, buy_exit, sell_entry, sell_exit = ea(
                inputs, symbol[i], timeframe)
        position = calc_position(
                buy_entry, buy_exit, sell_entry, sell_exit)[start:end]
        trade = calc_trade(position, start, end) 
        pnl = calc_pnl(position, symbol[i], timeframe, spread[i])
        if i == 0:
            trade_all = trade
            pnl_all = pnl
        else:
            trade_all += trade
            pnl_all += pnl
    apr = calc_apr(pnl_all, start, end)
    sharpe = calc_sharpe(pnl_all, start, end)
    drawdown = calc_drawdown(pnl_all, start, end)
    r2 = calc_r2(pnl_all, start, end)
    report.iloc[0, 0] = start.strftime('%Y.%m.%d')
    report.iloc[0, 1] = end.strftime('%Y.%m.%d')
    report.iloc[0, 2] = str(trade_all)
    report.iloc[0, 3] = str(np.round(apr, 2))
    report.iloc[0, 4] = str(np.round(sharpe, 2))
    report.iloc[0, 5] = str(np.round(drawdown, 2))
    report.iloc[0, 6] = str(np.round(r2, 2))
    if inputs is not None:
        report.iloc[0, 7] = np.round(inputs, 2)
    report = report.dropna(axis=1)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 1000)
    print(report)
    equity = (1.0+pnl_all[start:end]).cumprod() - 1.0
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
    return pnl_all

def backtest_wft(ea, symbol, timeframe, spread, start, end, rranges, min_trade,
                 method, in_sample_period, out_of_sample_period):
    empty_folder('temp')
    create_folder('temp')
    start = datetime.strptime(start + ' 00:00', '%Y.%m.%d %H:%M')
    end = datetime.strptime(end + ' 00:00', '%Y.%m.%d %H:%M')
    end -= timedelta(minutes=timeframe)
    report =  pd.DataFrame(
            index=[['']*1000], columns=['start_test', 'end_test', 'trade',
                  'apr', 'sharpe', 'drawdown', 'r2', 'inputs'])
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
        inputs = optimize_inputs(ea, symbol, timeframe, spread, start_train,
                                 end_train, min_trade, method, rranges)
        for j in range(len(symbol)):
            buy_entry, buy_exit, sell_entry, sell_exit = ea(
                    inputs, symbol[j], timeframe)
            position = calc_position(
                    buy_entry, buy_exit, sell_entry,
                    sell_exit)[start_test:end_test]
            trade = calc_trade(position, start_test, end_test) 
            pnl = calc_pnl(position, symbol[j], timeframe, spread[j])
            if j == 0:
                trade_all = trade
                pnl_all = pnl
            else:
                trade_all += trade
                pnl_all += pnl
        apr = calc_apr(pnl_all, start_test, end_test)
        sharpe = calc_sharpe(pnl_all, start_test, end_test)
        drawdown = calc_drawdown(pnl_all, start_test, end_test)
        r2 = calc_r2(pnl_all, start_test, end_test)
        report.iloc[i, 0] = start_test.strftime('%Y.%m.%d')
        report.iloc[i, 1] = end_test.strftime('%Y.%m.%d')
        report.iloc[i, 2] = str(trade_all)
        report.iloc[i, 3] = str(np.round(apr, 2))
        report.iloc[i, 4] = str(np.round(sharpe, 2))
        report.iloc[i, 5] = str(np.round(drawdown, 2))
        report.iloc[i, 6] = str(np.round(r2, 2))
        report.iloc[i, 7] = np.round(inputs, 2)
        if i == 0:
            temp = pnl_all[start_test:end_test]
            trade_all_all = trade_all
        else:
            temp = temp.append(pnl_all[start_test:end_test])
            trade_all_all += trade_all
        del position
        del pnl
        gc.collect()
        i += 1
    pnl_all = temp
    apr = calc_apr(pnl_all, start_all, end_all)
    sharpe = calc_sharpe(pnl_all, start_all, end_all)
    drawdown = calc_drawdown(pnl_all, start_all, end_all)
    r2 = calc_r2(pnl_all, start_all, end_all)
    report.iloc[i, 0] = start_all.strftime('%Y.%m.%d')
    report.iloc[i, 1] = end_all.strftime('%Y.%m.%d')
    report.iloc[i, 2] = str(trade_all_all)
    report.iloc[i, 3] = str(np.round(apr, 2))
    report.iloc[i, 4] = str(np.round(sharpe, 2))
    report.iloc[i, 5] = str(np.round(drawdown, 2))
    report.iloc[i, 6] = str(np.round(r2, 2))
    report.iloc[i, 7] = ''
    report = report.iloc[0:i+1, :]
    report = report.dropna(axis=1)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 1000)
    print(report)
    equity = (1.0+pnl_all[start_all:end_all]).cumprod() - 1.0
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
    return pnl_all

def calc_apr(pnl, start, end):
    cum_pnl = (1.0+pnl[start:end]).cumprod() - 1.0
    year = (end-start).total_seconds() / (60*60*24*365)
    apr = cum_pnl.iloc[len(cum_pnl)-2] / year # must use "-2"
    return apr

def calc_drawdown(pnl, start, end):
    equity = (1.0+pnl[start:end]).cumprod() - 1.0
    drawdown = (equity.cummax()-equity).max()
    return drawdown

def calc_pnl(position, symbol, timeframe, spread):
    op = i_open(symbol, timeframe, 0)
    if op[len(op)-1] >= 50.0:  # e.g. Cross Yen.
        adj_spread = spread / 100.0
    else:
        adj_spread = spread / 10000.0
    buy_position = position * (position>0.0)
    sell_position = position * (position<0.0)
    buy_cost = (
            (buy_position>EPS)
            * (buy_position-buy_position.shift(1)) * (adj_spread / op))
    sell_cost = (
            (-sell_position>EPS)
            * (-sell_position+sell_position.shift(1)) * (adj_spread / op))
    buy_pnl = (op.shift(-1)-op) / op * buy_position - buy_cost
    sell_pnl = (op.shift(-1)-op) / op * sell_position - sell_cost
    pnl = buy_pnl + sell_pnl
    pnl = pnl.fillna(0.0)
    return pnl

def calc_position(buy_entry, buy_exit, sell_entry, sell_exit,
                  holding_period=0):
    if holding_period == 0:
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
        position = (buy-sell)
        position = position.fillna(0.0)
    else:
        temp = buy_entry - sell_entry
        temp = temp.fillna(0.0)
        for i in range(holding_period):
            if i == 0:
                position = temp.copy()  # Must use "copy()"
            else:
                position += (temp.shift(i)).fillna(0.0)
        position /= holding_period
    return position

def calc_r2(pnl, start, end):
    clf = linear_model.LinearRegression()
    y = (1.0+pnl[start:end]).cumprod() - 1.0
    y = np.array(y)
    y = y.reshape(len(y), -1)
    x = np.arange(len(y))
    x = x.reshape(len(x), -1)
    clf.fit(x, y)
    r2 = clf.score(x, y)
    return r2

def calc_sharpe(pnl, start, end):
    mean = pnl[start:end].mean()
    std = pnl[start:end].std()
    timeframe = (end-start).total_seconds() / 60 / len(pnl[start:end])
    if std > EPS:
        sharpe = mean / std * np.sqrt(260*1440/timeframe)
    else:
        sharpe = 0.0
    return sharpe

def calc_trade(position, start, end):
    buy_position = position * (position>0)
    sell_position = position * (position<0)
    buy_trade = (buy_position<buy_position.shift(1)).astype(int)
    sell_trade = (sell_position>sell_position.shift(1)).astype(int)
    trade = (buy_trade+sell_trade)
    trade = trade[start:end].sum()
    return trade

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
    parser.add_argument('--symbol', action='append', type=str)
    parser.add_argument('--timeframe', type=int)
    parser.add_argument('--spread', action='append', type=float)
    parser.add_argument('--start', type=str)
    parser.add_argument('--end', type=str)
    parser.add_argument('--min_trade', type=int, default=520)
    parser.add_argument('--method', type=str, default='sharpe')
    parser.add_argument('--in_sample_period', type=int, default=360)
    parser.add_argument('--out_of_sample_period', type=int, default=30)
    args = parser.parse_args()
    mode = args.mode
    symbol = args.symbol
    timeframe = args.timeframe
    spread = args.spread
    start = args.start
    end = args.end
    min_trade = args.min_trade
    method = args.method
    in_sample_period = args.in_sample_period
    out_of_sample_period = args.out_of_sample_period
    calling_module = inspect.getmodule(inspect.stack()[1][0])
    ea = calling_module.ea
    get_model = calling_module.get_model
    inputs = calling_module.INPUTS
    rranges = calling_module.RRANGES
    pnl = None
    if mode == 'backtest':
        pnl = backtest(ea, symbol, timeframe, spread, start, end,
                       inputs=inputs)
    elif mode == 'backtest_opt':
        pnl = backtest_opt(ea, symbol, timeframe, spread, start, end,
                           rranges=rranges, min_trade=min_trade, method=method)
    elif mode == 'backtest_wft':
        pnl = backtest_wft(ea, symbol, timeframe, spread, start, end,
                           rranges=rranges, min_trade=min_trade, method=method,
                           in_sample_period=in_sample_period,
                           out_of_sample_period=out_of_sample_period)
    elif mode == 'backtest_ml':
        pnl = backtest_ml(ea, symbol, timeframe, spread, start, end,
                          get_model=get_model,
                          in_sample_period=in_sample_period,
                          out_of_sample_period=out_of_sample_period)
    return pnl

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

def i_high(symbol, timeframe, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
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

def i_no_entry_level(symbol, timeframe, period, pct, shift):
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

def i_time_zone(ts, start_hour, end_hour):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        ret = (time_hour(ts)>=start_hour) & (time_hour(ts)<end_hour)
        ret = fill_data(ret)
        ret = ret.astype(int)
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

def i_volatility(symbol, timeframe, period, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        close = i_close(symbol, timeframe, shift)
        close = close.apply(np.log)
        change = close - close.shift(1)
        mean = change.rolling(window=period).mean()
        std = change.rolling(window=period).std()
        ret = (change-mean) / std
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_volume(symbol, timeframe, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
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

def optimize_inputs(ea, symbol, timeframe, spread, start, end, min_trade,
                    method, rranges):
    def func(inputs, ea, symbol, timeframe, spread, start, end, min_trade):
        for i in range(len(symbol)):
            buy_entry, buy_exit, sell_entry, sell_exit = ea(
                    inputs, symbol[i], timeframe)
            position = calc_position(
                    buy_entry, buy_exit, sell_entry, sell_exit)[start:end]
            trade = calc_trade(position, start, end) 
            pnl = calc_pnl(position, symbol[i], timeframe, spread[i])
            if i == 0:
                trade_all = trade
                pnl_all = pnl
            else:
                trade_all += trade
                pnl_all += pnl
        if method == 'sharpe':
            ret = calc_sharpe(pnl_all, start, end)  
        else:
            ret = calc_r2(pnl_all, start, end)
        years = (end-start).total_seconds() / (60*60*24*365)
        if trade_all/years < min_trade*len(symbol):
            ret = 0.0
        del position
        del pnl
        del pnl_all
        gc.collect()
        return -ret

    inputs = optimize.brute(
            func, rranges, args=(
                    ea, symbol, timeframe, spread, start, end, min_trade),
                    finish=None)
    return inputs

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
    if os.path.exists(pkl_file_path) == True:
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
    create_folder('temp')
    joblib.dump(data, pkl_file_path)

def seconds():
    seconds = datetime.now().second
    return seconds

def time_day(ts):
    time_day = pd.Series(ts.index.day, index=ts.index)
    return time_day

def time_day_of_week(ts):
    # 0-Sunday,1,2,3,4,5,6
    time_day_of_week = pd.Series(ts.index.dayofweek, index=ts.index) + 1
    time_day_of_week[time_day_of_week==7] = 0
    return time_day_of_week

def time_hour(ts):
    time_hour = pd.Series(ts.index.hour, index=ts.index)
    return time_hour

def time_minute(ts):
    time_minute = pd.Series(ts.index.minute, index=ts.index)
    return time_minute

def time_month(ts):
    time_month = pd.Series(ts.index.month, index=ts.index)
    return time_month

def time_week_of_month(ts):
    day = time_day(ts.index)
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

def to_datetime(start, end):
    start = datetime.strptime(start + ' 00:00', '%Y.%m.%d %H:%M')
    end = datetime.strptime(end + ' 00:00', '%Y.%m.%d %H:%M')
    return start, end

def to_period(minute, timeframe):
    period = int(minute / timeframe)
    return period