# 標準ライブラリ
import gc
import glob
import inspect
import os
import struct
import time
from datetime import datetime, timedelta

# 外部ライブラリ
import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import spearmanr
from sklearn import linear_model

# これを入れないと警告が出てうざい。
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

eps = 1.0e-5

# バックテストを実行する。後で見直し。
def backtest(ea, symbol, timeframe, spread, start, end, mode=1, inputs=None,
             rranges=None, min_trade=260, method='sharpe',
             in_sample_period=365, out_of_sample_period=365, report=1):
    t1 = time.time()
    empty_folder('temp')
    start = datetime.strptime(start + ' 00:00', '%Y.%m.%d %H:%M')
    end = datetime.strptime(end + ' 23:59', '%Y.%m.%d %H:%M')
    table =  pd.DataFrame()
    if mode == 1:
        buy_entry, buy_exit, sell_entry, sell_exit = ea(
                inputs, symbol, timeframe)
        buy_position, sell_position = calc_position(
                buy_entry, buy_exit, sell_entry, sell_exit)
        buy_position = buy_position[start:end]
        sell_position = sell_position[start:end]
        trade = calc_trade(buy_position, sell_position, start, end) 
        pnl = calc_pnl(buy_position, sell_position, symbol, timeframe, spread)
        if report == 1:
            apr = calc_apr(pnl, start, end)
            sharpe = calc_sharpe(pnl, timeframe, start, end)
            drawdown = calc_drawdown(pnl, start, end)
            r2 = calc_r2(pnl, start, end)
            table.loc[0, 'start'] = start.strftime('%Y.%m.%d')
            table.loc[0, 'end'] = end.strftime('%Y.%m.%d')
            table.loc[0, 'trade'] = str(trade)
            table.loc[0, 'apr'] = str(np.round(apr, 2))
            table.loc[0, 'sharpe'] = str(np.round(sharpe, 2))
            table.loc[0, 'drawdown'] = str(np.round(drawdown, 2))
            table.loc[0, 'r2'] = str(np.round(r2, 2))
            if inputs is not None:
                table.loc[0, 'inputs'] = str(np.round(inputs, 2))
            table = table.dropna(axis=1)
            pd.set_option('display.max_columns', 100)
            pd.set_option('display.width', 1000)
            print(table)
            equity = pnl[start:end].cumsum()
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
    elif mode == 2:
        inputs = optimize_inputs(ea, symbol, timeframe, spread, start, end,
                                 min_trade, method, rranges)
        buy_entry, buy_exit, sell_entry, sell_exit = ea(
                inputs, symbol, timeframe)
        buy_position, sell_position = calc_position(
                buy_entry, buy_exit, sell_entry, sell_exit)
        buy_position = buy_position[start:end]
        sell_position = sell_position[start:end]
        trade = calc_trade(buy_position, sell_position, start, end) 
        pnl = calc_pnl(buy_position, sell_position, symbol, timeframe, spread)
        if report == 1:
            apr = calc_apr(pnl, start, end)
            sharpe = calc_sharpe(pnl, timeframe, start, end)
            drawdown = calc_drawdown(pnl, start, end)
            r2 = calc_r2(pnl, start, end)
            table.loc[0, 'start'] = start.strftime('%Y.%m.%d')
            table.loc[0, 'end'] = end.strftime('%Y.%m.%d')
            table.loc[0, 'trade'] = str(trade)
            table.loc[0, 'apr'] = str(np.round(apr, 2))
            table.loc[0, 'sharpe'] = str(np.round(sharpe, 2))
            table.loc[0, 'drawdown'] = str(np.round(drawdown, 2))
            table.loc[0, 'r2'] = str(np.round(r2, 2))
            if inputs is not None:
                table.loc[0, 'inputs'] = str(np.round(inputs, 2))
            table = table.dropna(axis=1)
            pd.set_option('display.max_columns', 100)
            pd.set_option('display.width', 1000)
            print(table)
            equity = pnl[start:end].cumsum()
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
    elif mode == 3:
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
            inputs = optimize_inputs(
                    ea, symbol, timeframe, spread, start_train, end_train,
                    min_trade, method, rranges)
            buy_entry, buy_exit, sell_entry, sell_exit = ea(
                    inputs, symbol, timeframe)
            buy_position, sell_position = calc_position(
                    buy_entry, buy_exit, sell_entry,
                    sell_exit)
            buy_position = buy_position[start_test:end_test]
            sell_position = sell_position[start_test:end_test]
            trade_temp = calc_trade(
                    buy_position, sell_position, start_test, end_test) 
            pnl_temp = calc_pnl(
                    buy_position, sell_position, symbol, timeframe, spread)
            if i == 0:
                pnl = pnl_temp[start_test:end_test]
                trade = trade_temp
            else:
                pnl = pnl.append(pnl_temp[start_test:end_test])
                trade += trade_temp
            if report == 1:
                apr = calc_apr(pnl_temp, start_test, end_test)
                sharpe = calc_sharpe(pnl_temp, timeframe, start_test, end_test)
                drawdown = calc_drawdown(pnl_temp, start_test, end_test)
                r2 = calc_r2(pnl_temp, start_test, end_test)
                table.loc[i, 'start'] = start_test.strftime('%Y.%m.%d')
                table.loc[i, 'end'] = end_test.strftime('%Y.%m.%d')
                table.loc[i, 'trade'] = str(trade_temp)
                table.loc[i, 'apr'] = str(np.round(apr, 2))
                table.loc[i, 'sharpe'] = str(np.round(sharpe, 2))
                table.loc[i, 'drawdown'] = str(np.round(drawdown, 2))
                table.loc[i, 'r2'] = str(np.round(r2, 2))
                table.loc[i, 'inputs'] = str(np.round(inputs, 2))
            del buy_position
            del sell_position
            del pnl_temp
            gc.collect()
            i += 1
        if report == 1:
            apr = calc_apr(pnl, start_all, end_all)
            sharpe = calc_sharpe(pnl, timeframe, start_all, end_all)
            drawdown = calc_drawdown(pnl, start_all, end_all)
            r2 = calc_r2(pnl, start_all, end_all)
            table.loc[i, 'start'] = start_all.strftime('%Y.%m.%d')
            table.loc[i, 'end'] = end_all.strftime('%Y.%m.%d')
            table.loc[i, 'trade'] = str(trade)
            table.loc[i, 'apr'] = str(np.round(apr, 2))
            table.loc[i, 'sharpe'] = str(np.round(sharpe, 2))
            table.loc[i, 'drawdown'] = str(np.round(drawdown, 2))
            table.loc[i, 'r2'] = str(np.round(r2, 2))
            table.loc[i, 'inputs'] = ''
            table = table.iloc[0:i+1, :]
            table = table.dropna(axis=1)
            pd.set_option('display.max_columns', 100)
            pd.set_option('display.width', 1000)
            print(table)
            equity = pnl[start_all:end_all].cumsum()
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
    if report == 1:
        t2 = time.time()
        m = np.floor((t2-t1)/60)
        s = t2-t1-m*60
        m = int(m)
        s = int(s)
        print('所要時間は'+str(m)+'分'+str(s)+'秒です。')
    return pnl

# 機械学習用のバックテストを実行する。
# 今のままでは使えないので後日手直し。
def backtest_ml(ea, symbol, timeframe, spread, start, end, get_model,
                in_sample_period, out_of_sample_period):
    empty_folder('temp')
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
            buy_position, sell_position = calc_position(
                    buy_entry, buy_exit, sell_entry,
                    sell_exit)[start_test:end_test]
            trade = calc_trade(
                    buy_position, sell_position, start_test, end_test) 
            pnl = calc_pnl(
                    buy_position, sell_position, symbol[j], timeframe,
                    spread[j])
            if j == 0:
                trade_all = trade
                pnl_all = pnl
            else:
                trade_all += trade
                pnl_all += pnl
        apr = calc_apr(pnl_all, start_test, end_test)
        sharpe = calc_sharpe(pnl_all, timeframe, start_test, end_test)
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
    sharpe = calc_sharpe(pnl_all, timeframe, start_all, end_all)
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

# 年利率（annual profit rate）を計算する。
# 複利にはしていない。
def calc_apr(pnl, start, end):
    cum_pnl = pnl[start:end].cumsum()
    year = (end-start).total_seconds() / (60*60*24*365)
    apr = cum_pnl.iloc[len(cum_pnl)-1] / year
    return apr

# 最大ドローダウン（％）を計算する。
def calc_drawdown(pnl, start, end):
    equity = pnl[start:end].cumsum()
    drawdown = (equity.cummax()-equity).max()
    return drawdown

# 損益を計算する。
# コストはポジションを持ったタイミングで発生したと考える。
def calc_pnl(buy_position, sell_position, symbol, timeframe, spread):
    op = i_open(symbol, timeframe, 0)
    # 通貨ペアによってスプレッドを調整する。
    if op[len(op)-1] >= 50.0:  # 例えばドル円で0.4pisなら0.004円。
        adj_spread = spread / 100.0
    else:  # 例えばユーロドルで0.5pisなら0.00005ドル。
        adj_spread = spread / 10000.0
    # 買いポジションのコストを求める。
    buy_entry_point = (buy_position==1.0) & (buy_position.shift(1)==0.0)
    buy_entry_point = buy_entry_point.astype(int)
    buy_cost = buy_entry_point * (adj_spread/op)
    # 売りポジションのコストを求める。
    sell_entry_point = (sell_position==-1.0) & (sell_position.shift(1)==0.0)
    sell_entry_point = sell_entry_point.astype(int)
    sell_cost = sell_entry_point * (adj_spread/op)
    # 損益を計算する。
    buy_pnl = (op.shift(-1)-op) / op * buy_position - buy_cost
    sell_pnl = (op.shift(-1)-op) / op * sell_position - sell_cost
    pnl = buy_pnl + sell_pnl
    pnl = pnl.fillna(0.0)
    return pnl

# ポジションを計算する。ドテンに対応しているか後日確認。
def calc_position(buy_entry, buy_exit, sell_entry, sell_exit):
    # 買いポジションを求める。
    buy = buy_entry.copy()
    buy[buy_entry==False] = np.nan
    buy[buy_exit==True] = 0.0
    buy.iloc[0] = 0.0
    buy_position = buy.fillna(method='ffill')
    # 売りポジションを求める。
    sell = sell_entry.copy()
    sell[sell_entry==False] = np.nan
    sell[sell_exit==True] = 0.0
    sell.iloc[0] = 0.0
    sell_position = -sell.fillna(method='ffill')
    return buy_position, sell_position

# 資産曲線の形状をR^2で計算する。
def calc_r2(pnl, start, end):
    clf = linear_model.LinearRegression()
    y = (pnl[start:end]).cumsum()
    y = np.array(y)
    y = y.reshape(len(y), -1)
    x = np.arange(len(y))
    x = x.reshape(len(x), -1)
    clf.fit(x, y)
    r2 = clf.score(x, y)
    return r2

# シャープレシオを計算する。
def calc_sharpe(pnl, timeframe, start, end):
    mean = pnl[start:end].mean()
    std = pnl[start:end].std()
    if std > eps:
        sharpe = mean / std * np.sqrt(260*1440/timeframe)
    else:
        sharpe = 0.0
    return sharpe

# トレード数を計算する。
def calc_trade(buy_position, sell_position, start, end):
    buy_entry_point = (buy_position==1.0) & (buy_position.shift(1)==0.0)
    buy_entry_point = buy_entry_point.astype(int)
    sell_entry_point = (sell_position==-1.0) & (sell_position.shift(1)==0.0)
    sell_entry_point = sell_entry_point.astype(int)
    entry_point = buy_entry_point + sell_entry_point
    trade = entry_point[start:end].sum()
    return trade

# フォルダーを空にする。
def empty_folder(folder):
    pathname = os.path.dirname(__file__)
    # 指定したフォルダーがなければ作成する。
    if os.path.exists(pathname + '/' + folder) == False:
        os.mkdir(pathname + '/' + folder)
    # 指定したフォルダーを空にする。
    for filename in glob.glob(pathname + '/' + folder + '/*'):
        os.remove(filename)

# データを補完する。
# NAを埋める関数を利用してデータで補完する。
def fill_data(data):
    filled_data = data.copy()
    # 無限大、無限小も保管したいのでNAに変換する。
    filled_data[(filled_data==np.inf) | (filled_data==-np.inf)] = np.nan
    # 前のデータでNAを補完する。
    filled_data = filled_data.fillna(method='ffill')
    # 先頭からNAが存在する場合があるので、そのときは後のデータで補完する。
    filled_data = filled_data.fillna(method='bfill')
    return filled_data

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

def get_model_dir():
    dirname = os.path.dirname(__file__)
    filename = inspect.currentframe().f_back.f_code.co_filename
    filename = filename.replace(dirname + '/', '')
    filename, ext = os.path.splitext(filename)
    model_dir = dirname + '/' + filename
    return model_dir

def get_pkl_file_path():
    pathname = os.path.dirname(__file__)
    # tempフォルダーがなければ作成する。
    if os.path.exists(pathname + '/temp') == False:
        os.mkdir(pathname + '/temp')
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

def i_four_hourly_open(symbol, timeframe, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        op = i_open(symbol, timeframe, shift)
        index = op.index
        ret = op.copy()
        ret[time_hour(index)%4!=0] = np.nan
        ret[time_minute(index)!=0] = np.nan
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
        ret = high.rolling(window=period).apply(func, raw=True)
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

def i_hourly_open(symbol, timeframe, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        op = i_open(symbol, timeframe, shift)
        index = op.index
        ret = op.copy()
        ret[time_minute(index)!=0] = np.nan
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

def i_ku_chart(timeframe, open_hour, shift, aud=0, cad=0, chf=0, eur=0, gbp=0,
               jpy=0, nzd=0, usd=0):
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
        temp = ret.copy()
        index = temp.index
        temp[(time_hour(index)!=open_hour)|(time_minute(index)!=0)] = np.nan
        temp = fill_data(temp)
        ret -= temp
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

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

def i_level(symbol, timeframe, period, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        ret = pd.DataFrame()
        close = i_close(symbol, timeframe, shift)
        hl_band = i_hl_band(symbol, timeframe, period, shift)
        ret['high'] = (hl_band['high']-close) / close * 100.0
        ret['low'] = (close-hl_band['low']) / hl_band['low'] * 100.0
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
        ret = low.rolling(window=period).apply(func, raw=True)
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

def i_random_walk(symbol, timeframe, fast_period, slow_period, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        close = i_close(symbol, timeframe, shift)
        change = (close-close.shift(1)) / close.shift(1)
        # mean = 0.0
        std = change.rolling(window=slow_period).std()
        ret = ((close-close.shift(fast_period))/close.shift(fast_period)) / (std*np.sqrt(fast_period))
#        change = np.log(close) - np.log(close.shift(1))
#        # mean = 0.0
#        std = change.rolling(window=slow_period).std()
#        ret = (np.log(close)-np.log(close.shift(fast_period))) / (std*np.sqrt(fast_period))
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

# RCIを返す。
# 短い足だと時間かかりまくり。
def i_rci(symbol, timeframe, period, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        def func(close):
            n = len(close)
            no = np.arange(n)
            ret = spearmanr(no, close)[0]
            return ret
        close = i_close(symbol, timeframe, shift)
        ret = close.rolling(window=period).apply(func, raw=True)
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

def i_standardized_kairi(symbol, timeframe, fast_period, slow_period, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        close = i_close(symbol, timeframe, shift)
        ma = close.rolling(window=fast_period).mean()
        kairi = (close-ma) / ma * 100.0
        mean = kairi.rolling(window=slow_period).mean()
        std = kairi.rolling(window=slow_period).std()
        ret = (kairi-mean) / std
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_std_dev(symbol, timeframe, period, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        close = i_close(symbol, timeframe, shift)
        ret = close.rolling(window=period).std()
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_trading_hours(ts, exchange):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    #trading hours
    # tse: 02:00-08:00 (03:00-09:00)
    # les: 10:00-18:30
    # nyse: 16:30-23:00
    #summer time
    # usa: 03-10 (not exact)
    if ret is None:
        if exchange == 'tse':
            ret = pd.Series(np.zeros(len(ts)), index=ts.index)
            ret[(time_month(ts)<3) | (time_month(ts)>10)] = (
                    (time_hour(ts)>=2) & (time_hour(ts)<8))
            ret[(time_month(ts)>=3) & (time_month(ts)<=10)] = (
                    (time_hour(ts)>=3) & (time_hour(ts)<9))
        elif exchange == 'lse':
            ret = (((time_hour(ts)>=10) & (time_hour(ts)<18)) |
                    ((time_hour(ts)==18) & (time_minute(ts)<30)))
        elif exchange == 'nyse':
            ret = (((time_hour(ts)==16) & (time_minute(ts)>=30)) |
                    ((time_hour(ts)>=17) & (time_hour(ts)<23)))
        else:
            ret = time_hour(ts) < 0
        ret = fill_data(ret)
        ret = ret.astype(int)
        save_pkl(ret, pkl_file_path)
    return ret

def i_trend_duration(symbol, timeframe, period, mode, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        if mode == 'close':
            close = i_close(symbol, timeframe, shift)
            ma = i_ma(symbol, timeframe, period, shift)
            above = close > ma
            above = above * (above.groupby(
                    (above!=above.shift()).cumsum()).cumcount()+1)
            below = close < ma
            below = below * (below.groupby(
                    (below!=below.shift()).cumsum()).cumcount()+1)
        elif mode == 'highlow':
            high = i_high(symbol, timeframe, shift)
            low = i_low(symbol, timeframe, shift)
            ma = i_ma(symbol, timeframe, period, shift)
            above = low > ma
            above = above * (above.groupby(
                    (above!=above.shift()).cumsum()).cumcount()+1)
            below = high < ma
            below = below * (below.groupby(
                    (below!=below.shift()).cumsum()).cumcount()+1)
        ret = (above-below) / period
        ret = fill_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_volatility(symbol, timeframe, period, shift):
    pkl_file_path = get_pkl_file_path()  # Must put this first.
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        close = i_close(symbol, timeframe, shift)
        change = (close-close.shift(1)) / (close-close.shift(1))
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

def i_z_score(symbol, timeframe, period, shift):
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
        buy_entry, buy_exit, sell_entry, sell_exit = ea(inputs, symbol,
                                                        timeframe)
        buy_position, sell_position = calc_position(
                buy_entry, buy_exit, sell_entry, sell_exit)
        buy_position = buy_position[start:end]
        sell_position = sell_position[start:end]
        trade = calc_trade(buy_position, sell_position, start, end) 
        pnl = calc_pnl(buy_position, sell_position, symbol, timeframe, spread)
        if method == 'sharpe':
            ret = calc_sharpe(pnl, timeframe, start, end)
        elif method == 'drawdown':
            ret = -calc_drawdown(pnl, start, end)
        elif method == 'r2':
            ret = calc_r2(pnl, start, end)
        years = (end-start).total_seconds() / (60*60*24*365)
        if trade/years < min_trade:
            ret = 0.0
        del buy_position
        del sell_position
        del pnl
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
    joblib.dump(data, pkl_file_path)

def seconds():
    seconds = datetime.now().second
    return seconds

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

def to_datetime(start, end):
    start = datetime.strptime(start + ' 00:00', '%Y.%m.%d %H:%M')
    end = datetime.strptime(end + ' 23:59', '%Y.%m.%d %H:%M')
    return start, end

def to_period(minute, timeframe):
    period = int(minute / timeframe)
    return period