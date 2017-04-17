

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
import shutil
import smtplib
import struct
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from numba import float64, int64, jit
from scipy import optimize
from scipy.stats import pearson3
from sklearn.externals import joblib

COUNT = 500
EPS = 1.0e-5  # 許容誤差を設定する。

OANDA = None
ENVIRONMENT = None
ACCESS_TOKEN = None
ACCOUNT_ID = None

def backtest(
        strategy, symbol, timeframe, spread, start, end,
        build_model, sl, tp, position, min_trade, optimization,
        in_sample_period, out_of_sample_period, parameter, rranges):
    '''
    Args:
    Returns:
    '''
    delete_temp_folder()
    create_temp_folder()
    start = datetime.strptime(start + ' 00:00', '%Y.%m.%d %H:%M')
    end = datetime.strptime(end + ' 00:00', '%Y.%m.%d %H:%M')
    end -= timedelta(minutes=timeframe)
    # 最適化なし、または最適化ありのバックテストを行う。
    if optimization == 0 or optimization == 1:
        # レポートを作成する。
        report =  pd.DataFrame(
                index=[''], columns=['start', 'end', 'trades', 'apr',
                'sharpe', 'kelly', 'drawdown', 'duration', 'parameter'])
        # バックテストを行う。
        if optimization == 1:
            parameter = optimize_parameter(
                    strategy, symbol, timeframe, spread, sl, tp, start, end,
                    position, min_trade, rranges)
        buy_entry, buy_exit, sell_entry, sell_exit, units = strategy(
                parameter, symbol, timeframe)
        signal = get_signal(buy_entry, buy_exit, sell_entry, sell_exit, symbol,
                            timeframe, sl, tp)
        ret = get_return(signal, symbol, timeframe, units, spread, start, end,
                         position)
        trades = get_trades(signal, start, end, position)
        # 各パフォーマンスを計算する。
        apr = get_apr(ret, start, end)
        sharpe = get_sharpe(ret, start, end)
        kelly = get_optimal_leverage(ret)
        drawdowns = get_max_drawdown(ret)
        durations = get_max_duration(ret, timeframe)
        # レポートを作成する。
        report.iloc[0, 0] = start.strftime('%Y.%m.%d')
        report.iloc[0, 1] = end.strftime('%Y.%m.%d')
        report.iloc[0, 2] = trades
        report.iloc[0, 3] = np.round(apr, 3)
        report.iloc[0, 4] = np.round(sharpe, 3)
        report.iloc[0, 5] = np.round(kelly, 3)
        report.iloc[0, 6] = np.round(drawdowns, 3)
        report.iloc[0, 7] = durations
        if parameter is not None:
            report.iloc[0, 8] = np.round(parameter, 3)
        report = report.dropna(axis=1)
        # レポートを出力する。
        pd.set_option('display.width', 1000)
        print(report)
        # グラフを作成、出力する。
        cum_ret = (ret + 1.0).cumprod() - 1.0
        ax=plt.subplot()
        ax.set_xticklabels(cum_ret.index, rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.plot(cum_ret)
        plt.title('Backtest')
        plt.xlabel('Date')
        plt.ylabel('Cumulative returns')
        plt.text(0.05, 0.9, 'APR = ' + str(apr), transform=ax.transAxes)
        plt.text(0.05, 0.85, 'Sharpe ratio = ' + str(sharpe),
                 transform=ax.transAxes)
        plt.tight_layout()  # これを入れないとラベルがはみ出る。
        plt.savefig('backtest.png', dpi=150)
        plt.show()
        plt.close()
    # ウォークフォワードテスト、または機械学習を用いたバックテストを行う。
    elif optimization == 2 or optimization == 3:
        # レポートを作成する。
        report =  pd.DataFrame(
                index=[['']*1000], columns=['start_test', 'end_test',
                           'trades', 'apr', 'sharpe', 'kelly', 'drawdowns',
                           'durations', 'parameter',])
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
            # EAのバックテストを行う。
            if optimization == 2:
                parameter = optimize_parameter(
                    strategy, symbol, timeframe, spread, sl, tp, start_train,
                    end_train, position, min_trade, rranges)
                buy_entry, buy_exit, sell_entry, sell_exit, units = strategy(
                        parameter, symbol, timeframe)
                signal = get_signal(
                        buy_entry, buy_exit, sell_entry, sell_exit, symbol,
                        timeframe, sl, tp)
            elif optimization == 3:
                build_model(symbol, timeframe, start_train, end_train)
                buy_entry, buy_exit, sell_entry, sell_exit, units = strategy(
                        parameter, symbol, timeframe)
                signal = get_signal(
                        buy_entry, buy_exit, sell_entry, sell_exit, symbol,
                        timeframe, sl, tp)
            ret_test = get_return(signal, symbol, timeframe, units, spread,
                                  start_test, end_test, position)
            trades_test = get_trades(
                    signal, start_test, end_test, position)
            ret_test = ret_test.fillna(0.0)
            if i == 0:
                ret_test_all = ret_test
            else:
                ret_test_all = ret_test_all.append(ret_test)
            # 各パフォーマンスを計算する。
            apr = get_apr(ret_test, start_test, end_test)
            sharpe = get_sharpe(ret_test, start_test, end_test)
            kelly = get_optimal_leverage(ret_test)
            max_drawdown = get_max_drawdown(ret_test)
            max_duration = get_max_duration(ret_test, timeframe)
            # レポートを作成する。
            report.iloc[i, 0] = start_test.strftime('%Y.%m.%d')
            report.iloc[i, 1] = end_test.strftime('%Y.%m.%d')
            report.iloc[i, 2] = trades_test
            report.iloc[i, 3] = np.round(apr, 3)
            report.iloc[i, 4] = np.round(sharpe, 3)
            report.iloc[i, 5] = np.round(kelly, 3)
            report.iloc[i, 6] = np.round(max_drawdown, 3)
            report.iloc[i, 7] = max_duration
            if parameter is not None:
                report.iloc[i, 8] = np.round(parameter, 3)
            i += 1
        # 全体のレポートを最後に追加する。
        apr = get_apr(ret_test_all, start_all, end_all)
        sharpe = get_sharpe(ret_test_all, start_all, end_all)
        kelly = get_optimal_leverage(ret_test_all)
        max_drawdown = get_max_drawdown(ret_test_all)
        max_duration = get_max_duration(ret_test_all, timeframe)
        report.iloc[i, 0] = start_all.strftime('%Y.%m.%d')
        report.iloc[i, 1] = end_all.strftime('%Y.%m.%d')
        report.iloc[i, 2] = report.iloc[:, 2].sum()
        report.iloc[i, 3] = np.round(apr, 3)
        report.iloc[i, 4] = np.round(sharpe, 3)
        report.iloc[i, 5] = np.round(kelly, 3)
        report.iloc[i, 6] = np.round(max_drawdown, 3)
        report.iloc[i, 7] = max_duration
        if parameter is not None:
            report.iloc[i, 8] = ''
        report = report.iloc[0:i+1, :]
        report = report.dropna(axis=1)
        # レポートを出力する。
        pd.set_option('display.width', 1000)
        print(report)
        # グラフを作成、出力する。
        cum_ret = (ret_test_all + 1.0).cumprod() - 1.0
        ax=plt.subplot()
        ax.set_xticklabels(cum_ret.index, rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.plot(cum_ret)
        plt.title('Backtest')
        plt.xlabel('Date')
        plt.ylabel('Cumulative returns')
        plt.text(0.05, 0.9, 'APR = ' + str(apr), transform=ax.transAxes)
        plt.text(0.05, 0.85, 'Sharpe ratio = ' + str(sharpe),
                 transform=ax.transAxes)
        plt.tight_layout()  # これを入れないとラベルがはみ出る。
        plt.savefig('backtest.png', dpi=150)
        plt.show()
        plt.close()
    # 一時フォルダを削除する。
    delete_temp_folder()

def clean_historical_data_folder():
    '''Clean historical data folder.
    '''
    for filename in glob.glob('../historical_data/*'):
        os.remove(filename)

def convert_hst_to_csv(
        audcad=0, audchf=0, audjpy=0, audnzd=0, audusd=0, cadchf=0, cadjpy=0,
        chfjpy=0, euraud=0, eurcad=0, eurchf=0, eurgbp=0, eurjpy=0, eurnzd=0,
        eurusd=0, gbpaud=0, gbpcad=0, gbpchf=0, gbpjpy=0, gbpnzd=0, gbpusd=0,
        nzdcad=0, nzdchf=0, nzdjpy=0, nzdusd=0, usdcad=0, usdchf=0, usdjpy=0):
    """Convert HST file to CSV file.
    Args:
        audcad: AUDCAD.
        ...
    """
    for i in range(28):
        if i == 0:
            if audcad == 0:
                continue
            symbol = 'AUDCAD'
        elif i == 1:
            if audchf == 0:
                continue
            symbol = 'AUDCHF'
        elif i == 2:
            if audjpy == 0:
                continue
            symbol = 'AUDJPY'
        elif i == 3:
            if audnzd == 0:
                continue
            symbol = 'AUDNZD'
        elif i == 4:
            if audusd == 0:
                continue
            symbol = 'AUDUSD'
        elif i == 5:
            if cadchf == 0:
                continue
            symbol = 'CADCHF'
        elif i == 6:
            if cadjpy == 0:
                continue
            symbol = 'CADJPY'
        elif i == 7:
            if chfjpy == 0:
                continue
            symbol = 'CHFJPY'
        elif i == 8:
            if euraud == 0:
                continue
            symbol = 'EURAUD'
        elif i == 9:
            if eurcad == 0:
                continue
            symbol = 'EURCAD'
        elif i == 10:
            if eurchf == 0:
                continue
            symbol = 'EURCHF'
        elif i == 11:
            if eurgbp == 0:
                continue
            symbol = 'EURGBP'
        elif i == 12:
            if eurjpy == 0:
                continue
            symbol = 'EURJPY'
        elif i == 13:
            if eurnzd == 0:
                continue
            symbol = 'EURNZD'
        elif i == 14:
            if eurusd == 0:
                continue 
            symbol = 'EURUSD'
        elif i == 15:
            if gbpaud == 0:
                continue
            symbol = 'GBPAUD'
        elif i == 16:
            if gbpcad == 0:
                continue
            symbol = 'GBPCAD'
        elif i == 17:
            if gbpchf == 0:
                continue
            symbol = 'GBPCHF'
        elif i == 18:
            if gbpjpy == 0:
                continue
            symbol = 'GBPJPY'
        elif i == 19:
            if gbpnzd == 0:
                continue
            symbol = 'GBPNZD'
        elif i == 20:
            if gbpusd == 0:
                continue
            symbol = 'GBPUSD'
        elif i == 21:
            if nzdcad == 0:
                continue
            symbol = 'NZDCAD'
        elif i == 22:
            if nzdchf == 0:
                continue
            symbol = 'NZDCHF'
        elif i == 23:
            if nzdjpy == 0:
                continue
            symbol = 'NZDJPY'
        elif i == 24:
            if nzdusd == 0:
                continue
            symbol = 'NZDUSD'
        elif i == 25:
            if usdcad == 0:
                continue
            symbol = 'USDCAD'
        elif i == 26:
            if usdchf == 0:
                continue
            symbol = 'USDCHF'
        elif i == 27:
            if usdjpy == 0:
                continue
            symbol = 'USDJPY'
        else:
            pass
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
        result.columns = ['Time (UTC)', 'Open', 'High', 'Low', 'Close',
                          'Volume']
        result = result.set_index('Time (UTC)')
        result.to_csv(filename_csv)

def convert_minute_to_period(minute, timeframe):
    '''分を計算期間に変換する。
    Args:
        minute: 分。
        timeframe: 足の種類。
    Returns:
        計算期間。
    '''
    period = int(minute / timeframe)
    return period

def convert_symbol_to_instrument(symbol):
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

def convert_timeframe_to_granularity(timeframe):
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

def create_temp_folder():
    '''一時フォルダを作成する。
    '''
    pathname = os.path.dirname(__file__)
    if os.path.exists(pathname + '/temp') == False:
        os.mkdir(pathname + '/temp')

def delete_temp_folder():
    '''一時フォルダを削除する。
    '''
    pathname = os.path.dirname(__file__)
    if os.path.exists(pathname + '/temp') == True:
        shutil.rmtree(pathname + '/temp')

def fill_invalidate_data(data):
    '''無効なデータを埋める。
    Args:
        data: データ。
    Returns:
        filled data.
    '''
    ret = data.copy()
    ret[(ret==float("inf")) | (ret==float("-inf"))] = np.nan
    ret = ret.fillna(method='ffill')
    ret[np.isnan(ret)] = 0.0
    return ret

def get_apr(returns, start, end):
    '''年率を計算する。
    Args:
        returns: リターン。
        start: 開始日。
        end: 終了日。
    Returns:
        年率。
    '''
    rate = (returns + 1.0).prod() - 1.0
    years = (end - start).total_seconds() / 60.0 / 60.0 / 24.0 / 365.0
    ret = rate / years
    return ret

def get_base_and_quote(symbol):
    '''通貨ペアをベース通貨とクウォート通貨に分ける。
    Args:
        symbol: 通貨ペア。
    Returns:
        ベース通貨、クウォート通貨。
    '''
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
    '''拡張子なしの現在のファイル名を返す。
    Returns:
        拡張子なしの現在のファイル名。
    '''
    pathname = os.path.dirname(__file__)
    current_filename = inspect.currentframe().f_back.f_code.co_filename
    current_filename = current_filename.replace(pathname + '/', '') 
    current_filename, ext = os.path.splitext(current_filename)
    return current_filename


def get_historical_data(
        start, end,
        audcad=0, audchf=0, audjpy=0, audnzd=0, audusd=0, cadchf=0, cadjpy=0,
        chfjpy=0, euraud=0, eurcad=0, eurchf=0, eurgbp=0, eurjpy=0, eurnzd=0,
        eurusd=0, gbpaud=0, gbpcad=0, gbpchf=0, gbpjpy=0, gbpnzd=0, gbpusd=0,
        nzdcad=0, nzdchf=0, nzdjpy=0, nzdusd=0, usdcad=0, usdchf=0, usdjpy=0):
    '''ヒストリカルデータを作成する。
    Args:
        start:
        end:
        audusd: AUDUSD.
    '''
    start = start + ' 00:00'
    end = end + ' 00:00'
    index = pd.date_range(start, end, freq='T')
    data = pd.DataFrame(index=index)
    # csvファイルを読み込む。
    for i in range(28):
        if i == 0:
            if audcad == 0:
                continue
            symbol = 'AUDCAD'
        elif i == 1:
            if audchf == 0:
                continue
            symbol = 'AUDCHF'
        elif i == 2:
            if audjpy == 0:
                continue
            symbol = 'AUDJPY'
        elif i == 3:
            if audnzd == 0:
                continue
            symbol = 'AUDNZD'
        elif i == 4:
            if audusd == 0:
                continue
            symbol = 'AUDUSD'
        elif i == 5:
            if cadchf == 0:
                continue
            symbol = 'CADCHF'
        elif i == 6:
            if cadjpy == 0:
                continue
            symbol = 'CADJPY'
        elif i == 7:
            if chfjpy == 0:
                continue
            symbol = 'CHFJPY'
        elif i == 8:
            if euraud == 0:
                continue
            symbol = 'EURAUD'
        elif i == 9:
            if eurcad == 0:
                continue
            symbol = 'EURCAD'
        elif i == 10:
            if eurchf == 0:
                continue
            symbol = 'EURCHF'
        elif i == 11:
            if eurgbp == 0:
                continue
            symbol = 'EURGBP'
        elif i == 12:
            if eurjpy == 0:
                continue
            symbol = 'EURJPY'
        elif i == 13:
            if eurnzd == 0:
                continue
            symbol = 'EURNZD'
        elif i == 14:
            if eurusd == 0:
                continue 
            symbol = 'EURUSD'
        elif i == 15:
            if gbpaud == 0:
                continue
            symbol = 'GBPAUD'
        elif i == 16:
            if gbpcad == 0:
                continue
            symbol = 'GBPCAD'
        elif i == 17:
            if gbpchf == 0:
                continue
            symbol = 'GBPCHF'
        elif i == 18:
            if gbpjpy == 0:
                continue
            symbol = 'GBPJPY'
        elif i == 19:
            if gbpnzd == 0:
                continue
            symbol = 'GBPNZD'
        elif i == 20:
            if gbpusd == 0:
                continue
            symbol = 'GBPUSD'
        elif i == 21:
            if nzdcad == 0:
                continue
            symbol = 'NZDCAD'
        elif i == 22:
            if nzdchf == 0:
                continue
            symbol = 'NZDCHF'
        elif i == 23:
            if nzdjpy == 0:
                continue
            symbol = 'NZDJPY'
        elif i == 24:
            if nzdusd == 0:
                continue
            symbol = 'NZDUSD'
        elif i == 25:
            if usdcad == 0:
                continue
            symbol = 'USDCAD'
        elif i == 26:
            if usdchf == 0:
                continue
            symbol = 'USDCHF'
        elif i == 27:
            if usdjpy == 0:
                continue
            symbol = 'USDJPY'
        else:
            pass
        # 選択した通貨ペアの数を数える。
        n = (
                audcad + audchf + audjpy + audnzd + audusd + cadchf + cadjpy
              + chfjpy + euraud + eurcad + eurchf + eurgbp + eurjpy + eurnzd 
              + eurusd + gbpaud + gbpcad + gbpchf + gbpjpy + gbpnzd + gbpusd
              + nzdcad + nzdchf + nzdjpy + nzdusd + usdcad + usdchf + usdjpy)
        # 1分足の作成
        filename = '~/historical_data/' + symbol + '.csv'
        temp = pd.read_csv(filename, index_col=0)
        temp.index = pd.to_datetime(temp.index)
        # デューカスコピーのデータはUTC時間のようなので、UTC+2に変更する。
        temp.index = temp.index + timedelta(hours=2)
        data = pd.concat([data, temp], axis=1)
    # 列名を変更する。
    label = ['open', 'high', 'low', 'close', 'volume']
    data.columns = label * n
    # リサンプリングの方法を設定する。
    ohlcv_dict = OrderedDict()
    ohlcv_dict['open'] = 'first'
    ohlcv_dict['high'] = 'max'
    ohlcv_dict['low'] = 'min'
    ohlcv_dict['close'] = 'last'
    ohlcv_dict['volume'] = 'sum'
    # 各足を作成する。
    count = 0
    for i in range(28):
        if i == 0:
            if audcad == 0:
                continue
            symbol = 'AUDCAD'
            count = count + 1
        elif i == 1:
            if audchf == 0:
                continue
            symbol = 'AUDCHF'
            count = count + 1
        elif i == 2:
            if audjpy == 0:
                continue
            symbol = 'AUDJPY'
            count = count + 1
        elif i == 3:
            if audnzd == 0:
                continue
            symbol = 'AUDNZD'
            count = count + 1
        elif i == 4:
            if audusd == 0:
                continue
            symbol = 'AUDUSD'
            count = count + 1
        elif i == 5:
            if cadchf == 0:
                continue
            symbol = 'CADCHF'
            count = count + 1
        elif i == 6:
            if cadjpy == 0:
                continue
            symbol = 'CADJPY'
            count = count + 1
        elif i == 7:
            if chfjpy == 0:
                continue
            symbol = 'CHFJPY'
            count = count + 1
        elif i == 8:
            if euraud == 0:
                continue
            symbol = 'EURAUD'
            count = count + 1
        elif i == 9:
            if eurcad == 0:
                continue
            symbol = 'EURCAD'
            count = count + 1
        elif i == 10:
            if eurchf == 0:
                continue
            symbol = 'EURCHF'
            count = count + 1
        elif i == 11:
            if eurgbp == 0:
                continue
            symbol = 'EURGBP'
            count = count + 1
        elif i == 12:
            if eurjpy == 0:
                continue
            symbol = 'EURJPY'
            count = count + 1
        elif i == 13:
            if eurnzd == 0:
                continue
            symbol = 'EURNZD'
            count = count + 1
        elif i == 14:
            if eurusd == 0:
                continue 
            symbol = 'EURUSD'
            count = count + 1
        elif i == 15:
            if gbpaud == 0:
                continue
            symbol = 'GBPAUD'
            count = count + 1
        elif i == 16:
            if gbpcad == 0:
                continue
            symbol = 'GBPCAD'
            count = count + 1
        elif i == 17:
            if gbpchf == 0:
                continue
            symbol = 'GBPCHF'
            count = count + 1
        elif i == 18:
            if gbpjpy == 0:
                continue
            symbol = 'GBPJPY'
            count = count + 1
        elif i == 19:
            if gbpnzd == 0:
                continue
            symbol = 'GBPNZD'
            count = count + 1
        elif i == 20:
            if gbpusd == 0:
                continue
            symbol = 'GBPUSD'
            count = count + 1
        elif i == 21:
            if nzdcad == 0:
                continue
            symbol = 'NZDCAD'
            count = count + 1
        elif i == 22:
            if nzdchf == 0:
                continue
            symbol = 'NZDCHF'
            count = count + 1
        elif i == 23:
            if nzdjpy == 0:
                continue
            symbol = 'NZDJPY'
            count = count + 1
        elif i == 24:
            if nzdusd == 0:
                continue
            symbol = 'NZDUSD'
            count = count + 1
        elif i == 25:
            if usdcad == 0:
                continue
            symbol = 'USDCAD'
            count = count + 1
        elif i == 26:
            if usdchf == 0:
                continue
            symbol = 'USDCHF'
            count = count + 1
        elif i == 27:
            if usdjpy == 0:
                continue
            symbol = 'USDJPY'
            count = count + 1
        else:
            pass
        data1 = data.iloc[:, 0+(5*(count-1)): 5+(5*(count-1))]
        # 欠損値を補間する。
        data1 = data1.fillna(method='ffill')
        # 重複行を削除する。
        data1 = data1[~data1.index.duplicated()]
         # 2分、3分、4分、5分、6分、10分、12分、15分、20分、30分、1時間、2時間、3時間、
         # 4時間、6時間、8時間、12時間、日の各足を作成する。
        data2 = data1.resample(
            '2T', label='left', closed='left').apply(ohlcv_dict)
        data3 = data1.resample(
            '3T', label='left', closed='left').apply(ohlcv_dict)
        data4 = data1.resample(
            '4T', label='left', closed='left').apply(ohlcv_dict)
        data5 = data1.resample(
            '5T', label='left', closed='left').apply(ohlcv_dict)
        data6 = data1.resample(
            '6T', label='left', closed='left').apply(ohlcv_dict)
        data10 = data1.resample(
            '10T', label='left', closed='left').apply(ohlcv_dict)
        data12 = data1.resample(
            '12T', label='left', closed='left').apply(ohlcv_dict)
        data15 = data1.resample(
            '15T', label='left', closed='left').apply(ohlcv_dict)
        data20 = data1.resample(
            '20T', label='left', closed='left').apply(ohlcv_dict)
        data30 = data1.resample(
            '30T', label='left', closed='left').apply(ohlcv_dict)
        data60 = data1.resample(
            '60T', label='left', closed='left').apply(ohlcv_dict)
        data120 = data1.resample(
            '120T', label='left', closed='left').apply(ohlcv_dict)
        data180 = data1.resample(
            '180T', label='left', closed='left').apply(ohlcv_dict)
        data240 = data1.resample(
            '240T', label='left', closed='left').apply(ohlcv_dict)
        data360 = data1.resample(
            '360T', label='left', closed='left').apply(ohlcv_dict)
        data480 = data1.resample(
            '480T', label='left', closed='left').apply(ohlcv_dict)
        data720 = data1.resample(
            '720T', label='left', closed='left').apply(ohlcv_dict)
        data1440 = data1.resample(
            '1440T', label='left', closed='left').apply(ohlcv_dict)
        # 土日を削除する。
        data1 = data1[data1.index.dayofweek<5]
        data2 = data2[data2.index.dayofweek<5]
        data3 = data3[data3.index.dayofweek<5]
        data4 = data4[data4.index.dayofweek<5]
        data5 = data5[data5.index.dayofweek<5]
        data6 = data6[data6.index.dayofweek<5]
        data10 = data10[data10.index.dayofweek<5]
        data12 = data12[data12.index.dayofweek<5]
        data15 = data15[data15.index.dayofweek<5]
        data20 = data20[data20.index.dayofweek<5]
        data30 = data30[data30.index.dayofweek<5]
        data60 = data60[data60.index.dayofweek<5]
        data120 = data120[data120.index.dayofweek<5]
        data180 = data180[data180.index.dayofweek<5]
        data240 = data240[data240.index.dayofweek<5]
        data360 = data360[data360.index.dayofweek<5]
        data480 = data480[data480.index.dayofweek<5]
        data720 = data720[data720.index.dayofweek<5]
        data1440 = data1440[data1440.index.dayofweek<5]
        # ファイルを出力する。
        filename1 =  '~/historical_data/' + symbol + '1.csv'
        filename2 =  '~/historical_data/' + symbol + '2.csv'
        filename3 =  '~/historical_data/' + symbol + '3.csv'
        filename4 =  '~/historical_data/' + symbol + '4.csv'
        filename5 =  '~/historical_data/' + symbol + '5.csv'
        filename6 =  '~/historical_data/' + symbol + '6.csv'
        filename10 =  '~/historical_data/' + symbol + '10.csv'
        filename12 =  '~/historical_data/' + symbol + '12.csv'
        filename15 =  '~/historical_data/' + symbol + '15.csv'
        filename20 =  '~/historical_data/' + symbol + '20.csv'
        filename30 =  '~/historical_data/' + symbol + '30.csv'
        filename60 =  '~/historical_data/' + symbol + '60.csv'
        filename120 =  '~/historical_data/' + symbol + '120.csv'
        filename180 =  '~/historical_data/' + symbol + '180.csv'
        filename240 =  '~/historical_data/' + symbol + '240.csv'
        filename360 =  '~/historical_data/' + symbol + '360.csv'
        filename480 =  '~/historical_data/' + symbol + '480.csv'
        filename720 =  '~/historical_data/' + symbol + '720.csv'
        filename1440 =  '~/historical_data/' + symbol + '1440.csv'
        data1.to_csv(filename1)
        data2.to_csv(filename2)
        data3.to_csv(filename3)
        data4.to_csv(filename4)
        data5.to_csv(filename5)
        data6.to_csv(filename6)
        data10.to_csv(filename10)
        data12.to_csv(filename12)
        data15.to_csv(filename15)
        data20.to_csv(filename20)
        data30.to_csv(filename30)
        data60.to_csv(filename60)
        data120.to_csv(filename120)
        data180.to_csv(filename180)
        data240.to_csv(filename240)
        data360.to_csv(filename360)
        data480.to_csv(filename480)
        data720.to_csv(filename720)
        data1440.to_csv(filename1440)

def get_max_drawdown(ret):
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
                max_drawdown = drawdown[0];
            else:
                if drawdown[i] > max_drawdown:
                    max_drawdown = drawdown[i];
        return max_drawdown
    max_drawdown = func(cum_ret)
    return max_drawdown
 
def get_max_duration(ret, timeframe):
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
                max_duration = duration[0];
            else:
                if duration[i] > max_duration:
                    max_duration = duration[i]
        max_duration = max_duration / (1440 / timeframe)  # 営業日単位に変換
        return max_duration
    max_duration = int(func(cum_ret))
    return max_duration
 
def get_optimal_leverage(returns):
    '''ケリー基準による最適レバレッジを計算する。
    Args:
        returns: リターン。
    Returns:
        ケリー基準による最適レバレッジ。
    '''
    mean = returns.mean()
    std = returns.std()
    # 標準偏差が0であった場合、とりあえず最適レバレッジは0ということにしておく。
    if np.abs(std) < EPS:  # 標準偏差がマイナスになるはずはないが一応。
        ret = 0.0
    else:
        ret = mean / (std * std)
    return ret

def get_pkl_file_path():
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

def get_randomwalk_data(mean=0.0, std=0.01/np.sqrt(1440), skew=0.0):
    '''Get randomwalk data.
    Args:
        mean: mean.
        std: standard deviation.
        skew: skew.
    '''
    # データが1/6分なので1分に調整（歪度は合ってる？）
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
    # リサンプリングの方法を設定する。
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
    # ファイルを出力する。
    randomwalk1.to_csv('~/historical_data/RANDOM1.csv')
    randomwalk2.to_csv('~/historical_data/RANDOM2.csv')
    randomwalk3.to_csv('~/historical_data/RANDOM3.csv')
    randomwalk4.to_csv('~/historical_data/RANDOM4.csv')
    randomwalk5.to_csv('~/historical_data/RANDOM5.csv')
    randomwalk6.to_csv('~/historical_data/RANDOM6.csv')
    randomwalk10.to_csv('~/historical_data/RANDOM10.csv')
    randomwalk12.to_csv('~/historical_data/RANDOM12.csv')
    randomwalk15.to_csv('~/historical_data/RANDOM15.csv')
    randomwalk20.to_csv('~/historical_data/RANDOM20.csv')
    randomwalk30.to_csv('~/historical_data/RANDOM30.csv')
    randomwalk60.to_csv('~/historical_data/RANDOM60.csv')
    randomwalk120.to_csv('~/historical_data/RANDOM120.csv')
    randomwalk180.to_csv('~/historical_data/RANDOM180.csv')
    randomwalk240.to_csv('~/historical_data/RANDOM240.csv')
    randomwalk360.to_csv('~/historical_data/RANDOM360.csv')
    randomwalk480.to_csv('~/historical_data/RANDOM480.csv')
    randomwalk720.to_csv('~/historical_data/RANDOM720.csv')
    randomwalk1440.to_csv('~/historical_data/RANDOM1440.csv')

def get_return(signal, symbol, timeframe, units, spread, start, end, position):
    '''リターンを計算する。
    Args:
        signal: シグナル。
        symbol: 通貨ペア。
        timeframe: 期間。
        units: ユニット。
        spread: スプレッド。
        start: 開始日。
        end: 終了日。
        position: ポジションの設定。
            0: 買いのみ。
            1: 売りのみ。
            2: 売買両方。

    Returns:
        リターン。
    '''
    # スプレッドの単位を調整する。
    if (symbol == 'AUDJPY' or symbol == 'CADJPY' or symbol == 'CHFJPY' or
        symbol == 'EURJPY' or symbol == 'GBPJPY' or symbol == 'NZDJPY' or
        symbol == 'USDJPY'):
        adjusted_spread = spread / 100.0
    else:
        adjusted_spread = spread / 10000.0
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
    ret = ((op.shift(-1) - op) * adjusted_signal - cost) / op * units
    ret = fill_invalidate_data(ret)
    ret = ret[start:end]
    return ret

def get_sharpe(ret, start, end):
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

def get_signal(buy_entry, buy_exit, sell_entry, sell_exit, symbol, timeframe,
               sl, tp):
    '''シグナルを計算する。
    Args:
        buy_entry:
        buy_exit:
        sell_entry:
        sell_exit:
        symbol:
        timeframe:
        sl:
        tp:
    Returns:
        シグナル。
    '''
    # シグナルを計算する。
    buy_entry_copy = buy_entry.copy()
    sell_entry_copy = sell_entry.copy()
    buy_exit_copy = buy_exit.copy()
    sell_exit_copy = sell_exit.copy()
    buy = buy_entry_copy.copy()
    buy[buy_entry_copy==False] = np.nan
    buy[buy_exit_copy==True] = 0.0
    buy.iloc[0] = 0.0  # 不要なnanをなくすために先頭に0を入れておく。
    buy = buy.fillna(method='ffill')
    sell = sell_entry_copy.copy()
    sell[sell_entry_copy==False] = np.nan
    sell[sell_exit_copy==True] = 0.0
    sell.iloc[0] = 0.0  # 不要なnanをなくすために先頭に0を入れておく。
    sell = sell.fillna(method='ffill')
    buy_copy = buy.copy()
    sell_copy = sell.copy()
    if sl != 0 or tp != 0:
        # pipsの単位を調整する。
        if (symbol == 'AUDJPY' or symbol == 'CADJPY' or
            symbol == 'CHFJPY' or symbol == 'EURJPY' or
            symbol == 'GBPJPY' or symbol == 'NZDJPY' or
            symbol == 'USDJPY'):
            adj_sl = sl / 100.0
            adj_tp = tp / 100.0
        else:
            adj_sl = sl / 10000.0
            adj_tp = tp / 10000.0
        op = i_open(symbol, timeframe, 0)
        cl = i_close(symbol, timeframe, 0)
        buy_price = pd.Series(index=op.index)
        sell_price = pd.Series(index=op.index)
        buy_price[(buy_copy>=0.5) & (buy_copy.shift(1)<0.5)] = op.copy()
        sell_price[(sell_copy>=0.5) & (sell_copy.shift(1)<0.5)] = op.copy()
        buy_price[(buy_copy<0.5) & (buy_copy.shift(1)>=0.5)] = 0.0
        sell_price[(sell_copy<0.5) & (sell_copy.shift(1)>=0.5)] = 0.0
        buy_price.iloc[0] = 0.0
        sell_price.iloc[0] = 0.0
        buy_price = buy_price.fillna(method='ffill')
        sell_price = sell_price.fillna(method='ffill')
        buy_pl = cl * (buy_copy>=0.5) - buy_price
        sell_pl = sell_price - cl * (sell_copy>=0.5)
        if sl != 0:
            buy_exit[(buy_pl.shift(2)>-adj_sl) &
                     (buy_pl.shift(1)<=-adj_sl)] = True
            sell_exit[(sell_pl.shift(2)>-adj_sl) &
                      (sell_pl.shift(1)<=-adj_sl)] = True
        if tp != 0:
            buy_exit[(buy_pl.shift(2)<adj_tp) &
                     (buy_pl.shift(1)>=adj_tp)] = True
            sell_exit[(sell_pl.shift(2)<adj_tp) &
                      (sell_pl.shift(1)>=adj_tp)] = True
        buy = buy_entry.copy()
        buy[buy_entry==False] = np.nan
        buy[buy_exit==True] = 0.0
        buy.iloc[0] = 0.0  # 不要なnanをなくすために先頭に0を入れておく。
        buy = buy.fillna(method='ffill')
        sell = sell_entry.copy()
        sell[sell_entry==False] = np.nan
        sell[sell_exit==True] = 0.0
        sell.iloc[0] = 0.0  # 不要なnanをなくすために先頭に0を入れておく。
        sell = sell.fillna(method='ffill')
    signal = buy - sell
    signal = signal.fillna(0.0)
    signal = signal.astype(int)
    return signal

def get_trades(signal, start, end, position):
    '''トレード数を計算する。
    Args:
        signal: シグナル。
        start: 開始日。
        end: 終了日。
        position: ポジションの設定。
            0: 買いのみ。
            1: 売りのみ。
            2: 売買両方。
    Returns:
        トレード数。
    '''
    adjusted_signal = signal.copy()
    if position == 0:
        adjusted_signal[adjusted_signal==-1] = 0
    elif position == 1:
        adjusted_signal[adjusted_signal==1] = 0
    buy_trade = (adjusted_signal.shift(1) != 1) & (adjusted_signal == 1)
    sell_trade = (adjusted_signal.shift(1) != -1) & (adjusted_signal == -1)
    trade = buy_trade | sell_trade
    trade = trade.fillna(0)
    trade = trade.astype(int)
    trades = trade[start:end].sum()
    return trades

def get_units(units, index):
    '''Get number of units.
    Args:
        lots: Units.
        index: Index.
    Returns:
        Number of units.
    '''
    ret = pd.Series(units, index=index)
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
    pkl_file_path = get_pkl_file_path()  # 必ず最初に置く。
    # トレードのとき、
    if OANDA is not None:
        instrument = convert_symbol_to_instrument(symbol)
        granularity = convert_timeframe_to_granularity(timeframe)
        temp = OANDA.get_history(
            instrument=instrument, granularity=granularity,
            COUNT=COUNT)
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
            ret = fill_invalidate_data(ret)
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
    pkl_file_path = get_pkl_file_path()  # 必ず最初に置く。
    # トレードのとき、
    if OANDA is not None:
        instrument = convert_symbol_to_instrument(symbol)
        granularity = convert_timeframe_to_granularity(timeframe)
        temp = OANDA.get_history(
            instrument=instrument, granularity=granularity,
            COUNT=COUNT)
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
            ret = fill_invalidate_data(ret)
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
    pkl_file_path = get_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        high = i_high(symbol, timeframe, shift)
        low = i_low(symbol, timeframe, shift)
        ret = pd.DataFrame()
        ret['high'] = high.rolling(window=period).max()
        ret['low'] = low.rolling(window=period).min()
        ret['middle'] = (ret['high'] + ret['low']) / 2
        ret = fill_invalidate_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_ku_close(timeframe, shift, aud=0, cad=0, chf=0, eur=0, gbp=0, jpy=0,
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
    pkl_file_path = get_pkl_file_path()  # 必ず最初に置く。
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
        # Ku-Closeを作成する。
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
        ret = fill_invalidate_data(ret)
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
    pkl_file_path = get_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        ku_close = i_ku_close(timeframe, shift, aud=aud, cad=cad, chf=chf,
                              eur=eur, gbp=gbp, jpy=jpy, nzd=nzd, usd=usd)
        ret = ku_close - ku_close.shift(period)
        ret = fill_invalidate_data(ret)
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
    pkl_file_path = get_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        ku_close = i_ku_close(timeframe, shift, aud=aud, cad=cad, chf=chf,
                              eur=eur, gbp=gbp, jpy=jpy, nzd=nzd, usd=usd)
        if ma_method == 'MODE_SMA':
            mean = ku_close.rolling(window=period).mean()
            std = ku_close.rolling(window=period).std()
        elif ma_method == 'MODE_EMA':
            mean = ku_close.ewm(span=period).mean()
            std = ku_close.ewm(span=period).std()
        std = std.mean(axis=1)
        ret = (ku_close - mean).div(std, axis=0)  # メモリーエラー対策。
        ret = fill_invalidate_data(ret)
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
    pkl_file_path = get_pkl_file_path()  # 必ず最初に置く。
    # トレードのとき、
    if OANDA is not None:
        instrument = convert_symbol_to_instrument(symbol)
        granularity = convert_timeframe_to_granularity(timeframe)
        temp = OANDA.get_history(
            instrument=instrument, granularity=granularity,
            COUNT=COUNT)
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
            ret = fill_invalidate_data(ret)
            save_pkl(ret, pkl_file_path)
    return ret

def i_ma(symbol, timeframe, period, ma_method, shift):
    '''移動平均を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        period: 計算期間。
        ma_method: 移動平均のメソッド。
        shift: シフト。
    Returns:
        移動平均。
    '''
    pkl_file_path = get_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        close = i_close(symbol, timeframe, shift)
        if ma_method == 'MODE_SMA':
            ret = close.rolling(window=period).mean()
        elif ma_method == 'MODE_EMA':
            ret = close.ewm(span=period).mean()
        ret = fill_invalidate_data(ret)
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
    pkl_file_path = get_pkl_file_path()  # 必ず最初に置く。
    # トレードのとき、
    if OANDA is not None:
        instrument = convert_symbol_to_instrument(symbol)
        granularity = convert_timeframe_to_granularity(timeframe)
        temp = OANDA.get_history(
            instrument=instrument, granularity=granularity,
            COUNT=COUNT)
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
            ret = fill_invalidate_data(ret)
            save_pkl(ret, pkl_file_path)
    return ret

def i_percentrank(timeframe, period, ma_method, shift, aud=0, cad=0, chf=0,
                  eur=0, gbp=0, jpy=0, nzd=0, usd=0):
    '''通貨の順位（基準はZスコア）をパーセントで返す。
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
        通貨の順位（パーセント）。
            首位: 1.0
            最下位:0.0
    '''
    pkl_file_path = get_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        temp = i_ku_zscore(
                timeframe, period, ma_method, shift, aud=aud, cad=cad, chf=chf,
                eur=eur, gbp=gbp, jpy=jpy, nzd=nzd, usd=usd)
        n = aud + cad + chf + eur + gbp + jpy + nzd + usd
        # 同値になることはほとんどないと思うが、その場合は観測順にしている点に注意。
        ret = temp.rank(axis=1, method='first')
        ret -= 1
        ret /= (n - 1)
        ret = fill_invalidate_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_resistance(symbol, timeframe, period, shift):
    '''レジスタンスを返す。
       ただし、検索範囲は計算期間以上離れている場合に限る。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        period: 計算期間。
        shift: シフト。
    Returns:
        レジスタンス。
    '''
    pkl_file_path = get_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        @jit(float64(float64[:], int64, int64), nopython=True, cache=True)
        def func(high, i, period):
            highest = i - period
            while(True):
                if high[highest] > high[i]:
                    break
                else:
                    highest -= 1
                if highest < 0:
                    highest = i
                    break
            ret = high[highest]
            return ret

        high = i_high(symbol, timeframe, shift)
        size = len(high)
        ret = np.empty(size)
        index = high.index
        high = np.array(high)
        for i in range(1, size):
            ret[i] = func(high[0:i+1], i, period)
        ret = pd.Series(ret, index=index)
        ret = fill_invalidate_data(ret)
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
    pkl_file_path = get_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        close = i_close(symbol, timeframe, shift)
        ret = (close / close.shift(period) - 1.0) * 100.0
        ret = fill_invalidate_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_support(symbol, timeframe, period, shift):
    '''サポートを返す。
       ただし、検索範囲は計算期間以上離れている場合に限る。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        period: 計算期間。
        shift: シフト。
    Returns:
        サポート。
    '''
    pkl_file_path = get_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        @jit(float64(float64[:], int64, int64), nopython=True, cache=True)
        def func(low, i, period):
            lowest = i - period
            while(True):
                if low[lowest] < low[i]:
                    break
                else:
                    lowest -= 1
                if lowest < 0:
                    lowest = i
                    break
            ret = low[lowest]
            return ret

        low = i_low(symbol, timeframe, shift)
        size = len(low)
        ret = np.empty(size)
        index = low.index
        low = np.array(low)
        for i in range(1, size):
            ret[i] = func(low[0:i+1], i, period)
        ret = pd.Series(ret, index=index)
        ret = fill_invalidate_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def i_trend_duration(symbol, timeframe, period, ma_method, shift):
    '''トレンド期間を返す。
    Args:
        symbol: 通貨ペア。
        timeframe: 足の種類。
        period: 計算期間。
        ma_method: 移動平均のメソッド。
        shift: シフト。
    Returns:
        トレンド期間。
    '''
    pkl_file_path = get_pkl_file_path()  # 必ず最初に置く。
    ret = restore_pkl(pkl_file_path)
    if ret is None:
        high = i_high(symbol, timeframe, shift)
        low = i_low(symbol, timeframe, shift)
        ma = i_ma(symbol, timeframe, period, 'MODE_SMA', shift)
        above = low > ma
        above = above * (
                above.groupby((above!=above.shift()).cumsum()).cumcount()+1)
        below = high < ma
        below = below * (
                below.groupby((below!=below.shift()).cumsum()).cumcount()+1)
        ret = above - below
        ret = fill_invalidate_data(ret)
        ret = ret.astype(int)
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
    pkl_file_path = get_pkl_file_path()  # 必ず最初に置く。
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
        ret = fill_invalidate_data(ret)
        save_pkl(ret, pkl_file_path)
    return ret

def optimize_parameter(strategy, symbol, timeframe, spread, sl, tp,
                       start, end, position, min_trade, rranges):
    '''パラメーターを最適化する。
    Args:
        strategy: 戦略。
        symbol: 通貨ペア。
        timeframe: 足の種類。
        spread: スプレッド。
        lots: lots.
        sl: stop loss.
        tp: take profit.
        start: 開始日。
        end: 終了日。
        position: ポジションの設定。
            0: 買いのみ。
            1: 売りのみ。
            2: 売買両方。
        min_trade: 最低トレード数。
        rranges: パラメーターのスタート、ストップ、ステップ。
    Returns:
        パラメーター。
    '''
    def func(parameter, strategy, symbol, timeframe, spread, sl, tp,
             start, end, position, min_trade):
        # パフォーマンスを計算する。
        buy_entry, buy_exit, sell_entry, sell_exit, units = strategy(
                parameter, symbol, timeframe)
        signal = get_signal(buy_entry, buy_exit, sell_entry, sell_exit, symbol,
                            timeframe, sl, tp)
        ret = get_return(signal, symbol, timeframe, units, spread, start, end,
                         position)
        trades = get_trades(signal, start, end, position)
        sharpe = get_sharpe(ret, start, end)
        years = (end - start).total_seconds() / 60 / 60 / 24 / 365
        # 1年当たりのトレード数が最低トレード数に満たない場合、
        # 適応度を0にする。
        if trades / years >= min_trade:
            fitness = sharpe
        else:
            fitness = 0.0
        return -fitness

    parameter = optimize.brute(
            func, rranges, args=(
                    strategy, symbol, timeframe, spread, sl, tp,
                    start, end, position, min_trade), finish=None)
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
    instrument = convert_symbol_to_instrument(symbol)
    response = OANDA.create_order(account_id=ACCOUNT_ID, instrument=instrument,
        units=int(lots*10000), side=side, type='market')
    ticket = response['tradeOpened']['id']
    return ticket

def platform():
    '''
    Args:
    Returns:
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--symbol', type=str)
    parser.add_argument('--timeframe', type=int)
    parser.add_argument('--spread', type=float)
    parser.add_argument('--start', type=str)
    parser.add_argument('--end', type=str)
    parser.add_argument('--lots', type=float, default=0.1)
    parser.add_argument('--sl', type=float, default=0)
    parser.add_argument('--tp', type=float, default=0)
    parser.add_argument('--position', type=int, default=2)
    parser.add_argument('--min_trade', type=int, default=260)
    parser.add_argument('--optimization', type=int, default=0)
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
    lots = args.lots
    sl = args.sl
    tp = args.tp
    position = args.position
    min_trade = args.min_trade
    optimization = args.optimization
    in_sample_period = args.in_sample_period
    out_of_sample_period = args.out_of_sample_period
    start_train = args.start_train
    end_train = args.end_train
    mail = args.mail
    mt4 = args.mt4
    calling_module = inspect.getmodule(inspect.stack()[1][0])
    strategy = calling_module.strategy
    build_model = calling_module.build_model
    parameter = calling_module.PARAMETER
    rranges = calling_module.RRANGES
    if mode == 'backtest':
        backtest(
                strategy, symbol, timeframe, spread, start, end,
                build_model=build_model, sl=sl, tp=tp, position=position,
                min_trade=min_trade, optimization=optimization,
                in_sample_period=in_sample_period,
                out_of_sample_period=out_of_sample_period, parameter=parameter,
                rranges=rranges)
    elif mode == 'trade':
        pathname = os.path.dirname(__file__)
        temp = inspect.getmodule(inspect.stack()[1][0]).__file__
        temp = temp.replace(pathname + '/', '') 
        ea, ext = os.path.splitext(temp)
        trade(strategy, symbol, timeframe, ea,
              parameter=parameter, lots=lots, sl=sl, tp=tp,
              start_train=start_train, end_train=end_train, position=position,
              mail=mail, mt4=mt4)

def rename_historical_data_filename():
    '''Rename historical data filename.
    '''
    for symbol in ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADCHF',
                   'CADJPY', 'CHFJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP',
                   'EURJPY', 'EURNZD', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF',
                   'GBPJPY', 'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF', 'NZDJPY',
                   'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']:
        new_name = '../historical_data/' + symbol + '.csv'
        for old_name in glob.glob('../historical_data/' + symbol + '*'):
            os.rename(old_name, new_name)

def restore_model(filename):
    '''Restore model.
    Args:
        filename:
    Returns:
        model.
    '''
    pathname = os.path.dirname(__file__) + '/' + filename
    if os.path.exists(filename) == True:
        ret = joblib.load(pathname + '/' + filename + '.pkl')
    else:
        ret = None
    return ret

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
    '''Save model.
    Args:
        model:
        filename:
    '''
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
        create_temp_folder()
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

def send_signal_to_mt4(filename, signal):
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

def time_hour(index):
    '''時を返す。
    Args:
        index: インデックス。
    Returns:
        時。
    '''
    time_hour = pd.Series(index.hour, index=index)
    return time_hour

def time_month(index):
    '''月を返す。
    Args:
        index: インデックス。
    Returns:
        月。
    '''
    time_month = pd.Series(index.month, index=index)
    return time_month

def trade(strategy, symbol, timeframe, ea,
          parameter, lots, sl, tp, start_train, end_train, position, mail,
          mt4):
    '''
    Args:
    '''
    global OANDA
    global ENVIRONMENT
    global ACCESS_TOKEN
    global ACCOUNT_ID
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
    instrument = convert_symbol_to_instrument(symbol)
    granularity = convert_timeframe_to_granularity(timeframe)
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
                    buy_entry, buy_exit, sell_entry, sell_exit, units = (
                            strategy(parameter, symbol, timeframe))
                    signal = get_signal(
                            buy_entry, buy_exit, sell_entry, sell_exit,
                            symbol, timeframe, sl, tp)
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
                            send_signal_to_mt4(filename, signal)
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
                            send_signal_to_mt4(filename, signal)
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
                            send_signal_to_mt4(filename, signal)
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
                            send_signal_to_mt4(filename, signal)
                # シグナルを出力する。
                now = datetime.now()
                print(now.strftime('%Y.%m.%d %H:%M:%S'), ea, symbol,
                      timeframe, signal.iloc[end_row])
        # エラーを処理する。
        except Exception as e:
            print('エラーが発生しました。')
            print(e)
            time.sleep(1) # 1秒おきに表示させる。