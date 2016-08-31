# coding: utf-8
import forex_system as fs
import numpy as np
# パラメータの設定
PARAMETER = None
# 最適化の設定
RRANGES = None

def calc_signal(parameter, symbol, timeframe, position):
    '''シグナルを計算する。
    Args:
        parameter: 最適化したパラメータ。
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        position: ポジションの設定。
    Returns:
        シグナル。
    '''
    # シグナルを計算する。
    base_time, quote_time = fs.divide_symbol_time(symbol)
    close1 = fs.i_close(symbol, timeframe, 1)
    index = close1.index
    is_base_time = fs.is_trading_hours(index, base_time)
    is_quote_time = fs.is_trading_hours(index, quote_time)
    longs_entry = ((is_base_time == False) & (is_quote_time == True)) * 1
    longs_exit = (longs_entry != 1) * 1
    shorts_entry = ((is_base_time == True) & (is_quote_time == False)) * 1
    shorts_exit = (shorts_entry != 1) * 1
    longs = longs_entry.copy()
    longs[longs==0] = np.nan
    longs[longs_exit==1] = 0
    longs = longs.fillna(method='ffill')
    shorts = -shorts_entry.copy()
    shorts[shorts==0] = np.nan
    shorts[shorts_exit==1] = 0
    shorts = shorts.fillna(method='ffill')
    if position == 0:
        signal = longs
    elif position == 1:
        signal = shorts
    else:
        signal = longs + shorts
    signal = signal.fillna(0)
    signal = signal.astype(int)
    return signal