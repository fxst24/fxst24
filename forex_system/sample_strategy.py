# coding: utf-8
import forex_system as fs
import numpy as np
# パラメータの設定
PERIOD = 20
ENTRY_THRESHOLD = 2.0
PARAMETER = [PERIOD, ENTRY_THRESHOLD]
# 最適化の設定
START_PERIOD = 10
END_PERIOD = 50
STEP_PERIOD = 10
START_ENTRY_THRESHOLD = 0.5
END_ENTRY_THRESHOLD = 2.5
STEP_ENTRY_THRESHOLD = 0.5
RRANGES = (
    slice(START_PERIOD, END_PERIOD, STEP_PERIOD),
    slice(START_ENTRY_THRESHOLD, END_ENTRY_THRESHOLD, STEP_ENTRY_THRESHOLD),
)

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
    # 設定を格納する。
    period = int(parameter[0])
    entry_threshold = float(parameter[1])
    # シグナルを計算する。
    close1 = fs.i_close(symbol, timeframe, 1)
    bands1 = fs.i_bands(symbol, timeframe, period, entry_threshold, 1)
    ma1 = fs.i_ma(symbol, timeframe, period, 1)
    longs_entry = (close1 <= bands1['lower']) * 1
    longs_exit = (close1 >= ma1) * 1
    shorts_entry = (close1 >= bands1['upper']) * 1
    shorts_exit = (close1 <= ma1) * 1
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