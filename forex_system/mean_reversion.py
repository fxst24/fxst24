# coding: utf-8
import forex_system as fs
import numpy as np
# パラメータの設定
PERIOD = 10
ENTRY_THRESHOLD = 1.5
FILTER_THRESHOLD = 0.5
PARAMETER = [PERIOD, ENTRY_THRESHOLD, FILTER_THRESHOLD]
# 最適化の設定
START_PERIOD = 10
END_PERIOD = 50
STEP_PERIOD = 10
START_ENTRY_THRESHOLD = 0.5
END_ENTRY_THRESHOLD = 2.5
STEP_ENTRY_THRESHOLD = 0.5
START_FILTER_THRESHOLD = 0.5
END_FILTER_THRESHOLD = 2.5
STEP_FILTER_THRESHOLD = 0.5
RRANGES = (
    slice(START_PERIOD, END_PERIOD, STEP_PERIOD),
    slice(START_ENTRY_THRESHOLD, END_ENTRY_THRESHOLD, STEP_ENTRY_THRESHOLD),
    slice(START_FILTER_THRESHOLD, END_FILTER_THRESHOLD, STEP_FILTER_THRESHOLD),
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
    # パラメータを格納する。
    period = int(parameter[0])
    entry_threshold = float(parameter[1])
    filter_threshold = float(parameter[2])
    # シグナルを計算する。
    method = 'mean'
    zresid1 = fs.i_zresid(symbol, timeframe, period, method, 1)
    bandwalk1 = fs.i_bandwalk(symbol, timeframe, period, method, 1)
    stop_hunting_zone = fs.i_stop_hunting_zone(symbol, timeframe,
        int(1440 / timeframe), 1)
    longs_entry = (((zresid1 <= -entry_threshold) &
        (bandwalk1 <= -filter_threshold) &
        (stop_hunting_zone['lower'] == False)) * 1)
    longs_exit = (zresid1 >= 0.0) * 1
    shorts_entry = (((zresid1 >= entry_threshold) &
        (bandwalk1 >= filter_threshold) &
        (stop_hunting_zone['upper'] == False)) * 1)
    shorts_exit = (zresid1 <= 0.0) * 1
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