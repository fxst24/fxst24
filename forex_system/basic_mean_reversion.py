# coding: utf-8

import forex_system as fs
# パラメータの設定
MINUTE = 60
ENTRY_THRESHOLD = 1.5
FILTER_THRESHOLD = 1.5
PARAMETER = [MINUTE, ENTRY_THRESHOLD, FILTER_THRESHOLD]
# 最適化の設定
START_MINUTE = 60
END_MINUTE = 300
STEP_MINUTE = 60
START_ENTRY_THRESHOLD = 0.5
END_ENTRY_THRESHOLD = 2.5
STEP_ENTRY_THRESHOLD = 0.5
START_FILTER_THRESHOLD = 0.5
END_FILTER_THRESHOLD = 2.5
STEP_FILTER_THRESHOLD = 0.5
RRANGES = (
    slice(START_MINUTE, END_MINUTE, STEP_MINUTE),
    slice(START_ENTRY_THRESHOLD, END_ENTRY_THRESHOLD, STEP_ENTRY_THRESHOLD),
    slice(START_FILTER_THRESHOLD, END_FILTER_THRESHOLD, STEP_FILTER_THRESHOLD),
)

def strategy(parameter, symbol, timeframe):
    '''戦略を記述する。
    Args:
        parameter: パラメーター。
        symbol: 通貨ペア。
        timeframe: 期間。
    Returns:
        シグナル。
    '''
    # パラメーターを格納する。
    minute = int(parameter[0])
    entry_threshold = float(parameter[1])
    filter_threshold = float(parameter[2])
    # 戦略を記述する。
    period = fs.convert_minute2period(minute, timeframe)
    slope = 0.60527080848
    intercept = -0.971875975206
    divisor = slope * period + intercept
    zscore1 = fs.i_zscore(symbol, timeframe, period, 1)
    bandwalk1 = fs.i_bandwalk(symbol, timeframe, period, 1) / divisor
    buy_entry = (
        (zscore1 <= -entry_threshold) &
        (bandwalk1 <= -filter_threshold)
        )
    buy_exit = zscore1 >= 0.0
    sell_entry = (
        (zscore1 >= entry_threshold) &
        (bandwalk1 >= filter_threshold)
        )
    sell_exit = zscore1 <= 0.0
    signal = fs.calc_signal(buy_entry, buy_exit, sell_entry, sell_exit)

    return signal