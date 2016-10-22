# coding: utf-8

import forex_system as fs
# パラメータの設定
PERIOD = 10
ENTRY_THRESHOLD = 0.5
FILTER_THRESHOLD = 1.0
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

def strategy(parameter, symbol, timeframe, position):
    '''戦略を記述する。
    Args:
        parameter: パラメータ。
        symbol: 通貨ペア名。
        timeframe: 足の種類。
        position: ポジションの設定。  0: 買いのみ。  1: 売りのみ。  2: 売買両方。
    Returns:
        シグナル。
    '''
    # パラメータを格納する。
    period = int(parameter[0])
    entry_threshold = float(parameter[1])
    filter_threshold = float(parameter[2])
    # 戦略を記述する。
    zscore1 = fs.i_zscore(symbol, timeframe, period, 1)
    bandwalk1 = fs.i_bandwalk(symbol, timeframe, period, 1)
    buy_entry = ((
        (zscore1 <= -entry_threshold) &
        (bandwalk1 <= -filter_threshold)
        ) * 1)
    buy_exit = (zscore1 >= 0.0) * 1
    sell_entry = ((
        (zscore1 >= entry_threshold) &
        (bandwalk1 >= filter_threshold)
        ) * 1)
    sell_exit = (zscore1 <= 0.0) * 1
    signal = fs.calc_signal(buy_entry, buy_exit, sell_entry, sell_exit,
                            position)
    return signal