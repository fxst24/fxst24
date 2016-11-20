# coding: utf-8
import forex_system as fs

# パラメータの設定
PERIOD = 1500
ENTRY_THRESHOLD = 0.002
PROFIT_THRESHOLD = 0.08
PARAMETER = [PERIOD, ENTRY_THRESHOLD, PROFIT_THRESHOLD]
# 最適化の設定
START_PERIOD = 250
END_PERIOD = 2500
STEP_PERIOD = 250
START_ENTRY_THRESHOLD = 0.001
END_ENTRY_THRESHOLD = 0.01
STEP_ENTRY_THRESHOLD = 0.001
START_PROFIT_THRESHOLD = 0.01
END_PROFIT_THRESHOLD = 0.1
STEP_PROFIT_THRESHOLD = 0.01
RRANGES = (
    slice(START_PERIOD, END_PERIOD, STEP_PERIOD),
    slice(START_ENTRY_THRESHOLD, END_ENTRY_THRESHOLD, STEP_ENTRY_THRESHOLD),
    slice(START_PROFIT_THRESHOLD, END_PROFIT_THRESHOLD, STEP_PROFIT_THRESHOLD),
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
    # パラメータを格納する。
    period = int(parameter[0])
    entry_threshold = float(parameter[1])
    profit_threshold = float(parameter[2])
    # 戦略を記述する。
    close1 = fs.i_close(symbol, timeframe, 1)
    hl_band1 = fs.i_hl_band(symbol, timeframe, period, 1)
    buy_entry = (close1 <= hl_band1['low'] * (1 + entry_threshold / 100)) * 1
    buy_exit = (close1 >= hl_band1['low'] * (1 + (entry_threshold +
        profit_threshold) / 100)) * 1
    sell_entry = (close1 >= hl_band1['high'] *
        (1 - entry_threshold / 100)) * 1
    sell_exit = (close1 <= hl_band1['high'] * (1 - (entry_threshold +
        profit_threshold) / 100)) * 1
    signal = fs.calc_signal(buy_entry, buy_exit, sell_entry, sell_exit)

    return signal