# coding: utf-8
import forex_system as fs
# パラメータの設定
PERIOD = 30
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

def define_trading_rules(parameter, symbol, timeframe):
    '''トレードルールを定義する。
    Args:
        parameter: 最適化したパラメータ。
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
    Returns:
        買いエントリー、買いエグジット、売りエントリー、売りエグジット、最大保有期間。
    '''
    # 設定を格納する。
    period = int(parameter[0])
    entry_threshold = float(parameter[1])
    max_hold_bars = None
    # トレードルールを定義する。
    close1 = fs.i_close(symbol, timeframe, 1)
    bands1 = fs.i_bands(symbol, timeframe, period, entry_threshold, 1)
    ma1 = fs.i_ma(symbol, timeframe, period, 1)
    buy_entry = (close1 <= bands1['lower']) * 1
    buy_exit = (close1 >= ma1) * 1
    sell_entry = (close1 >= bands1['upper']) * 1
    sell_exit = (close1 <= ma1) * 1
    return buy_entry, buy_exit, sell_entry, sell_exit, max_hold_bars