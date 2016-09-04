# coding: utf-8
import forex_system as fs
# パラメータの設定
PARAMETER = None
# 最適化の設定
RRANGES = None

def define_trading_rules(parameter, symbol, timeframe):
    '''トレードルールを定義する。
    Args:
        parameter: 最適化したパラメータ。
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
    Returns:
        買いエントリー、買いエグジット、売りエントリー、売りエグジット、最大保有期間。
    '''
    # パラメータを格納する。
    max_hold_bars = None
    # トレードルールを定義する。
    base_time, quote_time = fs.divide_symbol_time(symbol)
    close1 = fs.i_close(symbol, timeframe, 1)
    index = close1.index
    is_base_time = fs.is_trading_hours(index, base_time)
    is_quote_time = fs.is_trading_hours(index, quote_time)
    buy_entry = ((is_base_time == False) & (is_quote_time == True)) * 1
    buy_exit = (buy_entry != 1) * 1
    sell_entry = ((is_base_time == True) & (is_quote_time == False)) * 1
    sell_exit = (sell_entry != 1) * 1
    return buy_entry, buy_exit, sell_entry, sell_exit, max_hold_bars