# coding: utf-8
import forex_system as fs
# パラメータの設定
PARAMETER = None
# 最適化の設定
RRANGES = None

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
    # 戦略を記述する。
    base_time, quote_time = fs.divide_symbol_time(symbol)
    close1 = fs.i_close(symbol, timeframe, 1)
    index = close1.index
    is_base_time = fs.is_trading_hours(index, base_time)
    is_quote_time = fs.is_trading_hours(index, quote_time)
    buy_entry = ((is_base_time == False) & (is_quote_time == True)) * 1
    buy_exit = (buy_entry != 1) * 1
    sell_entry = ((is_base_time == True) & (is_quote_time == False)) * 1
    sell_exit = (sell_entry != 1) * 1
    signal = fs.calc_signal(buy_entry, buy_exit, sell_entry, sell_exit,
                            position)
    return signal