# coding: utf-8

import forex_system as fs

# パラメータの設定
PARAMETER = None
# 最適化の設定
RRANGES = None

def strategy(parameter, symbol, timeframe):
    '''戦略を記述する。
    Args:
        parameter: パラメーター。
        symbol: 通貨ペア。
        timeframe: 期間。
    Returns:
        シグナル。
    '''
    # 戦略を記述する。
    close1 = fs.i_close(symbol, timeframe, 0)  # 誤って「0」にしている。
    close2 = fs.i_close(symbol, timeframe, 1)  # 誤って「1」にしている。
    buy_entry = (close2 < close1) * 1
    buy_exit = (close2 > close1) * 1
    sell_entry = (close2 > close1) * 1
    sell_exit = (close2 < close1) * 1
    signal = fs.calc_signal(buy_entry, buy_exit, sell_entry, sell_exit)

    return signal