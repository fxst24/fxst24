# coding: utf-8
import forex_system as fs
import numpy as np
import pandas as pd

# パラメータの設定
UP = 8
DOWN = 3
PARAMETER = [UP, DOWN]
# 最適化の設定
START_UP = 0
END_UP = 23
STEP_UP = 1
START_DOWN = 0
END_DOWN = 23
STEP_DOWN = 1
RRANGES = (
    slice(START_UP, END_UP, STEP_UP),
    slice(START_DOWN, END_DOWN, STEP_DOWN),
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
    up = int(parameter[0])
    down = int(parameter[1])
    # 戦略を記述する。
    close = fs.i_close(symbol, timeframe, 0)
    month = fs.time_month(close.index)
    hour = fs.time_hour(close.index)
    buy = pd.Series(index=close.index)
    sell = pd.Series(index=close.index)
    if up < down:
        buy[(month >= 3) & (month < 11)] = ((hour >= (up + 1) % 24) &
            (hour < (down + 1) % 24)) * 1
        buy[(month < 3) | (month >= 11)] = ((hour >= up) & (hour < down)) * 1
        buy = buy.fillna(0)
        sell[buy != 1] = -1
        sell = sell.fillna(0)
    elif up > down:
        sell[(month >= 3) & (month < 11)] = ((hour >= (down + 1) % 24) &
            (hour < (up + 1) % 24)) * (-1)
        sell[(month < 3) | (month >= 11)] = (((hour >= down) & (hour < up)) *
            (-1))
        sell = sell.fillna(0)
        buy[sell != -1] = 1
        buy = buy.fillna(0)
    else:
        buy = pd.Series(np.zeros(len(close)), index=close.index)
        sell = pd.Series(np.zeros(len(close)), index=close.index)
    signal = buy + sell
    signal = signal.astype(int)

    return signal