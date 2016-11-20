# coding: utf-8

import forex_system as fs
import pandas as pd
from datetime import datetime
from sklearn import tree

# パラメータの設定
PARAMETER = None
# 最適化の設定
RRANGES = None

def strategy(parameter, symbol, timeframe):
    '''戦略を記述する。
    Args:
        parameter: パラメータ。
        symbol: 通貨ペア名。
        timeframe: 足の種類。
    Returns:
        シグナル。
    '''
    # 戦略を記述する。
    ret1 = fs.i_log_return(symbol, timeframe, 1, 1)
    ret2 = fs.i_log_return(symbol, timeframe, 1, 2)
    index = ret1.index
    start = datetime.strptime('2010.01.01', '%Y.%m.%d')
    end = datetime.strptime('2012.12.31', '%Y.%m.%d')
    clf = tree.DecisionTreeRegressor()
    y = ret1[start:end]
    x = ret2[start:end]
    x = x.reshape(len(x), 1)
    clf.fit(x, y)
    pred = clf.predict(ret1.reshape(len(ret1), 1))
    pred = pd.Series(pred, index=index)
    buy_entry = (pred > 0.0) * 1
    buy_exit = (buy_entry != 1) * 1
    sell_entry = (pred < 0.0) * 1
    sell_exit = (sell_entry != 1) * 1
    buy_entry = buy_entry.fillna(0)
    buy_exit = buy_exit.fillna(0)
    sell_entry = sell_entry.fillna(0)
    sell_exit = sell_exit.fillna(0)
    signal = fs.calc_signal(buy_entry, buy_exit, sell_entry, sell_exit)

    return signal