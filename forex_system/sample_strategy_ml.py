# coding: utf-8

import forex_system as fs
import numpy as np
import pandas as pd
from sklearn import tree

MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 5
START = 60
STOP = 1500
STEP = 60
THRESHOLD = 2.0
PARAMETER = [START, STOP, STEP, THRESHOLD, MAX_DEPTH, MIN_SAMPLES_LEAF]

def load_x(parameter, symbol, timeframe):
    '''説明変数をロードする。
    Args:
        parameter: パラメーター。
        symbol: 通貨ペア。
        timeframe: 期間。
    Returns:
        説明変数。
    '''
    # パラメーターを格納する。
    start = int(parameter[0])
    stop = int(parameter[1])
    step = int(parameter[2])
    for minute in range(start, stop, step):
        period = fs.convert_minute2period(minute, timeframe)
        log_return1 = fs.i_log_return(symbol, timeframe, period, 1)
        if minute == start:
            x = log_return1
        else:
            x = pd.concat([x, log_return1], axis=1)
    x = x.fillna(0.0)

    return x

def build_model(parameter, symbol, timeframe, start_train, end_train):
    '''モデルを作成する。
    Args:
        parameter: パラメーター。
        symbol: 通貨ペア。
        timeframe: 期間。
        start_train: 学習期間の開始日。
        end_train: 学習期間の終了日。
    Returns:
        モデル、学習期間における予測値の標準偏差。
    '''
    # パラメーターを格納する。
    if parameter[4] is None:
        max_depth = parameter[4]
    else:
        max_depth = int(parameter[4])
    min_samples_leaf = int(parameter[5])
    # 説明変数をロードする。
    x = load_x(parameter, symbol, timeframe)
    # 目的変数のデータを準備する（絶対値が小さすぎるとうまくいかないようだ）。
    y = fs.i_log_return(symbol, timeframe, 1, 0) * 100.0
    # 学習用データを切り取る。
    x_train = x[start_train:end_train]
    y_train = y[start_train:end_train]
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    # モデルを作成する。
    model = tree.DecisionTreeRegressor(max_depth=max_depth,
                                       min_samples_leaf=min_samples_leaf)
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_train_std = np.std(pred_train)
    # モデルを保存する。
    filename = fs.get_current_filename_no_ext()
    fs.save_model(model, filename)

    return model, pred_train_std

def strategy(parameter, symbol, timeframe, model, pred_train_std):
    '''戦略を記述する。
    Args:
        parameter: パラメーター。
        symbol: 通貨ペア。
        timeframe: 期間。
        model: モデル。
        pred_train_std: 学習期間での予測値の標準偏差。
    Returns:
        シグナル。
    '''
    # パラメーターを格納する。
    threshold = float(parameter[3])
    # 説明変数をロードする。
    x = load_x(parameter, symbol, timeframe)
    index = x.index
    x = np.array(x)
    # シグナルを計算する。
    pred = model.predict(x) / pred_train_std
    pred = pd.Series(pred, index=index)
    buy_entry = (pred >= threshold) * 1
    buy_exit = (pred <= 0.0) * 1
    sell_entry = (pred <= -threshold) * 1
    sell_exit = (pred >= 0.0) * 1
    signal = fs.calc_signal(buy_entry, buy_exit, sell_entry, sell_exit)

    return signal