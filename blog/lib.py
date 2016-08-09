# coding: utf-8

import forex_system
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ts
import tensorflow as tf

from datetime import datetime
from numba import float64, jit
from pandas_datareader import data as web
from scipy.optimize import curve_fit

# 複数の記事で使われる関数
def calc_bandwalk_std(period, n):
    '''バンドウォークの標準偏差を計算する。
        Args:
            period: 計算期間。
            n: データ数。
        Returns:
              バンドウォークの標準偏差。
    '''

    temp = np.random.randn(n)
    close = temp.cumsum()
    v = np.ones(period) / period
    mean = np.convolve(close, v, 'valid')
    close = close[period-1:]  
 
    @jit(float64(float64[:], float64[:]), nopython=True, cache=True)
    def func(close, mean):
        up = 0
        down = 0
        length = len(close)
        bandwalk = np.zeros(length)
        for i in range(length):
            if (close[i] > mean[i]):
                up = up + 1
            else:
                up = 0
            if (close[i] < mean[i]):
                down = down + 1
            else:
                down = 0
            bandwalk[i] = up - down
        return bandwalk.std()
 
    bandwalk_std = func(close, mean)

    return bandwalk_std

# 複数の記事で使われる関数
def create_time_series_data(*args):
    '''擬似時系列データを作成する。
      Args:
          *args: 可変長引数。
    '''

    a = args[0]  # 係数。
    t = args[1]  # 時期（データ数と考えていい）。

    e = np.random.randn(t)
    y = np.empty(t)
    y[0] = 0.0

    for i in (range(1, t)):
        y[i] = a * y[i - 1] +  e[i]

    return y

# http://fxst24.blog.fc2.com/blog-entry-248.html
def entry248(*args):
    '''バンドウォークの標準偏差を出力する。
      Args:
          *args: 可変長引数。
    '''
    max_period = args[0]
    n = args[1]

    bandwalk_std = np.empty(max_period)
 
    for i in range(max_period):
        period = i + 1
        bandwalk_std[i] = calc_bandwalk_std(period, n)

    # バンドウォークは計算期間が短いと不規則に見えるのでperiod=5から始める。
    bandwalk_std = bandwalk_std[4:]

    # モデルを作成する。
    def func(x, a, b):
        return x ** a + b

    # period=5から始める。
    x = list(range(5, max_period+1, 1))
    x = np.array(x)
    popt, pcov = curve_fit(func, x, bandwalk_std)
    a = popt[0]
    b = popt[1]
    model = x ** a + b

    # DataFrameに変換する。
    bandwalk_std = pd.Series(bandwalk_std)
    model = pd.Series(model)
    # グラフを出力する。
    result = pd.concat([bandwalk_std, model], axis=1)
    result.columns  = ['bandwalk_std', 'model']
    graph = result.plot()
    graph.set_xlabel('period')
    graph.set_ylabel('bandwalk')
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
               [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.show()
 
    # 傾きと切片を出力する。
    print('指数 = ', a)
    print('切片 = ', b)

# http://fxst24.blog.fc2.com/blog-entry-250.html
def entry250(*args):
    '''一定の的中率におけるシャープレシオを計算する。
      Args:
          *args: 可変長引数。
    '''

    accuracy = float(args[0])  # 的中率

    # 単純化のため、1日のボラティリティを固定する（0より大きな数値なら何でもよい）。
    volatility = 1.0

    # 1年当たりの営業日を260日、1日1トレードとしてトレード数を設定する。
    trades = 260

    # 1年当たりの利益を計算する（コストは考慮しない）。
    yearly_return = (((volatility * accuracy) - (volatility *
        (1.0 - accuracy))) * trades)

    # √Tルールに基づき、1年当たりのリスクを計算する。
    yearly_risk = volatility * np.sqrt(trades)

    # シャープレシオを計算する。
    sharpe_ratio = yearly_return / yearly_risk

    # 結果を出力する。
    print('的中率', round(accuracy * 100, 1), '% のシャープレシオ = ',
          sharpe_ratio)

# http://fxst24.blog.fc2.com/blog-entry-251.html
def entry251():
    '''Googleのレポートを検証する（①）。
    '''

    # ヒストリカルデータの開始日と終了日を設定する。
    start = datetime(2010, 1, 1)
    end = datetime(2015, 10, 1)
     
    # ヒストリカルデータをダウンロードする。
    snp = web.DataReader('^GSPC', 'yahoo', start, end)
    nyse = web.DataReader('^NYA', 'yahoo', start, end)
    djia = web.DataReader('^DJI', 'yahoo', start, end)
    nikkei = web.DataReader('^N225', 'yahoo', start, end)
    hangseng = web.DataReader('000001.SS', 'yahoo', start, end)
    ftse = web.DataReader('^FTSE', 'yahoo', start, end)
    dax = web.DataReader('^GDAXI', 'yahoo', start, end)
    aord = web.DataReader('^AORD', 'yahoo', start, end)
     
    # 終値を格納する。
    closing_data = pd.DataFrame()
    closing_data['snp_close'] = snp['Close']
    closing_data['nyse_close'] = nyse['Close']
    closing_data['djia_close'] = djia['Close']
    closing_data['nikkei_close'] = nikkei['Close']
    closing_data['hangseng_close'] = hangseng['Close']
    closing_data['ftse_close'] = ftse['Close']
    closing_data['dax_close'] = dax['Close']
    closing_data['aord_close'] = aord['Close']
     
    # 終値の欠損値を前のデータで補間する。
    closing_data = closing_data.fillna(method='ffill')
     
    # 終値の対数変化率を格納する。
    log_return_data = pd.DataFrame()
    log_return_data['snp_log_return'] = (np.log(closing_data['snp_close'] /
        closing_data['snp_close'].shift()))
    log_return_data['nyse_log_return'] = (np.log(closing_data['nyse_close'] /
        closing_data['nyse_close'].shift()))
    log_return_data['djia_log_return'] = (np.log(closing_data['djia_close'] /
        closing_data['djia_close'].shift()))
    log_return_data['nikkei_log_return'] = (np.log(closing_data['nikkei_close']
        / closing_data['nikkei_close'].shift()))
    log_return_data['hangseng_log_return'] = (
        np.log(closing_data['hangseng_close'] /
        closing_data['hangseng_close'].shift()))
    log_return_data['ftse_log_return'] = (np.log(closing_data['ftse_close'] /
        closing_data['ftse_close'].shift()))
    log_return_data['dax_log_return'] = (np.log(closing_data['dax_close'] /
        closing_data['dax_close'].shift()))
    log_return_data['aord_log_return'] = (np.log(closing_data['aord_close'] /
        closing_data['aord_close'].shift()))
     
    # S&P500の対数変化率が0以上なら1、さもなければ0を格納した列を加える。
    log_return_data['snp_log_return_positive'] = 0
    log_return_data.ix[log_return_data['snp_log_return'] >= 0,
                       'snp_log_return_positive'] = 1
     
    # S&P500の対数変化率が0未満なら1、さもなければ0を格納した列を加える。
    log_return_data['snp_log_return_negative'] = 0
    log_return_data.ix[log_return_data['snp_log_return'] < 0,
                       'snp_log_return_negative'] = 1
     
    # 学習・テスト用データを作成する。
    training_test_data = pd.DataFrame(
        columns=[
            'snp_log_return_positive', 'snp_log_return_negative',
            'snp_log_return_1', 'snp_log_return_2', 'snp_log_return_3',
            'nyse_log_return_1', 'nyse_log_return_2', 'nyse_log_return_3',
            'djia_log_return_1', 'djia_log_return_2', 'djia_log_return_3',
            'nikkei_log_return_1', 'nikkei_log_return_2', 'nikkei_log_return_3',
            'hangseng_log_return_1', 'hangseng_log_return_2',
                'hangseng_log_return_3',
            'ftse_log_return_1', 'ftse_log_return_2', 'ftse_log_return_3',
            'dax_log_return_1', 'dax_log_return_2', 'dax_log_return_3',
            'aord_log_return_1', 'aord_log_return_2', 'aord_log_return_3'])
    for i in range(7, len(log_return_data)):
        snp_log_return_positive = (
            log_return_data['snp_log_return_positive'].ix[i])
        snp_log_return_negative = (
            log_return_data['snp_log_return_negative'].ix[i])
        # 先読みバイアスを排除するため、当日のデータを使わない。
        snp_log_return_1 = log_return_data['snp_log_return'].ix[i-1]
        snp_log_return_2 = log_return_data['snp_log_return'].ix[i-2]
        snp_log_return_3 = log_return_data['snp_log_return'].ix[i-3]
        nyse_log_return_1 = log_return_data['nyse_log_return'].ix[i-1]
        nyse_log_return_2 = log_return_data['nyse_log_return'].ix[i-2]
        nyse_log_return_3 = log_return_data['nyse_log_return'].ix[i-3]
        djia_log_return_1 = log_return_data['djia_log_return'].ix[i-1]
        djia_log_return_2 = log_return_data['djia_log_return'].ix[i-2]
        djia_log_return_3 = log_return_data['djia_log_return'].ix[i-3]
        nikkei_log_return_1 = log_return_data['nikkei_log_return'].ix[i-1]
        nikkei_log_return_2 = log_return_data['nikkei_log_return'].ix[i-2]
        nikkei_log_return_3 = log_return_data['nikkei_log_return'].ix[i-3]
        hangseng_log_return_1 = log_return_data['hangseng_log_return'].ix[i-1]
        hangseng_log_return_2 = log_return_data['hangseng_log_return'].ix[i-2]
        hangseng_log_return_3 = log_return_data['hangseng_log_return'].ix[i-3]
        ftse_log_return_1 = log_return_data['ftse_log_return'].ix[i-1]
        ftse_log_return_2 = log_return_data['ftse_log_return'].ix[i-2]
        ftse_log_return_3 = log_return_data['ftse_log_return'].ix[i-3]
        dax_log_return_1 = log_return_data['dax_log_return'].ix[i-1]
        dax_log_return_2 = log_return_data['dax_log_return'].ix[i-2]
        dax_log_return_3 = log_return_data['dax_log_return'].ix[i-3]
        aord_log_return_1 = log_return_data['aord_log_return'].ix[i-1]
        aord_log_return_2 = log_return_data['aord_log_return'].ix[i-2]
        aord_log_return_3 = log_return_data['aord_log_return'].ix[i-3]

        # 各データをインデックスのラベルを使用しないで結合する。
        training_test_data = training_test_data.append(
            {'snp_log_return_positive':snp_log_return_positive,
            'snp_log_return_negative':snp_log_return_negative,
            'snp_log_return_1':snp_log_return_1,
            'snp_log_return_2':snp_log_return_2,
            'snp_log_return_3':snp_log_return_3,
            'nyse_log_return_1':nyse_log_return_1,
            'nyse_log_return_2':nyse_log_return_2,
            'nyse_log_return_3':nyse_log_return_3,
            'djia_log_return_1':djia_log_return_1,
            'djia_log_return_2':djia_log_return_2,
            'djia_log_return_3':djia_log_return_3,
            'nikkei_log_return_1':nikkei_log_return_1,
            'nikkei_log_return_2':nikkei_log_return_2,
            'nikkei_log_return_3':nikkei_log_return_3,
            'hangseng_log_return_1':hangseng_log_return_1,
            'hangseng_log_return_2':hangseng_log_return_2,
            'hangseng_log_return_3':hangseng_log_return_3,
            'ftse_log_return_1':ftse_log_return_1,
            'ftse_log_return_2':ftse_log_return_2,
            'ftse_log_return_3':ftse_log_return_3,
            'dax_log_return_1':dax_log_return_1,
            'dax_log_return_2':dax_log_return_2,
            'dax_log_return_3':dax_log_return_3,
            'aord_log_return_1':aord_log_return_1,
            'aord_log_return_2':aord_log_return_2,
            'aord_log_return_3':aord_log_return_3},
            ignore_index=True)
     
    # 3列目以降を説明変数として格納する。
    predictors_tf = training_test_data[training_test_data.columns[2:]]
    # 1、2列目を目的変数として格納する。
    classes_tf = training_test_data[training_test_data.columns[:2]]
    # 学習用セットのサイズを学習・テスト用データの80%に設定する。
    training_set_size = int(len(training_test_data) * 0.8)
    # 説明変数の初めの80%を学習用データにする。
    training_predictors_tf = predictors_tf[:training_set_size]
    # 目的変数の初めの80%を学習用データにする。
    training_classes_tf = classes_tf[:training_set_size]
    # 説明変数の残りの20%をテスト用データにする。
    test_predictors_tf = predictors_tf[training_set_size:]
    # 目的変数の残りの20%をテスト用データにする。
    test_classes_tf = classes_tf[training_set_size:]
     
    def tf_confusion_metrics(model, actual_classes, session, feed_dict):
        '''正解率等を表示する。
          Args:
              model: 
              actual_classes: 
              session: 
              feed_dict: 
        '''

        #
        predictions = tf.argmax(model, 1)
        #
        actuals = tf.argmax(actual_classes, 1)
        #
        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)
        tp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                tf.equal(actuals, ones_like_actuals), 
                tf.equal(predictions, ones_like_predictions)
                ), 
                "float"
            )
        )
        tn_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                tf.equal(actuals, zeros_like_actuals), 
                tf.equal(predictions, zeros_like_predictions)
                ), 
                "float"
             )
        )    
        fp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                tf.equal(actuals, zeros_like_actuals), 
                tf.equal(predictions, ones_like_predictions)
                ), 
                "float"
            )
        )    
        fn_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                tf.equal(actuals, ones_like_actuals), 
                tf.equal(predictions, zeros_like_predictions)
                ), 
                "float"
            )
        )
        tp, tn, fp, fn = \
            session.run(
                [tp_op, tn_op, fp_op, fn_op], 
                feed_dict
            )
        tpr = float(tp)/(float(tp) + float(fn))
        accuracy = ((float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) +
            float(tn)))
        recall = tpr
        precision = float(tp)/(float(tp) + float(fp))  
        f1_score = (2 * (precision * recall)) / (precision + recall)  
        print('Precision = ', precision)
        print('Recall = ', recall)
        print('F1 Score = ', f1_score)
        print('Accuracy = ', accuracy)
     
    #
    sess = tf.Session()
     
    # 説明変数と目的変数の数を格納する。
    num_predictors = len(training_predictors_tf.columns)
    num_classes = len(training_classes_tf.columns)
     
    # 
    feature_data = tf.placeholder("float", [None, num_predictors])
    actual_classes = tf.placeholder("float", [None, num_classes])
     
    # 重みとバイアスを格納する変数を生成する。
    weights = tf.Variable(
        tf.truncated_normal([num_predictors, num_classes], stddev=0.0001))
    biases = tf.Variable(tf.ones([num_classes]))
     
    # モデルを定義する。
    model = tf.nn.softmax(tf.matmul(feature_data, weights) + biases)
     
    # コスト関数を定義する。
    cost = -tf.reduce_sum(actual_classes*tf.log(model))
     
    # 学習率を設定する。
    training_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
     
    # 
    init = tf.initialize_all_variables()
    sess.run(init)
     
    # 正解した予測を格納する。
    correct_prediction = tf.equal(tf.argmax(model, 1),
                                  tf.argmax(actual_classes, 1))
     
    # 正解率を格納する。
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
     
    # 学習を実行する。
    print('二項分類を用いたモデル')
    print('学習用データを用いた正解率')
    for i in range(1, 30001):
        sess.run(
            training_step, 
            feed_dict={
                feature_data: training_predictors_tf.values, 
                actual_classes: training_classes_tf.values.reshape(
                    len(training_classes_tf.values), 2)
            }
        )
        if i%5000 == 0:
            print(i, sess.run(
                accuracy,
                feed_dict={
                    feature_data: training_predictors_tf.values, 
                    actual_classes: training_classes_tf.values.reshape(
                        len(training_classes_tf.values), 2)
                }
            ))
     
    # 
    feed_dict= {
        feature_data: test_predictors_tf.values,
        actual_classes: test_classes_tf.values.reshape(
            len(test_classes_tf.values), 2)
    }
     
    #
    print('テスト用データを用いた検証結果')
    tf_confusion_metrics(model, actual_classes, sess, feed_dict)
     
    #
    sess1 = tf.Session()
     
    # 説明変数と目的変数の数を格納する。
    num_predictors = len(training_predictors_tf.columns)
    num_classes = len(training_classes_tf.columns)
     
    # 
    feature_data = tf.placeholder("float", [None, num_predictors])
    actual_classes = tf.placeholder("float", [None, 2])
     
    # 重み1-3、バイアス1-3を格納する変数を生成する。
    weights1 = tf.Variable(tf.truncated_normal([24, 50], stddev=0.0001))
    biases1 = tf.Variable(tf.ones([50]))
    weights2 = tf.Variable(tf.truncated_normal([50, 25], stddev=0.0001))
    biases2 = tf.Variable(tf.ones([25]))
    weights3 = tf.Variable(tf.truncated_normal([25, 2], stddev=0.0001))
    biases3 = tf.Variable(tf.ones([2]))
     
    # 隠れ層を生成する。
    hidden_layer_1 = tf.nn.relu(tf.matmul(feature_data, weights1) + biases1)
    hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weights2) + biases2)
     
    # モデルを定義する。
    model = tf.nn.softmax(tf.matmul(hidden_layer_2, weights3) + biases3)
     
    # コスト関数を定義する。
    cost = -tf.reduce_sum(actual_classes*tf.log(model))
     
    # 学習率を設定する。
    train_op1 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
     
    # 
    init = tf.initialize_all_variables()
    sess1.run(init)
     
    # 正解した予測を格納する。
    correct_prediction = tf.equal(tf.argmax(model, 1),
                                  tf.argmax(actual_classes, 1))
     
    # 正解率を格納する。
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
     
    # 学習を実行する。
    print('隠れ層を加えたモデル')
    print('学習用データを用いた正解率')
    for i in range(1, 30001):
        sess1.run(
            train_op1, 
            feed_dict={
                feature_data: training_predictors_tf.values, 
                actual_classes: training_classes_tf.values.reshape(
                    len(training_classes_tf.values), 2)
            }
        )
        if i%5000 == 0:
            print(i, sess1.run(
                accuracy,
                feed_dict = {
                    feature_data: training_predictors_tf.values, 
                    actual_classes: training_classes_tf.values.reshape(
                        len(training_classes_tf.values), 2)
                }
            ))
     
    #
    feed_dict= {
        feature_data: test_predictors_tf.values,
        actual_classes: test_classes_tf.values.reshape(
            len(test_classes_tf.values), 2)
    }
     
    #
    print('テスト用データを用いた検証結果')
    tf_confusion_metrics(model, actual_classes, sess1, feed_dict)

# http://fxst24.blog.fc2.com/blog-entry-255.html
def entry255():
    '''Googleのレポートを検証する（②）。
    '''

    # ヒストリカルデータの開始日と終了日を設定する。
    start = datetime(2010, 1, 1)
    end = datetime(2015, 10, 1)
      
    # ヒストリカルデータをダウンロードする。
    snp = web.DataReader('^GSPC', 'yahoo', start, end)
    ftse = web.DataReader('^FTSE', 'yahoo', start, end)

    # 終値を格納する。
    closing_data = pd.DataFrame()
    closing_data['snp_close'] = snp['Close']
    closing_data['ftse_close'] = ftse['Close']
      
    # 終値の欠損値を補間する。
    closing_data = closing_data.fillna(method='ffill')
      
    # 終値の対数変化率を格納する。
    log_return_data = pd.DataFrame()
    log_return_data['snp_log_return'] = (
    np.log(closing_data['snp_close'] /
        closing_data['snp_close'].shift()))
    log_return_data['ftse_log_return'] = (np.log(closing_data['ftse_close'] /
        closing_data['ftse_close'].shift()))
    # 最初の行はNaNなので削除する。
    log_return_data = log_return_data[1:]
      
    correct_prediction = (log_return_data['snp_log_return'] *
        log_return_data['ftse_log_return'])
    correct_prediction[correct_prediction>=0] = 1
    correct_prediction[correct_prediction<0] = 0
    accuracy = correct_prediction.sum() / len(correct_prediction)
    print('学習なしの超シンプルなモデル')
    print('Accuracy = ', accuracy)

# http://fxst24.blog.fc2.com/blog-entry-263.html
def entry263(*args):
    '''勝つためのトレード数を計算する。
      Args:
          *args: 可変長引数。
    '''

    target = float(args[0])  # 目標勝率
    wp = float(args[1])  # 1トレード当たりの想定勝率
    prob = 0.0
    n_trade = -1

    def choose(n, r):
        '''組み合わせの数を返す（PythonにRのchoose()関数に相当するものはないのか？）。
          Args:
              n: 異なるものの数。
              r: 選ぶ数。
          Returns:
              組み合わせの数。
        '''
        if n == 0 or r == 0:
            return 1
        else:
            return choose(n, r - 1) * (n - r + 1) / r

    while (prob < target):
        n_trade = n_trade + 2 # 必ず奇数にする
        prob = 0.0
        for i in range(int((n_trade + 1) / 2), n_trade + 1):
            r = i
            temp = choose(n_trade, r) * wp**r * (1 - wp)**(n_trade - r)
            prob = prob + temp

    print("週単位で勝つための1日当たりのトレード数 = ", n_trade / 5.0)
    print("月単位で勝つための1日当たりのトレード数 = ", n_trade / 20.0)

# http://fxst24.blog.fc2.com/blog-entry-269.html
def entry269(*args):
    '''エスケープ処理を行う。
      Args:
          *args: 可変長引数。
    '''

    # 入力ファイルを読み込む。
    filename_input = args[0]
    file_input = open(filename_input)
    code_input = file_input.read()
    file_input.close()

    # エスケープ処理を行う。
    code_output = code_input.replace("&", "&amp;")  # これは最初に持ってくる。
    code_output = code_output.replace(">", "&gt;")
    code_output = code_output.replace("<", "&lt;")
    code_output = code_output.replace("\"", "&quot;")

    # 出力ファイルに書き込む。
    filename_output = filename_input + ".txt"
    file_output = open(filename_output, "w")
    file_output.write(code_output)
    file_output.close()

# http://fxst24.blog.fc2.com/blog-entry-271.html
def entry271(*args):
    '''トレード数を推定する。
      Args:
          *args: 可変長引数。
    '''

    mode = int(args[0])  # モード 0:計算期間 1:zスコア
    n = int(args[1])  # データ数

    # 「zスコア = 0.0」の場合の計算期間とトレード数の関係を見る。
    if mode == 0:
        entry_threshold = 0.0
        trades = np.empty(46)
        # 実行時間を短縮するため、計算期間は5〜50に限定する。
        for i in range(46):
            # シグナルを計算する。
            period = i + 5
            close = create_time_series_data(1.0, n)
            close = pd.Series(close)
            mean = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            z_score = (close - mean) / std
            longs_entry = (z_score <= -entry_threshold) * 1
            longs_exit = (z_score >= 0.0) * 1
            shorts_entry = (z_score >= entry_threshold) * 1
            shorts_exit = (z_score <= 0.0) * 1
            longs = longs_entry.copy()
            longs[longs==0] = np.nan
            longs[longs_exit==1] = 0
            longs = longs.fillna(method='ffill')
            shorts = -shorts_entry.copy()
            shorts[shorts==0] = np.nan
            shorts[shorts_exit==1] = 0
            shorts = shorts.fillna(method='ffill')
            signal = longs + shorts
            signal = signal.shift()
            signal = signal.fillna(0)
            signal = signal.astype(int)

            # トレード数を計算する。
            temp1 = (((signal > 0) & (signal > signal.shift(1))) *
                (signal - signal.shift(1)))
            temp2 = (((signal < 0) & (signal < signal.shift(1))) *
                (signal.shift(1) - signal))
            trade = temp1 + temp2
            trade = trade.fillna(0)
            trade = trade.astype(int)
            trades[i] = trade.sum()

        # トレード数を割合に変換する。    
        trades = np.array(trades) / n
    
        # トレード数を予測する。。
        def func(x, a, b):
            return a * x ** b
        x = list(range(5, 51, 1))
        x = np.array(x)
        popt, pcov = curve_fit(func, x, trades)
        a = popt[0]
        b = popt[1]
        pred = a * x ** b
    
        # DataFrameに変換する。
        trades = pd.Series(trades)
        pred = pd.Series(pred)

        # グラフを出力する。
        result = pd.concat([trades, pred], axis=1)
        result.columns  = ['trades', 'pred']
        graph = result.plot()
        graph.set_xlabel('period')
        graph.set_ylabel('trades')
        plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                   [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        plt.show()
     
        # 傾きと指数を出力する。
        print('傾き = ', a)
        print('指数 = ', b)

    # 「計算期間 = 20」の場合のzスコアとトレード数の関係を見る。
    else:  # mode == 1
        period = 20
        trades = np.empty(46)
        # 実行時間を短縮するため、zスコアはは0.125〜1.25に限定する。
        for i in range(46):
            # シグナルを計算する。
            entry_threshold = (i + 5) * 0.025
            close = create_time_series_data(1.0, n)
            close = pd.Series(close)
            mean = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            z_score = (close - mean) / std
            longs_entry = (z_score <= -entry_threshold) * 1
            longs_exit = (z_score >= 0.0) * 1
            shorts_entry = (z_score >= entry_threshold) * 1
            shorts_exit = (z_score <= 0.0) * 1
            longs = longs_entry.copy()
            longs[longs==0] = np.nan
            longs[longs_exit==1] = 0
            longs = longs.fillna(method='ffill')
            shorts = -shorts_entry.copy()
            shorts[shorts==0] = np.nan
            shorts[shorts_exit==1] = 0
            shorts = shorts.fillna(method='ffill')
            signal = longs + shorts
            signal = signal.shift()
            signal = signal.fillna(0)
            signal = signal.astype(int)

            # トレード数を計算する。
            temp1 = (((signal > 0) & (signal > signal.shift(1))) *
                (signal - signal.shift(1)))
            temp2 = (((signal < 0) & (signal < signal.shift(1))) *
                (signal.shift(1) - signal))
            trade = temp1 + temp2
            trade = trade.fillna(0)
            trade = trade.astype(int)
            trades[i] = trade.sum()

        # トレード数を割合に変換する。
        trades = np.array(trades) / n
    
        # トレード数を予測する。
        def func(x, a, b, c):
            return a * x ** b + c
        x = list(range(5, 51, 1))
        x = np.array(x) / 40.0
        popt, pcov = curve_fit(func, x, trades)
        a = popt[0]
        b = popt[1]
        c = popt[2]
        pred = a * x ** b + c
    
        # DataFrameに変換する。
        trades = pd.Series(trades)
        pred = pd.Series(pred)

        # グラフを出力する。
        result = pd.concat([trades, pred], axis=1)
        result.columns  = ['trades', 'pred']
        graph = result.plot()
        graph.set_xlabel('z_score')
        graph.set_ylabel('trades')
        plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                   [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.125,
                    1.25])
        plt.show()
     
        # 傾き、指数、切片を出力する。
        print('傾き = ', a)
        print('指数 = ', b)
        print('切片 = ', c)

# http://fxst24.blog.fc2.com/blog-entry-272.html
def entry272(*args):
    '''平均回帰戦略をバックテストする。
      Args:
          *args: 可変長引数。
    '''

    a = args[0]  # 係数（係数=1で単位根）
    t = args[1]  # 時期（データ数と考えていい）

    period = 20  # 計算期間
    entry_threshold = 2.0

    # シグナルを計算する。
    close = create_time_series_data(a, t)
    close = pd.Series(close)
    mean = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    z_score = (close - mean) / std
    longs_entry = (z_score <= -entry_threshold) * 1
    longs_exit = (z_score >= 0.0) * 1
    shorts_entry = (z_score >= entry_threshold) * 1
    shorts_exit = (z_score <= 0.0) * 1
    longs = longs_entry.copy()
    longs[longs==0] = np.nan
    longs[longs_exit==1] = 0
    longs = longs.fillna(method='ffill')
    shorts = -shorts_entry.copy()
    shorts[shorts==0] = np.nan
    shorts[shorts_exit==1] = 0
    shorts = shorts.fillna(method='ffill')
    signal = longs + shorts
    signal = signal.shift()
    signal = signal.fillna(0)
    signal = signal.astype(int)

    # リターンを計算する。
    ret = ((close - close.shift()) * signal)
    ret = ret.fillna(0.0)
    ret[(ret==float('inf')) | (ret==float('-inf'))] = 0.0

    # 終値のグラフを出力する。
    graph = close.plot()
    graph.set_xlabel('t')
    graph.set_ylabel('close')
    plt.show()

    # 資産曲線のグラフを出力する。
    cumret = ret.cumsum()
    graph = cumret.plot()
    graph.set_xlabel('t')
    graph.set_ylabel('equity_curve')
    plt.show()

    # 単位根検定のp値を出力する。
    p_value = ts.adfuller(close)[1]
    p_value = "{0:%}".format(p_value)
    print('単位根検定のp値 = ', p_value)

# http://fxst24.blog.fc2.com/blog-entry-276.html
def entry276(*args):
    '''需給均衡とランダムウォークの関係を調べる。
      Args:
          *args: 可変長引数。
    '''

    # 初期設定
    N = 10000  # 試行回数
    PRICE = 100  # 価格
    EQUILIBRIUM_PRICE = 0  # 均衡価格
    
    # シミュレーション①
    price = PRICE
    equilibrium_price = EQUILIBRIUM_PRICE
    data = np.empty(N)  # 価格
    for i in range(N):
        data[i] = price
    
        if price > equilibrium_price:
            price = price - 1
        elif price < equilibrium_price:
            price = price + 1
        else:
            pass
    
    # 単位根検定のp値を計算する。
    p_value = ts.adfuller(data)[1]
    p_value = "{0:%}".format(p_value)
    
    data = pd.Series(data)
    graph = data.plot()
    graph.set_title('simulation1' + '(p_value = ' + p_value + ')')
    graph.set_xlabel('time')
    graph.set_ylabel('price')
    plt.show()
    
    # シミュレーション②
    price = PRICE
    equilibrium_price = EQUILIBRIUM_PRICE
    data = np.empty(N)
    for i in range(N):
        data[i] = price
    
        if price > equilibrium_price:
            price = price - 1
        elif price < equilibrium_price:
            price = price + 1
        else:
            pass
    
        price = price + np.random.choice([1, -1])
    
    p_value = ts.adfuller(data)[1]
    p_value = "{0:%}".format(p_value)
    
    data = pd.Series(data)
    graph = data.plot()
    graph.set_title('simulation2' + '(p_value = ' + p_value + ')')
    graph.set_xlabel('time')
    graph.set_ylabel('price')
    plt.show()
    
    # シミュレーション③
    price = PRICE
    equilibrium_price = EQUILIBRIUM_PRICE
    data = np.empty(N)
    for i in range(N):
        data[i] = price

        if price > equilibrium_price:
            price = price - 1
        elif price < equilibrium_price:
            price = price + 1
        else:
            pass

        if price > equilibrium_price:
            equilibrium_price = equilibrium_price + 1
        elif price < equilibrium_price:
            equilibrium_price = equilibrium_price - 1
        else:
            pass
    
    p_value = ts.adfuller(data)[1]
    p_value = "{0:%}".format(p_value)
    
    data = pd.Series(data)
    graph = data.plot()
    graph.set_title('simulation3' + '(p_value = ' + p_value + ')')
    graph.set_xlabel('time')
    graph.set_ylabel('price')
    plt.show()
    
    # シミュレーション④
    price = PRICE
    equilibrium_price = EQUILIBRIUM_PRICE
    data = np.empty(N)
    for i in range(N):
        data[i] = price

        if price > equilibrium_price:
            price = price - 1
        elif price < equilibrium_price:
            price = price + 1
        else:
            pass
    
        price = price + np.random.choice([1, -1])

        if price > equilibrium_price:
            equilibrium_price = equilibrium_price + 1
        elif price < equilibrium_price:
            equilibrium_price = equilibrium_price - 1
        else:
            pass
    
    p_value = ts.adfuller(data)[1]
    p_value = "{0:%}".format(p_value)
    
    data = pd.Series(data)
    graph = data.plot()
    graph.set_title('simulation4' + '(p_value = ' + p_value + ')')
    graph.set_xlabel('time')
    graph.set_ylabel('price')
    plt.show()

# http://fxst24.blog.fc2.com/blog-entry-279.html
def entry279(*args):
    '''勝つための的中率を計算する。
      Args:
          *args: 可変長引数。
    '''

    width = float(args[0])  # 値幅
    cost = float(args[1])  # コスト
    profit = width - cost  # 的中した場合の利益
    loss = width + cost  # 的中しなかった場合の損失
    n = 100  # 試行回数
    a = 0.0  # 下限（初期値）
    b = 1.0  # 上限（初期値）
    prob = (a + b) / 2 # 的中率

    # 勝つための的中率を二分法で計算する。
    for i in (range(n)):
        if (profit * prob >= loss * (1 - prob)):
            b = prob
            prob = (a + b) / 2
        else:
            a = prob
            prob = (a + b) / 2

    print("勝つための的中率 = ", prob)

# http://fxst24.blog.fc2.com/blog-entry-280.html
def entry280(*args):
    '''勝つためのボラティリティを計算する。
      Args:
          *args: 可変長引数。
    '''

    prob = float(args[0])  # 的中率
    cost = float(args[1])  # コスト
    n = 100 # 試行回数
    a = 0.0 # 下限（初期値）
    b = 100.0 # 上限（初期値）
    width = (a + b) / 2 # ボラティリティ
    profit = width - cost # 的中した場合の利益
    loss = width + cost # 的中しなかった場合の損失

    # 勝つためのボラティリティを二分法で計算する。
    for i in range(n):
        if (profit * prob >= loss * (1 - prob)):
            b = width
            width = (a + b) / 2
            profit = width - cost
            loss = width + cost
        else:
            a = width
            width = (a + b) / 2
            profit = width - cost
            loss = width + cost

    print("勝つためのボラティリティ = ", width)

# http://fxst24.blog.fc2.com/blog-entry-281.html
def entry281(*args):
    '''各足のボラティリティを計算する。
      Args:
          *args: 可変長引数。
    '''

    symbol = args[0]  # 通貨ペア
    start_year = int(args[1])  # 開始年
    start_month = int(args[2])  # 開始月
    start_day = int(args[3])  # 開始日
    end_year = int(args[4])  # 終了年
    end_month = int(args[5])  # 終了月
    end_day = int(args[6])  # 終了日

    start = datetime(start_year, start_month, start_day, 0, 0)
    end = datetime(end_year, end_month, end_day, 23, 59)
 
    # ボラティリティを計算する。
    fs = forex_system.ForexSystem()
    if (symbol == 'AUDJPY' or symbol == 'CADJPY' or symbol == 'CHFJPY' or
        symbol == 'EURJPY' or symbol == 'GBPJPY' or symbol == 'NZDJPY' or
        symbol == 'USDJPY'):
        values2pips = 100.0
    else:
    # symbol == 'AUDCAD', 'AUDCHF', 'AUDNZD', 'AUDUSD', 'CADCHF', 'EURAUD',
    #           'EURCAD', 'EURCHF', 'EURGBP', 'EURNZD', 'EURUSD', 'GBPAUD',
    #           'GBPCAD', 'GBPCHF', 'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF',
    #           'NZDUSD', 'USDCAD', 'USDCHF'
        values2pips = 10000.0
    vola1 = fs.i_diff(symbol, 1, 0)[start:end].std() * values2pips
    vola5 = fs.i_diff(symbol, 5, 0)[start:end].std() * values2pips
    vola15 = fs.i_diff(symbol, 15, 0)[start:end].std() * values2pips
    vola30 = fs.i_diff(symbol, 30, 0)[start:end].std() * values2pips
    vola60 = fs.i_diff(symbol, 60, 0)[start:end].std() * values2pips
    vola240 = fs.i_diff(symbol, 240, 0)[start:end].std() * values2pips
    vola1440 = fs.i_diff(symbol, 1440, 0)[start:end].std() * values2pips
 
    # ボラティリティを出力する。
    print("通貨ペア = ", symbol)
    print("1分足のボラティリティ = ", vola1)
    print("5分足のボラティリティ = ", vola5)
    print("15分足のボラティリティ = ", vola15)
    print("30分足のボラティリティ = ", vola30)
    print("1時間足のボラティリティ = ", vola60)
    print("4時間足のボラティリティ = ", vola240)
    print("日足のボラティリティ = ", vola1440)

# http://fxst24.blog.fc2.com/blog-entry-281.html
def entry285(*args):
    '''VIX（FX版）を計算する。
      Args:
          *args: 可変長引数。
    '''

    symbol = args[0]  # 通貨ペア
    timeframe = args[1]  # タイムフレーム。
    start_year = int(args[2])  # 開始年
    start_month = int(args[3])  # 開始月
    start_day = int(args[4])  # 開始日
    end_year = int(args[5])  # 終了年
    end_month = int(args[6])  # 終了月
    end_day = int(args[7])  # 終了日

    start = datetime(start_year, start_month, start_day, 0, 0)
    end = datetime(end_year, end_month, end_day, 23, 59)
 
    # VIX（FX版）を計算する。
    fs = forex_system.ForexSystem()
    vix4fx = fs.i_vix4fx(symbol, timeframe, 1)[start:end]

    # グラフを表示する。
    graph = vix4fx.plot()
    plt.show(graph)
    plt.close()