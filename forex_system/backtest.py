# coding: utf-8

import argparse
import forex_system as fs
import os
import shutil
import time

if __name__ == '__main__':
    # 開始時間を記録する。
    start_time = time.time()

    # 一時フォルダが残っていたら削除する。
    path = os.path.dirname(__file__)
    if os.path.exists(path + '/tmp') == True:
        shutil.rmtree(path + '/tmp')

    # 一時フォルダを作成する。
    os.mkdir(path + '/tmp')

    parser = argparse.ArgumentParser()
    parser.add_argument('--ea1', nargs='*')
    parser.add_argument('--ea2', nargs='*')
    parser.add_argument('--ea3', nargs='*')
    parser.add_argument('--ea4', nargs='*')
    parser.add_argument('--ea5', nargs='*')
    args = parser.parse_args()

    parameter_ea2 = None
    parameter_ea3 = None
    parameter_ea4 = None
    parameter_ea5 = None

    ret_ea1, trades_ea1, parameter_ea1, timeframe, start, end = (
        fs.backtest(args.ea1))
    ret = ret_ea1
    trades = trades_ea1
    if args.ea2 is not None:
        ret_ea2, trades_ea2, parameter_ea2, timeframe, start, end = (
            fs.backtest(args.ea2))
        ret += ret_ea2
        trades += trades_ea2
    if args.ea3 is not None:
        ret_ea3, trades_ea3, parameter_ea3, timeframe, start, end = (
            fs.backtest(args.ea3))
        ret += ret_ea3
        trades += trades_ea3
    if args.ea4 is not None:
        ret_ea4, trades_ea4, parameter_ea4, timeframe, start, end = (
            fs.backtest(args.ea4))
        ret += ret_ea4
        trades += trades_ea4
    if args.ea5 is not None:
        ret_ea5, trades_ea5, parameter_ea5, timeframe, start, end = (
            fs.backtest(args.ea5))
        ret += ret_ea5
        trades += trades_ea5

    fs.show_backtest_result(ret, trades, timeframe, start, end, parameter_ea1,
                parameter_ea2, parameter_ea3, parameter_ea4, parameter_ea5)

    # 一時フォルダを削除する。
    path = os.path.dirname(__file__)
    if os.path.exists(path + '/tmp') == True:
        shutil.rmtree(path + '/tmp')

    # 終了時間を記録する。
    end_time = time.time()

    # 実行時間を出力する。
    if end_time - start_time < 60.0:
        print(
            '実行時間は',
            int(round(end_time - start_time)), '秒です。')
    else:
        print(
            '実行時間は',
            int(round((end_time - start_time) / 60.0)), '分です。')