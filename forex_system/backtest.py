# coding: utf-8

import argparse
import forex_system as fs
from datetime import datetime

if __name__ == '__main__':
    # 一時フォルダを削除してから作成する。
    fs.remove_temp_folder()
    fs.make_temp_folder()
    # 全体のオプションを設定する。
    parser = argparse.ArgumentParser()
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--optimization', default=0, type=int)
    parser.add_argument('--in_sample_period', default=360, type=int)
    parser.add_argument('--out_of_sample_period', default=30, type=int)
    # EA1のオプションを設定する。
    parser.add_argument('--ea1')
    parser.add_argument('--symbol1')
    parser.add_argument('--timeframe1', type=int)
    parser.add_argument('--spread1', type=int)
    parser.add_argument('--position1', default=2, type=int)
    parser.add_argument('--min_trade1', default=260, type=int)
    parser.add_argument('--ml1', default=0, type=int)
    # EA2のオプションを設定する。
    parser.add_argument('--ea2')
    parser.add_argument('--symbol2')
    parser.add_argument('--timeframe2', type=int)
    parser.add_argument('--spread2', type=int)
    parser.add_argument('--position2', default=2, type=int)
    parser.add_argument('--min_trade2', default=260, type=int)
    parser.add_argument('--ml2', default=0, type=int)
    # EA3のオプションを設定する。
    parser.add_argument('--ea3')
    parser.add_argument('--symbol3')
    parser.add_argument('--timeframe3', type=int)
    parser.add_argument('--spread3', type=int)
    parser.add_argument('--position3', default=2, type=int)
    parser.add_argument('--min_trade3', default=260, type=int)
    parser.add_argument('--ml3', default=0, type=int)
    # EA4のオプションを設定する。
    parser.add_argument('--ea4')
    parser.add_argument('--symbol4')
    parser.add_argument('--timeframe4', type=int)
    parser.add_argument('--spread4', type=int)
    parser.add_argument('--position4', default=2, type=int)
    parser.add_argument('--min_trade4', default=260, type=int)
    parser.add_argument('--ml4', default=0, type=int)
    # EA5のオプションを設定する。
    parser.add_argument('--ea5')
    parser.add_argument('--symbol5')
    parser.add_argument('--timeframe5', type=int)
    parser.add_argument('--spread5', type=int)
    parser.add_argument('--position5', default=2, type=int)
    parser.add_argument('--min_trade5', default=260, type=int)
    parser.add_argument('--ml5', default=0, type=int)
    # 全体の設定を格納する。
    args = parser.parse_args()
    start = datetime.strptime(args.start + ' 00:00', '%Y.%m.%d %H:%M')
    end = datetime.strptime(args.end + ' 23:59', '%Y.%m.%d %H:%M')
    optimization = args.optimization
    in_sample_period = args.in_sample_period
    out_of_sample_period = args.out_of_sample_period
    # EA1のバックテストを行う。
    ea1 = args.ea1
    symbol1 = args.symbol1
    timeframe1 = args.timeframe1
    spread1 = args.spread1
    position1 = args.position1
    min_trade1 = args.min_trade1
    ml1 = args.ml1
    ret_ea1, trades_ea1, parameter_ea1, start_test, end_test = (
        fs.backtest(start, end, optimization, in_sample_period,
        out_of_sample_period, ea1, symbol1, timeframe1, spread1, position1,
        min_trade1, ml1))
    ret = ret_ea1
    trades = trades_ea1
    parameter_ea2 = None
    parameter_ea3 = None
    parameter_ea4 = None
    parameter_ea5 = None
    # EA2のバックテストを行う。
    if args.ea2 is not None:
        ea2 = args.ea2
        symbol2 = args.symbol2
        timeframe2 = args.timeframe2
        spread2 = args.spread2
        position2 = args.position2
        min_trade2 = args.min_trade2
        ml2 = args.ml2
        ret_ea2, trades_ea2, parameter_ea2, start_test, end_test = (
            fs.backtest(start, end, optimization, in_sample_period,
            out_of_sample_period, ea2, symbol2, timeframe2, spread2, position2,
            min_trade2, ml2))
        ret += ret_ea2
        trades += trades_ea2
    # EA3のバックテストを行う。
    if args.ea3 is not None:
        ea3 = args.ea3
        symbol3 = args.symbol3
        timeframe3 = args.timeframe3
        spread3 = args.spread3
        position3 = args.position3
        min_trade3 = args.min_trade3
        ml3 = args.ml3
        ret_ea3, trades_ea3, parameter_ea3, start_test, end_test = (
            fs.backtest(start, end, optimization, in_sample_period,
            out_of_sample_period, ea2, symbol2, timeframe2, spread2, position2,
            min_trade2, ml3))
        ret += ret_ea3
        trades += trades_ea3
    # EA4のバックテストを行う。
    if args.ea4 is not None:
        ea4 = args.ea4
        symbol4 = args.symbol4
        timeframe4 = args.timeframe4
        spread4 = args.spread4
        position4 = args.position4
        min_trade4 = args.min_trade4
        ml4 = args.ml4
        ret_ea4, trades_ea4, parameter_ea4,  start_test, end_test = (
            fs.backtest(start, end, optimization, in_sample_period,
            out_of_sample_period, ea2, symbol2, timeframe2, spread2, position2,
            min_trade2, ml4))
        ret += ret_ea4
        trades += trades_ea4
    # EA5のバックテストを行う。
    if args.ea5 is not None:
        ea5 = args.ea5
        symbol5 = args.symbol5
        timeframe5 = args.timeframe5
        spread5 = args.spread5
        position5 = args.position5
        min_trade5 = args.min_trade5
        ml5 = args.ml5
        ret_ea5, trades_ea5, parameter_ea5, start_test, end_test = (
            fs.backtest(start, end, optimization, in_sample_period,
            out_of_sample_period, ea2, symbol2, timeframe2, spread2, position2,
            min_trade2, ml5))
        ret += ret_ea5
        trades += trades_ea5
    ret = ret.fillna(0.0)
    # バックテストの結果を表示する。足の種類は「timeframe1」で代表。
    fs.show_backtest_result(ret, trades, timeframe1, start_test, end_test,
                            parameter_ea1, parameter_ea2, parameter_ea3,
                            parameter_ea4, parameter_ea5)
    # 一時フォルダを削除する。
    fs.remove_temp_folder()