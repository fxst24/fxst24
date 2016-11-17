# coding: utf-8

import argparse
import forex_system as fs
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

if __name__ == '__main__':
    # 一時フォルダを削除してから作成する。
    fs.remove_temp_folder()
    fs.make_temp_folder()
    # 全体のオプションを設定する。
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeframe', type=int)
    parser.add_argument('--start', type=str)
    parser.add_argument('--end', type=str)
    parser.add_argument('--optimization', default=0, type=int)
    parser.add_argument('--in_sample_period', default=360, type=int)
    parser.add_argument('--out_of_sample_period', default=30, type=int)
    parser.add_argument('--portfolio', default=0, type=int)
    # EA1のオプションを設定する。
    parser.add_argument('--ea1')
    parser.add_argument('--symbol1')
    parser.add_argument('--spread1', type=int)
    parser.add_argument('--position1', default=2, type=int)
    parser.add_argument('--min_trade1', default=260, type=int)
    parser.add_argument('--ml1', default=0, type=int)
    # EA2のオプションを設定する。
    parser.add_argument('--ea2')
    parser.add_argument('--symbol2')
    parser.add_argument('--spread2', type=int)
    parser.add_argument('--position2', default=2, type=int)
    parser.add_argument('--min_trade2', default=260, type=int)
    parser.add_argument('--ml2', default=0, type=int)
    # EA3のオプションを設定する。
    parser.add_argument('--ea3')
    parser.add_argument('--symbol3')
    parser.add_argument('--spread3', type=int)
    parser.add_argument('--position3', default=2, type=int)
    parser.add_argument('--min_trade3', default=260, type=int)
    parser.add_argument('--ml3', default=0, type=int)
    # EA4のオプションを設定する。
    parser.add_argument('--ea4')
    parser.add_argument('--symbol4')
    parser.add_argument('--spread4', type=int)
    parser.add_argument('--position4', default=2, type=int)
    parser.add_argument('--min_trade4', default=260, type=int)
    parser.add_argument('--ml4', default=0, type=int)
    # EA5のオプションを設定する。
    parser.add_argument('--ea5')
    parser.add_argument('--symbol5')
    parser.add_argument('--spread5', type=int)
    parser.add_argument('--position5', default=2, type=int)
    parser.add_argument('--min_trade5', default=260, type=int)
    parser.add_argument('--ml5', default=0, type=int)
    # 全体の設定を格納する。
    args = parser.parse_args()
    timeframe = args.timeframe
    start = datetime.strptime(args.start + ' 00:00', '%Y.%m.%d %H:%M')
    end = datetime.strptime(args.end + ' 23:59', '%Y.%m.%d %H:%M')
    optimization = args.optimization
    in_sample_period = args.in_sample_period
    out_of_sample_period = args.out_of_sample_period
    portfolio = args.portfolio
    # EA1の設定を格納する。
    ea1 = args.ea1
    symbol1 = args.symbol1
    spread1 = args.spread1
    position1 = args.position1
    min_trade1 = args.min_trade1
    ml1 = args.ml1
    exec('import ' + ea1 + ' as ea_file')
    strategy1 = eval('ea_file.strategy')
    parameter1 = eval('ea_file.PARAMETER')
    if optimization != 3:
        rranges1 = eval('ea_file.RRANGES')
    if optimization == 3:
        build_model1 = eval('ea_file.build_model')
    parameter2 = None
    parameter3 = None
    parameter4 = None
    parameter5 = None
    # EA2の設定を格納する。
    if args.ea2 is not None:
        ea2 = args.ea2
        symbol2 = args.symbol2
        spread2 = args.spread2
        position2 = args.position2
        min_trade2 = args.min_trade2
        ml2 = args.ml2
        exec('import ' + ea2 + ' as ea_file')
        strategy2 = eval('ea_file.strategy')
        parameter2 = eval('ea_file.PARAMETER')
        if optimization != 3:
            rranges2 = eval('ea_file.RRANGES')
        if optimization == 3:
            build_model2 = eval('ea_file.build_model')
    # EA3の設定を格納する。
    if args.ea3 is not None:
        ea3 = args.ea3
        symbol3 = args.symbol3
        spread3 = args.spread3
        position3 = args.position3
        min_trade3 = args.min_trade3
        ml3 = args.ml3
        exec('import ' + ea3 + ' as ea_file')
        strategy3 = eval('ea_file.strategy')
        parameter3 = eval('ea_file.PARAMETER')
        if optimization != 3:
            rranges3 = eval('ea_file.RRANGES')
        if optimization == 3:
            build_model3 = eval('ea_file.build_model')
    # EA4の設定を格納する。
    if args.ea4 is not None:
        ea4 = args.ea4
        symbol4 = args.symbol4
        spread4 = args.spread4
        position4 = args.position4
        min_trade4 = args.min_trade4
        ml4 = args.ml4
        exec('import ' + ea4 + ' as ea_file')
        strategy4 = eval('ea_file.strategy')
        parameter4 = eval('ea_file.PARAMETER')
        if optimization != 3:
            rranges4 = eval('ea_file.RRANGES')
        if optimization == 3:
            build_model4 = eval('ea_file.build_model')
    # EA5の設定を格納する。
    if args.ea5 is not None:
        ea5 = args.ea5
        symbol5 = args.symbol5
        spread5 = args.spread5
        position5 = args.position5
        min_trade5 = args.min_trade5
        ml5 = args.ml5    
        exec('import ' + ea5 + ' as ea_file')
        strategy5 = eval('ea_file.strategy')
        parameter5 = eval('ea_file.PARAMETER')
        if optimization != 3:
            rranges5 = eval('ea_file.RRANGES')
        if optimization == 3:
            build_model5 = eval('ea_file.build_model')

    # 最適化なし、または最適化ありのバックテストを行う。
    if optimization == 0 or optimization == 1:
        # レポートを作成する。
        report =  pd.DataFrame(index=range(1), columns=['start', 'end',
                               'trades', 'apr', 'sharpe', 'kelly', 'drawdowns',
                               'durations', 'parameter1', 'parameter2',
                               'parameter3', 'parameter4', 'parameter5',
                               'weights'])
        # EA1のバックテストを行う。
        if optimization == 1:
            parameter1 = fs.optimize_params(rranges1, strategy1, symbol1,
                                            timeframe, start, end, spread1,
                                            position1, min_trade1)
        signal1 = strategy1(parameter1, symbol1, timeframe, position1)
        ret1 = fs.calc_ret(symbol1, timeframe, signal1, spread1, start, end)
        trades1 = fs.calc_trades(signal1, start, end)
        ret = ret1
        trades = trades1
        # EA2のバックテストを行う。
        if args.ea2 is not None:
            if optimization == 1:
                parameter2 = fs.optimize_params(rranges2, strategy2, symbol2,
                                                timeframe, start, end, spread2,
                                                position2, min_trade2)
            signal2 = strategy2(parameter2, symbol2, timeframe, position2)
            ret2 = fs.calc_ret(symbol2, timeframe, signal2, spread2, start,
                               end)
            trades2 = fs.calc_trades(signal2, start, end)
            ret = pd.concat([ret, ret2], axis=1)
            trades += trades2
        # EA3のバックテストを行う。
        if args.ea3 is not None:
            if optimization == 1:
                parameter3 = fs.optimize_params(rranges3, strategy3, symbol3,
                                                timeframe, start, end, spread3,
                                                position3, min_trade3)
            signal3 = strategy3(parameter3, symbol3, timeframe, position3)
            ret3 = fs.calc_ret(symbol3, timeframe, signal3, spread3, start,
                               end)
            trades3 = fs.calc_trades(signal3, start, end)
            ret = pd.concat([ret, ret3], axis=1)
            trades += trades3
        # EA4のバックテストを行う。
        if args.ea4 is not None:
            if optimization == 1:
                parameter4 = fs.optimize_params(rranges4, strategy4, symbol4,
                                                timeframe, start, end, spread4,
                                                position4, min_trade4)
            signal4 = strategy4(parameter4, symbol4, timeframe, position4)
            ret4 = fs.calc_ret(symbol4, timeframe, signal4, spread4, start,
                               end)
            trades4 = fs.calc_trades(signal4, start, end)
            ret = pd.concat([ret, ret4], axis=1)
            trades += trades4
        # EA5のバックテストを行う。
        if args.ea5 is not None:
            if optimization == 1:
                parameter5 = fs.optimize_params(rranges5, strategy5, symbol5,
                                                timeframe, start, end, spread5,
                                                position5, min_trade5)
            signal5 = strategy5(parameter5, symbol5, timeframe, position5)
            ret5 = fs.calc_ret(symbol5, timeframe, signal5, spread5, start,
                               end)
            trades5 = fs.calc_trades(signal5, start, end)
            ret = pd.concat([ret, ret5], axis=1)
            trades += trades5
        ret = ret.fillna(0.0)
        # ウェイトを計算する。
        n_eas = len(ret.T)
        if n_eas > 5:  # EA1のみの場合、行がないので列数が返されることに注意する。
            weights = np.array([1.0])
        else:
            weights = fs.calc_weights(ret, portfolio)
        n = len(weights)
        if n != 1:
            for i in range(n):
                ret.iloc[:, i] = ret.iloc[:, i] * weights[i]
            ret = (ret + 1.0).prod(axis=1) - 1.0
        # 各パフォーマンスを計算する。
        apr = fs.calc_apr(ret, start, end)
        sharpe = fs.calc_sharpe(ret, start, end)
        kelly = fs.calc_kelly(ret)
        drawdowns = fs.calc_drawdowns(ret)
        durations = fs.calc_durations(ret, timeframe)
        # レポートを作成する。
        report.iloc[0, 0] = start.strftime('%Y.%m.%d')
        report.iloc[0, 1] = end.strftime('%Y.%m.%d')
        report.iloc[0, 2] = trades
        report.iloc[0, 3] = "{0:.3f}".format(apr)
        report.iloc[0, 4] = "{0:.3f}".format(sharpe)
        report.iloc[0, 5] = "{0:.3f}".format(kelly)
        report.iloc[0, 6] = "{0:.3f}".format(drawdowns)
        report.iloc[0, 7] = durations
        if parameter1 is not None:
            report.iloc[0, 8] = str(parameter1)
        if parameter2 is not None:
            report.iloc[0, 9] = str(parameter2)
        if parameter3 is not None:
            report.iloc[0, 10] = str(parameter3)
        if parameter4 is not None:
            report.iloc[0, 11] = str(parameter4)
        if parameter5 is not None:
            report.iloc[0, 12] = str(parameter5)
        if weights is not None:
            report.iloc[0, 13] = str(np.round(weights, 3))
        report = report.dropna(axis=1)
        # グラフを作成、出力する。
        cum_ret = (ret + 1.0).cumprod() - 1.0
        ax=plt.subplot()
        ax.set_xticklabels(cum_ret.index, rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.plot(cum_ret)
        plt.title('Backtest')
        plt.xlabel('Date')
        plt.ylabel('Cumulative returns')
        plt.text(0.05, 0.9, 'APR = ' + str(apr), transform=ax.transAxes)
        plt.text(0.05, 0.85, 'Sharpe ratio = ' + str(sharpe),
                 transform=ax.transAxes)
        plt.tight_layout()  # これを入れないとラベルがはみ出る。
        plt.savefig('backtest.png', dpi=150)
        plt.show()
        plt.close()
        # レポートを出力する。
        pd.set_option('display.width', 1000)
        print(report)

    # ウォークフォワードテスト、または機械学習を用いたバックテストを行う。
    elif optimization == 2 or optimization == 3:
        # レポートを作成する。
        report =  pd.DataFrame(index=range(1000), columns=['start_test',
                               'end_test', 'trades', 'apr', 'sharpe', 'kelly',
                               'drawdowns', 'durations', 'parameter1',
                               'parameter2', 'parameter3', 'parameter4',
                               'parameter5', 'weights'])
        end_test = start
        i = 0
        while True:
            start_train = start + timedelta(days=out_of_sample_period*i)
            end_train = (start_train + timedelta(days=in_sample_period)
                - timedelta(minutes=timeframe))
            start_test = end_train + timedelta(minutes=timeframe)
            if i == 0:
                start_all = start_test
            end_test = (start_test + timedelta(days=out_of_sample_period)
                - timedelta(minutes=timeframe))
            if end_test > end:
                end_all = end_test
                break
            # EA1のバックテストを行う。
            if optimization == 2:
                parameter1 = fs.optimize_params(rranges1, strategy1, symbol1,
                                                timeframe, start_train,
                                                end_train, spread1, position1,
                                                min_trade1)
                signal1 = strategy1(parameter1, symbol1, timeframe, position1)
            else:
                model1, pred_train_std1 = build_model1(symbol1, timeframe,
                                                       start_train, end_train)
                signal1 = strategy1(parameter1, symbol1, timeframe, position1,
                                    model1, pred_train_std1)
            if i == 0:
                signal1_all = signal1[start_test:end_test]
            else:
                signal1_all = signal1_all.append(signal1[start_test:end_test])
            ret1 = fs.calc_ret(symbol1, timeframe, signal1, spread1,
                               start_test, end_test)
            trades1 = fs.calc_trades(signal1, start_test, end_test)
            signal = signal1
            ret = ret1
            trades = trades1
            # EA2のバックテストを行う。
            if args.ea2 is not None:
                if optimization == 2:
                    parameter2 = fs.optimize_params(rranges2, strategy2,
                                                    symbol2, timeframe,
                                                    start_train, end_train,
                                                    spread2, position2,
                                                    min_trade2)
                    signal2 = strategy2(parameter2, symbol2, timeframe,
                                        position2)
                else:
                    model2, pred_train_std2 = build_model2(symbol2, timeframe,
                                                           start_train,
                                                           end_train)
                    signal2 = strategy2(parameter2, symbol2, timeframe,
                                        position2, model2, pred_train_std2)
                if i == 0:
                    signal2_all = signal2[start_test:end_test]
                else:
                    signal2_all = signal2_all.append(
                        signal2[start_test:end_test])
                ret2 = fs.calc_ret(symbol2, timeframe, signal2, spread2,
                                   start_test, end_test)
                trades2 = fs.calc_trades(signal2, start_test, end_test)
                signal = signal.append(signal2)
                ret = pd.concat([ret, ret2], axis=1)
                trades += trades2
            # EA3のバックテストを行う。
            if args.ea3 is not None:
                if optimization == 2:
                    parameter3 = fs.optimize_params(rranges3, strategy3,
                                                    symbol3, timeframe,
                                                    start_train, end_train,
                                                    spread3, position3,
                                                    min_trade3)
                    signal3 = strategy3(parameter3, symbol3, timeframe,
                                        position3)
                else:
                    model3, pred_train_std3 = build_model3(symbol3, timeframe,
                                                           start_train,
                                                           end_train)
                    signal3 = strategy3(parameter3, symbol3, timeframe,
                                        position3, model3, pred_train_std3)
                if i == 0:
                    signal3_all = signal3[start_test:end_test]
                else:
                    signal3_all = signal3_all.append(
                        signal3[start_test:end_test])
                ret3 = fs.calc_ret(symbol3, timeframe, signal3, spread3,
                                   start_test, end_test)
                trades3 = fs.calc_trades(signal3, start_test, end_test)
                ret = pd.concat([ret, ret3], axis=1)
                trades += trades3
            # EA4のバックテストを行う。
            if args.ea4 is not None:
                if optimization == 2:
                    parameter4 = fs.optimize_params(rranges4, strategy4,
                                                    symbol4, timeframe,
                                                    start_train, end_train,
                                                    spread4, position4,
                                                    min_trade4)
                    signal4 = strategy4(parameter4, symbol4, timeframe,
                                        position4)
                else:
                    model4, pred_train_std4 = build_model4(symbol4, timeframe,
                                                           start_train,
                                                           end_train)
                    signal4 = strategy4(parameter4, symbol4, timeframe,
                                        position4, model4, pred_train_std4)
                if i == 0:
                    signal4_all = signal4[start_test:end_test]
                else:
                    signal4_all = signal4_all.append(
                        signal4[start_test:end_test])
                ret4 = fs.calc_ret(symbol4, timeframe, signal4, spread4,
                                   start_test, end_test)
                trades4 = fs.calc_trades(signal4, start_test, end_test)
                ret = pd.concat([ret, ret4], axis=1)
                trades += trades4
            # EA5のバックテストを行う。
            if args.ea5 is not None:
                if optimization == 2:
                    parameter5 = fs.optimize_params(rranges5, strategy5,
                                                    symbol5, timeframe,
                                                    start_train, end_train,
                                                    spread5, position5,
                                                    min_trade5)
                    signal5 = strategy5(parameter5, symbol5, timeframe,
                                        position5)
                else:
                    model5, pred_train_std5 = build_model5(symbol5, timeframe,
                                                           start_train,
                                                           end_train)
                    signal5 = strategy5(parameter5, symbol5, timeframe,
                                        position5, model5, pred_train_std5)
                if i == 0:
                    signal5_all = signal5[start_test:end_test]
                else:
                    signal5_all = signal5_all.append(
                        signal5[start_test:end_test])
                ret5 = fs.calc_ret(symbol5, timeframe, signal5, spread5,
                                   start_test, end_test)
                trades5 = fs.calc_trades(signal5, start_test, end_test)
                ret = pd.concat([ret, ret5], axis=1)
                trades += trades5
            ret = ret.fillna(0.0)
            # ウェイトを計算する。
            n_eas = len(ret.T)
            if n_eas > 5:  # EA1のみの場合、行がないので列数が返されることに注意する。
                weights = np.array([1.0])
            else:
                weights = fs.calc_weights(ret, portfolio)
            n = len(weights)
            if n != 1:
                for j in range(n):
                    ret.iloc[:, j] = ret.iloc[:, j] * weights[j]
                ret = (ret + 1.0).prod(axis=1) - 1.0
            # 各パフォーマンスを計算する。
            apr = fs.calc_apr(ret, start_test, end_test)
            sharpe = fs.calc_sharpe(ret, start_test, end_test)
            kelly = fs.calc_kelly(ret)
            drawdowns = fs.calc_drawdowns(ret)
            durations = fs.calc_durations(ret, timeframe)
            # レポートを作成する。
            report.iloc[i, 0] = start_test.strftime('%Y.%m.%d')
            report.iloc[i, 1] = end_test.strftime('%Y.%m.%d')
            report.iloc[i, 2] = trades
            report.iloc[i, 3] = "{0:.3f}".format(apr)
            report.iloc[i, 4] = "{0:.3f}".format(sharpe)
            report.iloc[i, 5] = "{0:.3f}".format(kelly)
            report.iloc[i, 6] = "{0:.3f}".format(drawdowns)
            report.iloc[i, 7] = durations
            if parameter1 is not None:
                report.iloc[i, 8] = str(parameter1)
            if parameter2 is not None:
                report.iloc[i, 9] = str(parameter2)
            if parameter3 is not None:
                report.iloc[i, 10] = str(parameter3)
            if parameter4 is not None:
                report.iloc[i, 11] = str(parameter4)
            if parameter5 is not None:
                report.iloc[i, 12] = str(parameter5)
            if weights is not None:
                report.iloc[i, 13] = str(np.round(weights, 3))
            i += 1
        # 全体のレポートを最後に追加する。
        ret1_all = fs.calc_ret(symbol1, timeframe, signal1, spread1, start_all,
                               end_all)
        ret_all = ret1_all
        if args.ea2 is not None:
            ret2_all = fs.calc_ret(symbol2, timeframe, signal2, spread2,
                                   start_all, end_all)
            ret_all = pd.concat([ret_all, ret2_all], axis=1)
        if args.ea3 is not None:
            ret3_all = fs.calc_ret(symbol3, timeframe, signal3, spread3,
                                   start_all, end_all)
            ret_all = pd.concat([ret_all, ret3_all], axis=1)
        if args.ea4 is not None:
            ret4_all = fs.calc_ret(symbol4, timeframe, signal4, spread4,
                                   start_all, end_all)
            ret_all = pd.concat([ret_all, ret4_all], axis=1)
        if args.ea5 is not None:
            ret5_all = fs.calc_ret(symbol5, timeframe, signal5, spread5,
                                   start_all, end_all)
            ret_all = pd.concat([ret_all, ret5_all], axis=1)
        if n != 1:
            ret_all = (ret_all + 1.0).prod(axis=1) - 1.0
        apr = fs.calc_apr(ret_all, start_all, end_all)
        sharpe = fs.calc_sharpe(ret_all, start_all, end_all)
        kelly = fs.calc_kelly(ret_all)
        drawdowns = fs.calc_drawdowns(ret_all)
        durations = fs.calc_durations(ret_all, timeframe)
        report.iloc[i, 0] = start_all.strftime('%Y.%m.%d')
        report.iloc[i, 1] = end_all.strftime('%Y.%m.%d')
        report.iloc[i, 2] = report.iloc[:, 2].sum()
        report.iloc[i, 3] = "{0:.3f}".format(apr)
        report.iloc[i, 4] = "{0:.3f}".format(sharpe)
        report.iloc[i, 5] = "{0:.3f}".format(kelly)
        report.iloc[i, 6] = "{0:.3f}".format(drawdowns)
        report.iloc[i, 7] = durations
        if parameter1 is not None:
            report.iloc[i, 8] = ''
        if parameter2 is not None:
            report.iloc[i, 9] = ''
        if parameter3 is not None:
            report.iloc[i, 10] = ''
        if parameter4 is not None:
            report.iloc[i, 11] = ''
        if parameter5 is not None:
            report.iloc[i, 12] = ''
        if weights is not None:
            report.iloc[i, 13] = ''
        report = report.iloc[0:i+1, :]
        report = report.dropna(axis=1)
        # グラフを作成、出力する。
        cum_ret = (ret_all + 1.0).cumprod() - 1.0
        ax=plt.subplot()
        ax.set_xticklabels(cum_ret.index, rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.plot(cum_ret)
        plt.title('Backtest')
        plt.xlabel('Date')
        plt.ylabel('Cumulative returns')
        plt.text(0.05, 0.9, 'APR = ' + str(apr), transform=ax.transAxes)
        plt.text(0.05, 0.85, 'Sharpe ratio = ' + str(sharpe),
                 transform=ax.transAxes)
        plt.tight_layout()  # これを入れないとラベルがはみ出る。
        plt.savefig('backtest.png', dpi=150)
        plt.show()
        plt.close()
        # レポートを出力する。
        pd.set_option('display.width', 1000)
        print(report)
    # 一時フォルダを削除する。
    fs.remove_temp_folder()