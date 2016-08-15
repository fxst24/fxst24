# coding: utf-8

import argparse
import pandas as pd

from collections import OrderedDict

def make_historical_data(parser):
    '''ヒストリカルデータを作成する。
          Args:
              parser: パーサー。
    '''
    parser.add_argument('--audcad', type=int, default=0)
    parser.add_argument('--audchf', type=int, default=0)
    parser.add_argument('--audjpy', type=int, default=0)
    parser.add_argument('--audnzd', type=int, default=0)
    parser.add_argument('--audusd', type=int, default=0)
    parser.add_argument('--cadchf', type=int, default=0)
    parser.add_argument('--cadjpy', type=int, default=0)
    parser.add_argument('--chfjpy', type=int, default=0)
    parser.add_argument('--euraud', type=int, default=0)
    parser.add_argument('--eurcad', type=int, default=0)
    parser.add_argument('--eurchf', type=int, default=0)
    parser.add_argument('--eurgbp', type=int, default=0)
    parser.add_argument('--eurjpy', type=int, default=0)
    parser.add_argument('--eurnzd', type=int, default=0)
    parser.add_argument('--eurusd', type=int, default=0)
    parser.add_argument('--gbpaud', type=int, default=0)
    parser.add_argument('--gbpcad', type=int, default=0)
    parser.add_argument('--gbpchf', type=int, default=0)
    parser.add_argument('--gbpjpy', type=int, default=0)
    parser.add_argument('--gbpnzd', type=int, default=0)
    parser.add_argument('--gbpusd', type=int, default=0)
    parser.add_argument('--nzdcad', type=int, default=0)
    parser.add_argument('--nzdchf', type=int, default=0)
    parser.add_argument('--nzdjpy', type=int, default=0)
    parser.add_argument('--nzdusd', type=int, default=0)
    parser.add_argument('--usdcad', type=int, default=0)
    parser.add_argument('--usdchf', type=int, default=0)
    parser.add_argument('--usdjpy', type=int, default=0)
    args = parser.parse_args()

    audcad = args.audcad
    audchf = args.audchf
    audjpy = args.audjpy
    audnzd = args.audnzd
    audusd = args.audusd
    cadchf = args.cadchf
    cadjpy = args.cadjpy
    chfjpy = args.chfjpy
    euraud = args.euraud
    eurcad = args.eurcad
    eurchf = args.eurchf
    eurgbp = args.eurgbp
    eurjpy = args.eurjpy
    eurnzd = args.eurnzd
    eurusd = args.eurusd
    gbpaud = args.gbpaud
    gbpcad = args.gbpcad
    gbpchf = args.gbpchf
    gbpjpy = args.gbpjpy
    gbpnzd = args.gbpnzd
    gbpusd = args.gbpusd
    nzdcad = args.nzdcad
    nzdchf = args.nzdchf
    nzdjpy = args.nzdjpy
    nzdusd = args.nzdusd
    usdcad = args.usdcad
    usdchf = args.usdchf
    usdjpy = args.usdjpy

    data = pd.DataFrame()

    # csvファイルを読み込む。
    for i in range(28):
        if i == 0:
            if audcad == 0:
                continue
            symbol = 'AUDCAD'
        elif i == 1:
            if audchf == 0:
                continue
            symbol = 'AUDCHF'
        elif i == 2:
            if audjpy == 0:
                continue
            symbol = 'AUDJPY'
        elif i == 3:
            if audnzd == 0:
                continue
            symbol = 'AUDNZD'
        elif i == 4:
            if audusd == 0:
                continue
            symbol = 'AUDUSD'
        elif i == 5:
            if cadchf == 0:
                continue
            symbol = 'CADCHF'
        elif i == 6:
            if cadjpy == 0:
                continue
            symbol = 'CADJPY'
        elif i == 7:
            if chfjpy == 0:
                continue
            symbol = 'CHFJPY'
        elif i == 8:
            if euraud == 0:
                continue
            symbol = 'EURAUD'
        elif i == 9:
            if eurcad == 0:
                continue
            symbol = 'EURCAD'
        elif i == 10:
            if eurchf == 0:
                continue
            symbol = 'EURCHF'
        elif i == 11:
            if eurgbp == 0:
                continue
            symbol = 'EURGBP'
        elif i == 12:
            if eurjpy == 0:
                continue
            symbol = 'EURJPY'
        elif i == 13:
            if eurnzd == 0:
                continue
            symbol = 'EURNZD'
        elif i == 14:
            if eurusd == 0:
                continue 
            symbol = 'EURUSD'
        elif i == 15:
            if gbpaud == 0:
                continue
            symbol = 'GBPAUD'
        elif i == 16:
            if gbpcad == 0:
                continue
            symbol = 'GBPCAD'
        elif i == 17:
            if gbpchf == 0:
                continue
            symbol = 'GBPCHF'
        elif i == 18:
            if gbpjpy == 0:
                continue
            symbol = 'GBPJPY'
        elif i == 19:
            if gbpnzd == 0:
                continue
            symbol = 'GBPNZD'
        elif i == 20:
            if gbpusd == 0:
                continue
            symbol = 'GBPUSD'
        elif i == 21:
            if nzdcad == 0:
                continue
            symbol = 'NZDCAD'
        elif i == 22:
            if nzdchf == 0:
                continue
            symbol = 'NZDCHF'
        elif i == 23:
            if nzdjpy == 0:
                continue
            symbol = 'NZDJPY'
        elif i == 24:
            if nzdusd == 0:
                continue
            symbol = 'NZDUSD'
        elif i == 25:
            if usdcad == 0:
                continue
            symbol = 'USDCAD'
        elif i == 26:
            if usdchf == 0:
                continue
            symbol = 'USDCHF'
        elif i == 27:
            if usdjpy == 0:
                continue
            symbol = 'USDJPY'
        else:
            pass

        # 選択した通貨ペアの数を数える。
        n = (audcad + audchf + audjpy + audnzd + audusd + cadchf + cadjpy +
             chfjpy + euraud + eurcad + eurchf + eurgbp + eurjpy + eurnzd +
             eurusd + gbpaud + gbpcad + gbpchf + gbpjpy + gbpnzd + gbpusd +
             nzdcad + nzdchf + nzdjpy + nzdusd + usdcad + usdchf + usdjpy)

        # 1分足の作成
        filename = '~/historical_data/' + symbol + '.csv'
        if i == 0:
            data = pd.read_csv(filename, header=None, index_col=0)
            data.index = pd.to_datetime(data.index)
        else:
            temp = pd.read_csv(filename, header=None, index_col=0)
            temp.index = pd.to_datetime(temp.index)
            data = pd.merge(
                data, temp, left_index=True, right_index=True, how='outer')

    # 列名を変更する。
    label = ['open', 'high', 'low', 'close', 'volume']
    data.columns = label * n

    # リサンプリングの方法を設定する。
    ohlc_dict = OrderedDict()
    ohlc_dict['open'] = 'first'
    ohlc_dict['high'] = 'max'
    ohlc_dict['low'] = 'min'
    ohlc_dict['close'] = 'last'
    ohlc_dict['volume'] = 'sum'

    # 各足を作成する。
    count = 0
    for i in range(28):
        if i == 0:
            if audcad == 0:
                continue
            symbol = 'AUDCAD'
            count = count + 1
        elif i == 1:
            if audchf == 0:
                continue
            symbol = 'AUDCHF'
            count = count + 1
        elif i == 2:
            if audjpy == 0:
                continue
            symbol = 'AUDJPY'
            count = count + 1
        elif i == 3:
            if audnzd == 0:
                continue
            symbol = 'AUDNZD'
            count = count + 1
        elif i == 4:
            if audusd == 0:
                continue
            symbol = 'AUDUSD'
            count = count + 1
        elif i == 5:
            if cadchf == 0:
                continue
            symbol = 'CADCHF'
            count = count + 1
        elif i == 6:
            if cadjpy == 0:
                continue
            symbol = 'CADJPY'
            count = count + 1
        elif i == 7:
            if chfjpy == 0:
                continue
            symbol = 'CHFJPY'
            count = count + 1
        elif i == 8:
            if euraud == 0:
                continue
            symbol = 'EURAUD'
            count = count + 1
        elif i == 9:
            if eurcad == 0:
                continue
            symbol = 'EURCAD'
            count = count + 1
        elif i == 10:
            if eurchf == 0:
                continue
            symbol = 'EURCHF'
            count = count + 1
        elif i == 11:
            if eurgbp == 0:
                continue
            symbol = 'EURGBP'
            count = count + 1
        elif i == 12:
            if eurjpy == 0:
                continue
            symbol = 'EURJPY'
            count = count + 1
        elif i == 13:
            if eurnzd == 0:
                continue
            symbol = 'EURNZD'
            count = count + 1
        elif i == 14:
            if eurusd == 0:
                continue 
            symbol = 'EURUSD'
            count = count + 1
        elif i == 15:
            if gbpaud == 0:
                continue
            symbol = 'GBPAUD'
            count = count + 1
        elif i == 16:
            if gbpcad == 0:
                continue
            symbol = 'GBPCAD'
            count = count + 1
        elif i == 17:
            if gbpchf == 0:
                continue
            symbol = 'GBPCHF'
            count = count + 1
        elif i == 18:
            if gbpjpy == 0:
                continue
            symbol = 'GBPJPY'
            count = count + 1
        elif i == 19:
            if gbpnzd == 0:
                continue
            symbol = 'GBPNZD'
            count = count + 1
        elif i == 20:
            if gbpusd == 0:
                continue
            symbol = 'GBPUSD'
            count = count + 1
        elif i == 21:
            if nzdcad == 0:
                continue
            symbol = 'NZDCAD'
            count = count + 1
        elif i == 22:
            if nzdchf == 0:
                continue
            symbol = 'NZDCHF'
            count = count + 1
        elif i == 23:
            if nzdjpy == 0:
                continue
            symbol = 'NZDJPY'
            count = count + 1
        elif i == 24:
            if nzdusd == 0:
                continue
            symbol = 'NZDUSD'
            count = count + 1
        elif i == 25:
            if usdcad == 0:
                continue
            symbol = 'USDCAD'
            count = count + 1
        elif i == 26:
            if usdchf == 0:
                continue
            symbol = 'USDCHF'
            count = count + 1
        elif i == 27:
            if usdjpy == 0:
                continue
            symbol = 'USDJPY'
            count = count + 1
        else:
            pass
 
        data1 = data.iloc[:, 0+(5*(count-1)): 5+(5*(count-1))]

        # 欠損値を補間する。
        data1 = data1.fillna(method='ffill')
        data1 = data1.fillna(method='bfill')

        # 重複行を削除する。
        data1 = data1[~data1.index.duplicated()] 
 
        data5 = data1.resample(
            '5Min', label='left', closed='left').apply(ohlc_dict)
        data15 = data1.resample(
            '15Min', label='left', closed='left').apply(ohlc_dict)
        data30 = data1.resample(
            '30Min', label='left', closed='left').apply(ohlc_dict)
        data60 = data1.resample(
            '60Min', label='left', closed='left').apply(ohlc_dict)
        data240 = data1.resample(
            '240Min', label='left', closed='left').apply(ohlc_dict)
        data1440 = data1.resample(
            '1440Min', label='left', closed='left').apply(ohlc_dict)

        # 欠損値を削除する。
        data5 = data5.dropna()
        data15 = data15.dropna()
        data30 = data30.dropna()
        data60 = data60.dropna()
        data240 = data240.dropna()
        data1440 = data1440.dropna()

        # ファイルを出力する。
        filename1 =  '~/historical_data/' + symbol + '1.csv'
        filename5 =  '~/historical_data/' + symbol + '5.csv'
        filename15 =  '~/historical_data/' + symbol + '15.csv'
        filename30 =  '~/historical_data/' + symbol + '30.csv'
        filename60 =  '~/historical_data/' + symbol + '60.csv'
        filename240 =  '~/historical_data/' + symbol + '240.csv'
        filename1440 =  '~/historical_data/' + symbol + '1440.csv'
        data1.to_csv(filename1)
        data5.to_csv(filename5)
        data15.to_csv(filename15)
        data30.to_csv(filename30)
        data60.to_csv(filename60)
        data240.to_csv(filename240)
        data1440.to_csv(filename1440)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    make_historical_data(parser)