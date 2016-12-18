# coding: utf-8

import argparse
import pandas as pd
from collections import OrderedDict
from datetime import timedelta

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
    parser.add_argument('--start', type=str)
    parser.add_argument('--end', type=str)
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
    start = args.start + ' 00:00'
    end = args.end + ' 23:59'

    index = pd.date_range(start, end, freq='T')
    data = pd.DataFrame(index=index)
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
        temp = pd.read_csv(filename, index_col=0)
        temp.index = pd.to_datetime(temp.index)
        # デューカスコピーのデータはUTC時間のようなので、UTC+2に変更する。
        temp.index = temp.index + timedelta(hours=2)
        data = pd.concat([data, temp], axis=1)
    # 列名を変更する。
    label = ['open', 'high', 'low', 'close', 'volume']
    data.columns = label * n
    # リサンプリングの方法を設定する。
    ohlcv_dict = OrderedDict()
    ohlcv_dict['open'] = 'first'
    ohlcv_dict['high'] = 'max'
    ohlcv_dict['low'] = 'min'
    ohlcv_dict['close'] = 'last'
    ohlcv_dict['volume'] = 'sum'
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
        # 重複行を削除する。
        data1 = data1[~data1.index.duplicated()]
         # 2分、3分、4分、5分、6分、10分、12分、15分、20分、30分、1時間、2時間、3時間、4時間、
         # 6時間、8時間、12時間、日の各足を作成する。
        data2 = data1.resample(
            '2T', label='left', closed='left').apply(ohlcv_dict)
        data3 = data1.resample(
            '3T', label='left', closed='left').apply(ohlcv_dict)
        data4 = data1.resample(
            '4T', label='left', closed='left').apply(ohlcv_dict)
        data5 = data1.resample(
            '5T', label='left', closed='left').apply(ohlcv_dict)
        data6 = data1.resample(
            '6T', label='left', closed='left').apply(ohlcv_dict)
        data10 = data1.resample(
            '10T', label='left', closed='left').apply(ohlcv_dict)
        data12 = data1.resample(
            '12T', label='left', closed='left').apply(ohlcv_dict)
        data15 = data1.resample(
            '15T', label='left', closed='left').apply(ohlcv_dict)
        data20 = data1.resample(
            '20T', label='left', closed='left').apply(ohlcv_dict)
        data30 = data1.resample(
            '30T', label='left', closed='left').apply(ohlcv_dict)
        data60 = data1.resample(
            '60T', label='left', closed='left').apply(ohlcv_dict)
        data120 = data1.resample(
            '120T', label='left', closed='left').apply(ohlcv_dict)
        data180 = data1.resample(
            '180T', label='left', closed='left').apply(ohlcv_dict)
        data240 = data1.resample(
            '240T', label='left', closed='left').apply(ohlcv_dict)
        data360 = data1.resample(
            '360T', label='left', closed='left').apply(ohlcv_dict)
        data480 = data1.resample(
            '480T', label='left', closed='left').apply(ohlcv_dict)
        data720 = data1.resample(
            '720T', label='left', closed='left').apply(ohlcv_dict)
        data1440 = data1.resample(
            '1440T', label='left', closed='left').apply(ohlcv_dict)
        # 土日を削除する。
        data1 = data1[data1.index.dayofweek<5]
        data2 = data2[data2.index.dayofweek<5]
        data3 = data3[data3.index.dayofweek<5]
        data4 = data4[data4.index.dayofweek<5]
        data5 = data5[data5.index.dayofweek<5]
        data6 = data6[data6.index.dayofweek<5]
        data10 = data10[data10.index.dayofweek<5]
        data12 = data12[data12.index.dayofweek<5]
        data15 = data15[data15.index.dayofweek<5]
        data20 = data20[data20.index.dayofweek<5]
        data30 = data30[data30.index.dayofweek<5]
        data60 = data60[data60.index.dayofweek<5]
        data120 = data120[data120.index.dayofweek<5]
        data180 = data180[data180.index.dayofweek<5]
        data240 = data240[data240.index.dayofweek<5]
        data360 = data360[data360.index.dayofweek<5]
        data480 = data480[data480.index.dayofweek<5]
        data720 = data720[data720.index.dayofweek<5]
        data1440 = data1440[data1440.index.dayofweek<5]
        # ファイルを出力する。
        filename1 =  '~/historical_data/' + symbol + '1.csv'
        filename2 =  '~/historical_data/' + symbol + '2.csv'
        filename3 =  '~/historical_data/' + symbol + '3.csv'
        filename4 =  '~/historical_data/' + symbol + '4.csv'
        filename5 =  '~/historical_data/' + symbol + '5.csv'
        filename6 =  '~/historical_data/' + symbol + '6.csv'
        filename10 =  '~/historical_data/' + symbol + '10.csv'
        filename12 =  '~/historical_data/' + symbol + '12.csv'
        filename15 =  '~/historical_data/' + symbol + '15.csv'
        filename20 =  '~/historical_data/' + symbol + '20.csv'
        filename30 =  '~/historical_data/' + symbol + '30.csv'
        filename60 =  '~/historical_data/' + symbol + '60.csv'
        filename120 =  '~/historical_data/' + symbol + '120.csv'
        filename180 =  '~/historical_data/' + symbol + '180.csv'
        filename240 =  '~/historical_data/' + symbol + '240.csv'
        filename360 =  '~/historical_data/' + symbol + '360.csv'
        filename480 =  '~/historical_data/' + symbol + '480.csv'
        filename720 =  '~/historical_data/' + symbol + '720.csv'
        filename1440 =  '~/historical_data/' + symbol + '1440.csv'
        data1.to_csv(filename1)
        data2.to_csv(filename2)
        data3.to_csv(filename3)
        data4.to_csv(filename4)
        data5.to_csv(filename5)
        data6.to_csv(filename6)
        data10.to_csv(filename10)
        data12.to_csv(filename12)
        data15.to_csv(filename15)
        data20.to_csv(filename20)
        data30.to_csv(filename30)
        data60.to_csv(filename60)
        data120.to_csv(filename120)
        data180.to_csv(filename180)
        data240.to_csv(filename240)
        data360.to_csv(filename360)
        data480.to_csv(filename480)
        data720.to_csv(filename720)
        data1440.to_csv(filename1440)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    make_historical_data(parser)