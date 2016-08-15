# coding: utf-8

import argparse
import pandas as pd
import struct
import time

def convert_hst2csv(parser):
    '''hstファイルをcsvファイルに変換する。
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

        # 端末のカレントディレクトリが「~」で、そこからの相対パスであることに注意する。
        filename_hst = '../historical_data/' + symbol + '.hst'
        # 同上
        filename_csv = '../historical_data/' + symbol + '.csv'
        read = 0
        datetime = []
        open_price = []
        low_price = []
        high_price = []
        close_price = []
        volume = []
        with open(filename_hst, 'rb') as f:
            while True:
                if read >= 148:
                    buf = f.read(44)
                    read += 44
                    if not buf:
                        break
                    bar = struct.unpack('< iddddd', buf)
                    datetime.append(
                        time.strftime('%Y-%m-%d %H:%M:%S',
                        time.gmtime(bar[0])))
                    open_price.append(bar[1])
                    high_price.append(bar[3])  # 高値と安値の順序が違うようだ
                    low_price.append(bar[2])  # 同上
                    close_price.append(bar[4])
                    volume.append(bar[5])
                else:
                    buf = f.read(148)
                    read += 148
        data = {'0_datetime':datetime, '1_open_price':open_price,
            '2_high_price':high_price, '3_low_price':low_price,
            '4_close_price':close_price, '5_volume':volume}
        result = pd.DataFrame.from_dict(data)
        result = result.set_index('0_datetime')
        result.to_csv(filename_csv, header = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    convert_hst2csv(parser)