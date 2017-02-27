# coding: utf-8
import argparse
import forex_system as fs
import threading
import time

if __name__ == '__main__':
    # 全体のオプションを設定する。
    parser = argparse.ArgumentParser()
    parser.add_argument('--mail', default=1, type=int)
    parser.add_argument('--mt4', default=1, type=int)
    # EA1のオプションを設定する。
    parser.add_argument('--ea1')
    parser.add_argument('--symbol1')
    parser.add_argument('--timeframe1', type=int)
    parser.add_argument('--position1', default=2, type=int)
    parser.add_argument('--ml1', default=0, type=int)
    parser.add_argument('--start_train1', default=None, type=str)
    parser.add_argument('--end_train1', default=None, type=str)
    # EA2のオプションを設定する。
    parser.add_argument('--ea2')
    parser.add_argument('--symbol2')
    parser.add_argument('--timeframe2', type=int)
    parser.add_argument('--position2', default=2, type=int)
    parser.add_argument('--ml2', default=0, type=int)
    parser.add_argument('--start_train2', default=None, type=str)
    parser.add_argument('--end_train2', default=None, type=str)
    # EA3のオプションを設定する。
    parser.add_argument('--ea3')
    parser.add_argument('--symbol3')
    parser.add_argument('--timeframe3', type=int)
    parser.add_argument('--position3', default=2, type=int)
    parser.add_argument('--ml3', default=0, type=int)
    parser.add_argument('--start_train3', default=None, type=str)
    parser.add_argument('--end_train3', default=None, type=str)
    # EA4のオプションを設定する。
    parser.add_argument('--ea4')
    parser.add_argument('--symbol4')
    parser.add_argument('--timeframe4', type=int)
    parser.add_argument('--position4', default=2, type=int)
    parser.add_argument('--ml4', default=0, type=int)
    parser.add_argument('--start_train4', default=None, type=str)
    parser.add_argument('--end_train4', default=None, type=str)
    # EA5のオプションを設定する。
    parser.add_argument('--ea5')
    parser.add_argument('--symbol5')
    parser.add_argument('--timeframe5', type=int)
    parser.add_argument('--position5', default=2, type=int)
    parser.add_argument('--ml5', default=0, type=int)
    parser.add_argument('--start_train5', default=None, type=str)
    parser.add_argument('--end_train5', default=None, type=str)
    # 全体の設定を格納する。
    args = parser.parse_args()
    mail = args.mail
    mt4 = args.mt4
    wait_time = 1  # 秒単位。
    # トレードを行う。
    ea1 = args.ea1
    symbol1 = args.symbol1
    timeframe1 = args.timeframe1
    position1 = args.position1
    ml1 = args.ml1
    start_train1 = args.start_train1
    end_train1 = args.end_train1
    threading1 = threading.Thread(target=fs.trade, args=(mail, mt4, ea1,
        symbol1, timeframe1, position1, ml1, start_train1, end_train1))
    threading1.start()
    if args.ea2 is not None:
        time.sleep(wait_time) # 念のため、しばらく待ってから開始する。
        ea2 = args.ea2
        symbol2 = args.symbol2
        timeframe2 = args.timeframe2
        position2 = args.position2
        ml2 = args.ml2
        start_train2 = args.start_train2
        end_train2 = args.end_train2
        threading2 = threading.Thread(target=fs.trade, args=(mail, mt4, ea2,
            symbol2, timeframe2, position2, ml2, start_train2, end_train2))
        threading2.start()
    if args.ea3 is not None:
        time.sleep(wait_time) # 念のため、しばらく待ってから開始する。
        ea3 = args.ea3
        symbol3 = args.symbol3
        timeframe3 = args.timeframe3
        position3 = args.position3
        ml3 = args.ml3
        start_train3 = args.start_train3
        end_train3 = args.end_train3
        threading3 = threading.Thread(target=fs.trade, args=(mail, mt4, ea3,
            symbol3, timeframe3, position3, ml3, start_train3, end_train3))
        threading3.start()
    if args.ea4 is not None:
        time.sleep(wait_time) # 念のため、しばらく待ってから開始する。
        ea4 = args.ea4
        symbol4 = args.symbol4
        timeframe4 = args.timeframe4
        position4 = args.position4
        ml4 = args.ml4
        start_train4 = args.start_train4
        end_train4 = args.end_train4
        threading4 = threading.Thread(target=fs.trade, args=(mail, mt4, ea4,
            symbol4, timeframe4, position4, ml4, start_train4, end_train4))
        threading4.start()
    if args.ea5 is not None:
        time.sleep(wait_time) # 念のため、しばらく待ってから開始する。
        ea5 = args.ea5
        symbol5 = args.symbol5
        timeframe5 = args.timeframe5
        position5 = args.position5
        ml5 = args.ml5
        start_train5 = args.start_train5
        end_train5 = args.end_train5
        threading5 = threading.Thread(target=fs.trade, args=(mail, mt4, ea5,
            symbol5, timeframe5, position5, ml5, start_train5, end_train5))
        threading5.start()