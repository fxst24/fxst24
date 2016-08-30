# coding: utf-8

import argparse
import forex_system as fs
import threading
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ea1', nargs='*')
    parser.add_argument('--ea2', nargs='*')
    parser.add_argument('--ea3', nargs='*')
    parser.add_argument('--ea4', nargs='*')
    parser.add_argument('--ea5', nargs='*')
    args = parser.parse_args()

    wait_time = 1  # 秒単位。

    threading1 = threading.Thread(target=fs.trade, args=args.ea1)
    threading1.start()
    if args.ea2 is not None:
        time.sleep(wait_time) # 念のため、しばらく待ってから開始する。
        threading2 = threading.Thread(target=fs.trade, args=args.ea2)
        threading2.start()
    if args.ea3 is not None:
        time.sleep(wait_time) # 念のため、しばらく待ってから開始する。
        threading3 = threading.Thread(target=fs.trade, args=args.ea3)
        threading3.start()
    if args.ea4 is not None:
        time.sleep(wait_time) # 念のため、しばらく待ってから開始する。
        threading4 = threading.Thread(target=fs.trade, args=args.ea4)
        threading4.start()
    if args.ea5 is not None:
        time.sleep(wait_time) # 念のため、しばらく待ってから開始する。
        threading5 = threading.Thread(target=fs.trade, args=args.ea5)
        threading5.start()