# coding: utf-8

import argparse
import forex_system as fs
import threading

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ea1', nargs='*')
    parser.add_argument('--ea2', nargs='*')
    parser.add_argument('--ea3', nargs='*')
    args = parser.parse_args()

    threading1 = threading.Thread(target=fs.trade, args=args.ea1)
    threading1.start()
    if args.ea2 is not None:
        threading2 = threading.Thread(target=fs.trade, args=args.ea2)
        threading2.start()
    if args.ea3 is not None:
        threading3 = threading.Thread(target=fs.trade, args=args.ea3)
        threading3.start()    