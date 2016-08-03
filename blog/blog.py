# coding: utf-8

import os
import shutil
import sys
import time
 
def func(*args):
    '''入力した文字列を関数として実行する。
      Args:
          *args: 可変長引数。
    '''
    exec('from lib import *')
    exec(args[0][1])

if __name__ == '__main__':
    # 開始時間を格納する。
    start_time = time.time()

    # 作業ディレクトリのパスを格納する（このファイルは作業ディレクトリ内にある）。
    wd_path = os.path.dirname(__file__)

    # もし実行開始時に一時フォルダが残っていたら削除する。
    if os.path.exists(wd_path + '/tmp') == True:
        shutil.rmtree(wd_path + '/tmp')

    # 一時フォルダを作成する。
    os.mkdir(wd_path + '/tmp')

    # 引数を格納する。
    args = sys.argv

    # 実行する。
    func(args)

    # 一時フォルダを削除する。
    shutil.rmtree(wd_path + '/tmp')

    # 終了時間を格納する。
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