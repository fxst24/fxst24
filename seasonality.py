# coding: utf-8

import numpy as np
import os
from datetime import datetime

# 一般設定（ファイル名は31文字以内にすること）
STRATEGY, ext = os.path.splitext(os.path.basename(__file__))  # extは使用しない。
SYMBOL = 'EURUSD'
TIMEFRAME = 30
SPREAD = 5  # 1 = 0.1pips
BASE_TIME = 'ldn'
QUOTE_TIME = 'ny'
POSITION = 2  # 0：買いのみ 1：売りのみ 2：売買両方

# バックテスト設定
START = datetime.strptime('2013.01.01', '%Y.%m.%d')
END = datetime.strptime('2015.12.31', '%Y.%m.%d')

# 最適化設定
WFT = 0  # 0: バックテスト 1: ウォークフォワードテスト（固定） 2: 同（非固定）
OPTIMIZATION = 0 # 0: 最適化なし 1: 最適化あり (注)この戦略では0しか選べない。
MIN_TRADE = 260

# ウォークフォワードテスト設定
IN_SAMPLE_PERIOD = 360 * 1
OUT_OF_SAMPLE_PERIOD = 30 * 1

# トレード設定
LOTS = 0.1  # 1ロット=1万通貨単位

# EA設定
EA = 0  # 0: EAにシグナル送信なし 1: EAにシグナル送信あり

# メール設定
MAIL = 0  # 0: メールにシグナル送信なし 1: メールにシグナル送信あり

def strategy(parameter, fs, symbol, timeframe, position, start=None, end=None,
             spread=0, optimization=0, min_trade=0):
    '''戦略を記述する。
      Args:
          parameter: 最適化したパラメータ。
          fs: ForexSystemクラスのインスタンス。
          symbol: 通貨ペア名。
          timeframe: タイムフレーム。
          position: ポジションの設定。
          start: 開始年月日。
          end: 終了年月日。
          spread: スプレッド。
          optimization: 最適化の設定。
          min_trade: 最低トレード数。
      Returns:
          トレードの場合はシグナル、バックテストの場合はパフォーマンス。
    '''

    # シグナルを計算する。
    close1 = fs.i_close(symbol, timeframe, 1)
    index = close1.index
    is_base_time = fs.is_trading_hours(index, BASE_TIME)
    is_quote_time = fs.is_trading_hours(index, QUOTE_TIME)
    longs_entry = ((is_base_time == False) & (is_quote_time == True)) * 1
    longs_exit = (longs_entry != 1) * 1
    shorts_entry = ((is_base_time == True) & (is_quote_time == False)) * 1
    shorts_exit = (shorts_entry != 1) * 1
    longs = longs_entry.copy()
    longs[longs==0] = np.nan
    longs[longs_exit==1] = 0
    longs = longs.fillna(method='ffill')
    shorts = -shorts_entry.copy()
    shorts[shorts==0] = np.nan
    shorts[shorts_exit==1] = 0
    shorts = shorts.fillna(method='ffill')
    if position == 0:
        signal = longs
    elif position == 1:
        signal = shorts
    else:  # position == 2
        signal = longs + shorts
    signal = signal.fillna(0)
    signal = signal.astype(int)

    # トレードである場合（トレードの場合は環境を指定している）はシグナルを返して終了する。
    if fs.environment is not None: return signal

    # パフォーマンスを計算する。
    ret = fs.calc_ret(symbol, timeframe, signal, spread, start, end)
    trades = fs.calc_trades(signal, start, end)
    sharpe = fs.calc_sharpe(ret, start, end)
    years = (end - start).total_seconds() / 60 / 60 / 24 / 365

    # 1年当たりのトレード数が最低トレード数に満たない場合、適応度を0にする。
    if trades / years >= min_trade:
        fitness = sharpe
    else:
        fitness = 0.0

    # 最適化しない場合、各パフォーマンスを返す。
    if optimization == 0:
        return ret, trades, sharpe
    # 最適化する場合、適応度の符号を逆にして返す（最適化関数が最小値のみ求めるため）。
    else:
        return -fitness