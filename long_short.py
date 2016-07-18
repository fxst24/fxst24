# coding: utf-8

import numpy as np
import os
from datetime import datetime

# 一般設定（ファイル名は31文字以内にすること）
STRATEGY, ext = os.path.splitext(os.path.basename(__file__))  # extは使用しない。
SYMBOL = 'USDJPY'
BASE = 'usd'
QUOTE = 'jpy'
TIMEFRAME = 5
SPREAD = 4  # 1 = 0.1pips
PERIOD = 20
ENTRY_THRESHOLD = 1.0
FILTER_THRESHOLD = 1.0
POSITION = 2  # 0：買いのみ 1：売りのみ 2：売買両方

# ウェイトの設定
AUD = 0.0
CAD = 0.0
CHF = 0.0
EUR = 1.0
GBP = 0.0
JPY = 1.0
NZD = 0.0
USD = 1.0

# バックテスト設定
START = datetime.strptime('2012.01.01', '%Y.%m.%d')
END = datetime.strptime('2015.12.31', '%Y.%m.%d')

# 最適化設定
WFT = 1  # 0: バックテスト 1: ウォークフォワードテスト（固定） 2: 同（非固定）
OPTIMIZATION = 1 # 0: 最適化なし 1: 最適化あり
MIN_TRADE = 260
START_PERIOD = 10
END_PERIOD = 50
STEP_PERIOD = 10
START_ENTRY_THRESHOLD = 0.5
END_ENTRY_THRESHOLD = 2.5
STEP_ENTRY_THRESHOLD = 0.5
START_FILTER_THRESHOLD = 0.5
END_FILTER_THRESHOLD = 2.5
STEP_FILTER_THRESHOLD = 0.5
PARAMETER = [PERIOD, ENTRY_THRESHOLD, FILTER_THRESHOLD]
RRANGES = (
    slice(START_PERIOD, END_PERIOD, STEP_PERIOD),
    slice(START_ENTRY_THRESHOLD, END_ENTRY_THRESHOLD, STEP_ENTRY_THRESHOLD),
    slice(START_FILTER_THRESHOLD, END_FILTER_THRESHOLD, STEP_FILTER_THRESHOLD),
)

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

    period = int(parameter[0])
    entry_threshold = float(parameter[1])
    filter_threshold = float(parameter[2])

    # シグナルを計算する。
    z_score1 = fs.i_z_score(symbol, timeframe, period, 1)[start:end]
    bandwalk1 = fs.i_bandwalk(symbol, timeframe, period, 1)[start:end]
    ku_z_score1 = (fs.i_ku_z_score(timeframe, period, 1, aud=AUD, cad=CAD,
        chf=CHF, eur=EUR, gbp=GBP, jpy=JPY, nzd=NZD, usd=USD)[start:end])
    ku_bandwalk1 = (fs.i_ku_bandwalk(timeframe, period, 1, aud=AUD, cad=CAD,
        chf=CHF, eur=EUR, gbp=GBP, jpy=JPY, nzd=NZD, usd=USD)[start:end])
    can_buy = ((ku_z_score1[BASE] <= -entry_threshold) &
        (ku_z_score1[QUOTE] >= entry_threshold) &
        (ku_bandwalk1[BASE] <= -filter_threshold) &
        (ku_bandwalk1[QUOTE] >= filter_threshold))
    longs_entry = (((z_score1 <= -entry_threshold) &
        (bandwalk1 <= -filter_threshold) & (can_buy == True)) * 1)
    longs_exit = (z_score1 >= 0.0) * 1
    can_sell = ((ku_z_score1[BASE] >= entry_threshold) &
        (ku_z_score1[QUOTE] <= -entry_threshold) &
        (ku_bandwalk1[BASE] >= filter_threshold) &
        (ku_bandwalk1[QUOTE] <= -filter_threshold))
    shorts_entry = (((z_score1 >= entry_threshold)
        & (bandwalk1 >= filter_threshold) & (can_sell == True)) * 1)
    shorts_exit = (z_score1 <= 0.0) * 1
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