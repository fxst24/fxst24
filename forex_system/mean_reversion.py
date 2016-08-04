# coding: utf-8
 
import numpy as np
import pandas as pd

# パラメータの設定
PERIOD = 10
ENTRY_THRESHOLD = 1.5
FILTER_THRESHOLD = 1.0
PARAMETER = [PERIOD, ENTRY_THRESHOLD, FILTER_THRESHOLD]

# 最適化の設定
START_PERIOD = 10
END_PERIOD = 50
STEP_PERIOD = 10
START_ENTRY_THRESHOLD = 0.5
END_ENTRY_THRESHOLD = 2.5
STEP_ENTRY_THRESHOLD = 0.5
START_FILTER_THRESHOLD = 0.5
END_FILTER_THRESHOLD = 2.5
STEP_FILTER_THRESHOLD = 0.5
RRANGES = (
    slice(START_PERIOD, END_PERIOD, STEP_PERIOD),
    slice(START_ENTRY_THRESHOLD, END_ENTRY_THRESHOLD, STEP_ENTRY_THRESHOLD),
    slice(START_FILTER_THRESHOLD, END_FILTER_THRESHOLD, STEP_FILTER_THRESHOLD),
)
 
def calc_signal(parameter, fs, symbol, timeframe, position, start=None,
                end=None, spread=0, optimization=0, min_trade=0):
    '''シグナルを計算する。
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
          シグナル。
    '''

    period = int(parameter[0])
    entry_threshold = float(parameter[1])
    filter_threshold = float(parameter[2])

    # 買わない、または売らないゾーンを設ける。
    close1 = fs.i_close(symbol, timeframe, 1)[start:end]
    temp_no_buy_zone = pd.Series(index=close1.index)
    temp_no_sell_zone = pd.Series(index=close1.index)
    if (symbol == 'AUDJPY' or symbol == 'CADJPY' or symbol == 'CHFJPY' or
        symbol == 'EURJPY' or symbol == 'GBPJPY' or symbol == 'NZDJPY' or
        symbol == 'USDJPY'):
        width = 0.1
    else:
        width = 0.001
    for i in range(4):
        shift = int(2 + (1440 / timeframe * i))
        hl_band2 = fs.i_hl_band(symbol, timeframe, int(1440 / timeframe),
                                shift)[start:end]
        temp_no_buy_zone[((close1>= hl_band2['low']) &
            (close1 <= hl_band2['low'] + width))] = 1
        temp_no_sell_zone[(close1 <= hl_band2['high']) &
            (close1 >= hl_band2['high'] - width)] = 1
        temp_no_buy_zone = temp_no_buy_zone.fillna(0)
        temp_no_sell_zone = temp_no_sell_zone.fillna(0)
        if i == 0:
            no_buy_zone = temp_no_buy_zone
            no_sell_zone = temp_no_sell_zone
        else:
            no_buy_zone = no_buy_zone + temp_no_buy_zone
            no_sell_zone = no_sell_zone + temp_no_sell_zone
    no_buy_zone = no_buy_zone.astype(int)
    no_sell_zone = no_sell_zone.astype(int)

    # シグナルを計算する。
    z_score1 = fs.i_z_score(symbol, timeframe, period, 1)[start:end]
    bandwalk1 = fs.i_bandwalk(symbol, timeframe, period, 1)[start:end]
    longs_entry = (((z_score1 <= -entry_threshold) &
        (bandwalk1 <= -filter_threshold) & (no_buy_zone == 0)) * 1)
    longs_exit = (z_score1 >= 0.0) * 1
    shorts_entry = (((z_score1 >= entry_threshold)
        & (bandwalk1 >= filter_threshold) & (no_sell_zone == 0)) * 1)
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
    elif position == 2:
        signal = longs + shorts
    else:
        pass

    signal = signal.fillna(0)
    signal = signal.astype(int)

    return signal