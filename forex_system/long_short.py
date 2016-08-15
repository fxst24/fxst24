# coding: utf-8

import argparse
import forex_system as fs
import numpy as np

# パラメータの設定
PERIOD = 10
ENTRY_THRESHOLD = 1.0
FILTER_THRESHOLD = 0.5

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
PARAMETER = [PERIOD, ENTRY_THRESHOLD, FILTER_THRESHOLD]
RRANGES = (
    slice(START_PERIOD, END_PERIOD, STEP_PERIOD),
    slice(START_ENTRY_THRESHOLD, END_ENTRY_THRESHOLD, STEP_ENTRY_THRESHOLD),
    slice(START_FILTER_THRESHOLD, END_FILTER_THRESHOLD, STEP_FILTER_THRESHOLD),
)

def calc_signal(parameter, symbol, timeframe, start, end, spread, optimization,
                position, min_trade):
    '''シグナルを計算する。
    Args:
        parameter: 最適化したパラメータ。
        symbol: 通貨ペア名。
        timeframe: タイムフレーム。
        start: 開始年月日。
        end: 終了年月日。
        spread: スプレッド。
        optimization: 最適化の設定。
        position: ポジションの設定。
        min_trade: 最低トレード数。
    Returns:
        シグナル。
    '''
    period = int(parameter[0])
    entry_threshold = float(parameter[1])
    filter_threshold = float(parameter[2])

    # Ku-Chartのモデルを設定する。
    base, quote = fs.divide_symbol(symbol)
    eur = 1.0  # デフォルトでモデルに組み込む。
    jpy = 1.0  # 同上。
    usd = 1.0  # 同上。
    if base == 'aud' or quote == 'aud':
        aud = 1.0  # ベース通貨、またはクウォート通貨に含まれる場合のみ組み込む。
    else:
        aud = 0.0
    if base == 'cad' or quote == 'cad':
        cad = 1.0  # 同上。
    else:
        cad = 0.0
    if base == 'chf' or quote == 'chf':
        chf = 1.0  # 同上。
    else:
        chf = 0.0
    if base == 'gbp' or quote == 'gbp':
        gbp = 1.0  # 同上。
    else:
        gbp = 0.0
    if base == 'nzd' or quote == 'nzd':
        nzd = 1.0  # 同上。
    else:
        nzd = 0.0

    # シグナルを計算する。
    z_score1 = fs.i_z_score(symbol, timeframe, period, 1)[start:end]
    bandwalk1 = fs.i_bandwalk(symbol, timeframe, period, 1)[start:end]
    ku_z_score1 = (fs.i_ku_z_score(timeframe, period, 1, aud=aud, cad=cad,
        chf=chf, eur=eur, gbp=gbp, jpy=jpy, nzd=nzd, usd=usd)[start:end])
    stop_hunting_zone = fs.i_stop_hunting_zone(symbol, timeframe,
        int(1440 / timeframe), 1)[start:end]
    can_buy = ((ku_z_score1[base] <= -entry_threshold) &
        (ku_z_score1[quote] >= entry_threshold) &
        (stop_hunting_zone['lower'] == False))
    longs_entry = (((z_score1 <= -entry_threshold) &
        (bandwalk1 <= -filter_threshold) & (can_buy == True)) * 1)
    longs_exit = (z_score1 >= 0.0) * 1
    can_sell = ((ku_z_score1[base] >= entry_threshold) &
        (ku_z_score1[quote] <= -entry_threshold) &
        (stop_hunting_zone['upper'] == False))
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
    elif position == 2:
        signal = longs + shorts
    else:
        pass

    signal = signal.fillna(0)
    signal = signal.astype(int)

    return signal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    fs.forex_system(calc_signal, parser, PARAMETER, RRANGES)