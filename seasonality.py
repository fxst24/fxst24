# coding: utf-8

import numpy as np

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

    base_time, quote_time = fs.divide_symbol_time(symbol)

    # シグナルを計算する。
    close1 = fs.i_close(symbol, timeframe, 1)
    index = close1.index
    is_base_time = fs.is_trading_hours(index, base_time)
    is_quote_time = fs.is_trading_hours(index, quote_time)
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