# coding: utf-8

import forex_system as fs
import numpy as np
import pandas as pd
from collections import OrderedDict
from datetime import timedelta

def func():
    usdjpy = fs.i_close('USDJPY', 1, 0)
    start = usdjpy.index[0]
    end = usdjpy.index[len(usdjpy)-1] + timedelta(seconds=59)
    index = pd.date_range(start, end, freq='10S')
    n = len(index)
    rnd = np.random.randn(n) * 0.0001
    randomwalk = rnd.cumsum() + np.log(100)
    randomwalk = np.exp(randomwalk)
    randomwalk = pd.Series(randomwalk, index=index)
    randomwalk1 = randomwalk.resample('T').ohlc()
    volume = pd.DataFrame([6]*len(randomwalk1), index=randomwalk1.index,
                       columns=['volume'])
    randomwalk1 = pd.concat([randomwalk1, volume], axis=1)
    # リサンプリングの方法を設定する。
    ohlcv_dict = OrderedDict()
    ohlcv_dict['open'] = 'first'
    ohlcv_dict['high'] = 'max'
    ohlcv_dict['low'] = 'min'
    ohlcv_dict['close'] = 'last'
    ohlcv_dict['volume'] = 'sum'
    
    randomwalk2 = randomwalk1.resample('2T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk3 = randomwalk1.resample('3T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk4 = randomwalk1.resample('4T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk5 = randomwalk1.resample('5T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk6 = randomwalk1.resample('6T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk10 = randomwalk1.resample('10T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk12 = randomwalk1.resample('12T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk15 = randomwalk1.resample('15T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk20 = randomwalk1.resample('20T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk30 = randomwalk1.resample('30T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk60 = randomwalk1.resample('60T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk120 = randomwalk1.resample('120T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk180 = randomwalk1.resample('180T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk240 = randomwalk1.resample('240T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk360 = randomwalk1.resample('360T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk480 = randomwalk1.resample('480T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk720 = randomwalk1.resample('720T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk1440 = randomwalk1.resample('1440T', label='left',
                                       closed='left').apply(ohlcv_dict)
    randomwalk1 = randomwalk1[randomwalk1.index.dayofweek<5]
    randomwalk2 = randomwalk2[randomwalk2.index.dayofweek<5]
    randomwalk3 = randomwalk3[randomwalk3.index.dayofweek<5]
    randomwalk4 = randomwalk4[randomwalk4.index.dayofweek<5]
    randomwalk5 = randomwalk5[randomwalk5.index.dayofweek<5]
    randomwalk6 = randomwalk6[randomwalk6.index.dayofweek<5]
    randomwalk10 = randomwalk10[randomwalk10.index.dayofweek<5]
    randomwalk12 = randomwalk12[randomwalk12.index.dayofweek<5]
    randomwalk15 = randomwalk15[randomwalk15.index.dayofweek<5]
    randomwalk20 = randomwalk20[randomwalk20.index.dayofweek<5]
    randomwalk30 = randomwalk30[randomwalk30.index.dayofweek<5]
    randomwalk60 = randomwalk60[randomwalk60.index.dayofweek<5]
    randomwalk120 = randomwalk120[randomwalk120.index.dayofweek<5]
    randomwalk180 = randomwalk180[randomwalk180.index.dayofweek<5]
    randomwalk240 = randomwalk240[randomwalk240.index.dayofweek<5]
    randomwalk360 = randomwalk360[randomwalk360.index.dayofweek<5]
    randomwalk480 = randomwalk480[randomwalk480.index.dayofweek<5]
    randomwalk720 = randomwalk720[randomwalk720.index.dayofweek<5]
    randomwalk1440 = randomwalk1440[randomwalk1440.index.dayofweek<5]
    # ファイルを出力する。
    randomwalk1.to_csv('~/historical_data/RANDOM1.csv')
    randomwalk2.to_csv('~/historical_data/RANDOM2.csv')
    randomwalk3.to_csv('~/historical_data/RANDOM3.csv')
    randomwalk4.to_csv('~/historical_data/RANDOM4.csv')
    randomwalk5.to_csv('~/historical_data/RANDOM5.csv')
    randomwalk6.to_csv('~/historical_data/RANDOM6.csv')
    randomwalk10.to_csv('~/historical_data/RANDOM10.csv')
    randomwalk12.to_csv('~/historical_data/RANDOM12.csv')
    randomwalk15.to_csv('~/historical_data/RANDOM15.csv')
    randomwalk20.to_csv('~/historical_data/RANDOM20.csv')
    randomwalk30.to_csv('~/historical_data/RANDOM30.csv')
    randomwalk60.to_csv('~/historical_data/RANDOM60.csv')
    randomwalk120.to_csv('~/historical_data/RANDOM120.csv')
    randomwalk180.to_csv('~/historical_data/RANDOM180.csv')
    randomwalk240.to_csv('~/historical_data/RANDOM240.csv')
    randomwalk360.to_csv('~/historical_data/RANDOM360.csv')
    randomwalk480.to_csv('~/historical_data/RANDOM480.csv')
    randomwalk720.to_csv('~/historical_data/RANDOM720.csv')
    randomwalk1440.to_csv('~/historical_data/RANDOM1440.csv')

if __name__ == '__main__':
    func()