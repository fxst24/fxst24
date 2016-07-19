# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Spyderのバグ（？）で警告が出るので無視する。
import warnings
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)

def create_time_series_data(a, t):
    e = np.random.randn(t)
    y = np.empty(t)
    y[0] = 0.0
    for i in (range(1, t)):
        y[i] = a * y[i - 1] +  e[i]
    return y

if __name__ == '__main__':
    data = create_time_series_data(1.0, 100000)
    data = pd.Series(data)
    data.plot()
    plt.show()