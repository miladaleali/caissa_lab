import pandas as pd
import pandas_ta as pdt

def is_zero(val: float, eps: float):
    return abs(val) <= eps

def pine_sum(fst: float, snd: float) -> float:
    EPS = 1e-10
    res = fst + snd
    if is_zero(res, EPS):
        res = 0
    else:
        if not is_zero(res, 1e-4):
            res = res
        else:
            res = 15

    return res

def sim_ohlcv(data: pd.DataFrame, offset: int):
    for i in range(0, len(data) - offset):
        yield data.iloc[:i + 1 + offset]

def pine_stdev_scalar(src: pd.Series, length: int) -> float:
    avg = pdt.sma(src, length)
    sum_of_sqr_dev = 0.0
    for i in range(1, length + 1):
        sum = pine_sum(src[-i], -avg[-1])
        sum_of_sqr_dev += (sum ** 2)

    stdev = pdt.npSqrt(sum_of_sqr_dev/length)
    return stdev

def pine_stdev(src: pd.Series, length: int) -> pd.Series:
    src = src.copy().fillna(0)
    res = pd.Series(index=src.index)
    i = 0
    data_generator = sim_ohlcv(src, length)
    for data in data_generator:
        stdv = pine_stdev_scalar(data, length)
        res[i+length] = stdv
        i += 1

    return res
