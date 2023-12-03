import pandas as pd
import pandas_ta as pdt

def trend_magic(
    data: pd.DataFrame,
    cci_length: int = 20,
    atr_multiplier: int = 1,
    atr_length: int = 5,
) -> pd.Series:
    atr = pdt.true_range(data.high, data.low, data.close)
    atr = pdt.sma(atr, atr_length)
    cci = pdt.cci(data.close, data.close, data.close, length=cci_length)
    up_t = data.low - (atr * atr_multiplier)
    down_t = data.high + (atr * atr_multiplier)
    last = 0
    magic_trend = pd.Series(index=data.index)
    for i, (_up, _down, _cci) in enumerate(zip(up_t, down_t, cci)):
        if _cci >= 0:
            val = max(last, _up)
        else:
            val = min(last, _down)
        magic_trend.iloc[i] = last = val

    return pd.DataFrame(
        {
            'trend_magic': magic_trend,
            'cci': cci
        },
        index=data.index
    )
