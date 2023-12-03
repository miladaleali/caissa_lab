import pandas as pd

def vwma(src: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    weighted_sum = (src * volume).rolling(window=period).sum()
    volume_sum = volume.rolling(window=period).sum()
    vwma = weighted_sum / volume_sum
    return vwma