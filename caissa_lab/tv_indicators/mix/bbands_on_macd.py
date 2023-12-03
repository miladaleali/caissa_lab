import pandas as pd
import pandas_ta as pdt

from caissa_lab.tv_indicators.utils import pine_stdev

def bbands_on_macd(
    close: pd.Series,
    rapida: int = 8,
    lenta: int = 26,
    stdv: float = 0.8
) -> pd.DataFrame:
    m_rapida = pdt.ema(close, rapida)
    m_lenta = pdt.ema(close, lenta)

    # Calculate the MACD and its average
    BBMacd = m_rapida - m_lenta
    Avg = pdt.ema(BBMacd, 9)

    # Calculate standard deviation
    SDev = pine_stdev(BBMacd, 9)

    # Calculate the upper and lower bands
    banda_supe = Avg + (stdv * SDev)
    banda_inf = Avg - (stdv * SDev)
    res = pd.DataFrame(
        {
        'bb_macd': BBMacd.values,
        'banda_inf': banda_inf.values,
        'banda_supe': banda_supe.values,
        },
        index=close.index
        )
    return res
