import pandas as pd
import pandas_ta as pdt

def qqe(
    close: pd.DataFrame,
    length: int = 14,
    SSF: int = 5
) -> pd.DataFrame:
    # Input variables
    src = close

    # Calculate RSII
    rsi_value = pdt.rsi(src, length)
    ema_rsi = pdt.ema(rsi_value, SSF)
    RSII = ema_rsi

    # Calculate TR
    TR = RSII.diff(1).fillna(0).abs()

    # Calculate WWMA
    wwalpha = 1 / length
    last = 0
    WWMA = pd.Series(index=TR.index)
    for i, _tr in enumerate(TR):
        WWMA.iloc[i] = last = (wwalpha * _tr) + ((1 - wwalpha) * last)

    # Calculate ATRRSI
    last = 0
    ATRRSI = pd.Series(index=WWMA.index)
    for i, _wwma in enumerate(WWMA):
        ATRRSI.iloc[i] = last = (wwalpha * _wwma) + ((1 - wwalpha) * last)

    # Calculate QQEF
    QQEF = pdt.ema(rsi_value, SSF).fillna(0)

    # Calculate QUP and QDN
    QUP = QQEF + (ATRRSI * 4.236)
    QDN = QQEF - (ATRRSI * 4.236)

    # Calculate QQES
    # QQES = 0.0
    # QQES = QUP if QUP < nz(QQES[1]) else (QDN if QQEF > nz(QQES[1]) and QQEF[1] < nz(QQES[1]) else
    #         (QDN if QDN > nz(QQES[1]) else (QUP if QQEF < nz(QQES[1]) and QQEF[1] > nz(QQES[1]) else nz(QQES[1]))))

    QQES = pd.Series(index=QUP.index)
    lqqes = 0
    lqqef = 0
    for i, (qup, qdn, qqef) in enumerate(zip(QUP, QDN, QQEF)):
        if qup < lqqes:
            res = qup
        elif (qqef > lqqes) and (lqqef < lqqes):
            res = qdn
        elif qdn > lqqes:
            res = qdn
        elif (qqef < lqqes) and (lqqef > lqqes):
            res = qup
        else:
            res = lqqes
        QQES.iloc[i] = res
        lqqes = res
        lqqef = qqef

    return pd.DataFrame(
        {
            'fast': QQEF,
            'slow': QQES,
        },
        index=src.index
    )
