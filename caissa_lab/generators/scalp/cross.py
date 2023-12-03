# %%
import pandas as pd
import pandas_ta as pdt

try:
    from caissa_lab.signalmaker import SignalGeneratorBase, SignalRow
except ModuleNotFoundError:
    import sys
    sys.path.append('/mnt/work/caissa_lab')
    from caissa_lab.signalmaker import SignalGeneratorBase, SignalRow


class DoubleTimeFrameEMA(SignalGeneratorBase):
    """
    Double timeframe generator that in lower tm check whether ema cross happen and in upper tm
    check trend of asset.
    """
    EMA_FAST: int
    EMA_SLOW: int
    EMA_TREND: int

    def generate(self) -> None:
        # Get Data
        df = self.get_data('main')
        upper_data = self.get_data('upper')

        # Calculate Indicators
        ema_f = pdt.ema(df.close, self.EMA_FAST)
        ema_s = pdt.ema(df.close, self.EMA_SLOW)
        df['cross'] = pdt.cross(ema_f, ema_s, above=self.signal_type == 'long')
        df = df[df['cross'] == 1]
        ema_t = pd.DataFrame({'trend': pdt.ema(upper_data.close, self.EMA_TREND).values}, index=upper_data.index)

        # Calculate Signals
        signals = []
        for row in df.iterrows():
            signals.append(self.check_upper_timeframe_trend(row, ema_t))
        self.set_signals(df, signals)

class DoubleTimeFrameTEMA(SignalGeneratorBase):
    EMA_FAST: int
    EMA_MID: int
    EMA_SLOW: int
    EMA_TREND: int

    def is_cross_happen(
        self,
        close: pd.Series,
        fast: pd.Series,
        mid: pd.Series,
        above: bool = True
    ) -> pd.Series:
        cross1 = pdt.cross(close, fast, above)
        cross2 = pdt.cross(fast, mid, above)
        cross3 = cross1 | cross2
        return cross3

    def generate(self) -> None:
        # Get Data
        df = self.get_data('main')
        upper_data = self.get_data('upper')

        # Calculate Indicators
        ef = pdt.ema(df.close, self.EMA_FAST)
        em = pdt.ema(df.close, self.EMA_MID)
        es = pdt.ema(df.close, self.EMA_SLOW)
        upper_data['trend'] = pdt.ema(upper_data.close, self.EMA_TREND)

        # Calculate Base Signals
        is_above = self.is_above()
        df['fast'] = ef
        df['mid'] = em
        df['slow'] = es
        df['cross'] = self.is_cross_happen(df.close, ef, em, above=is_above)
        df = df[df['cross'] == 1]
        if is_above:
            df['lower_trend'] = (df.close > df.fast) & (df.fast > df.mid) & (df.mid > df.slow)
        else:
            df['lower_trend'] = (df.close < df.fast) & (df.fast < df.mid) & (df.mid < df.slow)

        df = df[df['lower_trend'] == True]

        signals = []
        for signal in df.iterrows():
            signals.append(self.check_upper_timeframe_trend(signal, upper_data))
        self.set_signals(df, signals)

class BBCross(SignalGeneratorBase):
    
    def generate(self) -> None:
        # Get Data
        df = self.get_data('main')

        # Calculate Indicators
        bb = pdt.bbands(
            close=df.close,
            length=self.configs['length'],
            std=self.configs['std'],
            mamode=self.configs['mamode']
        )
        bb.columns = ['lower', 'mid', 'upper', 'bandwidth', 'close_percent']
        df = pd.concat([df, bb], axis=1)

        # Calculate Base Signals
        df['cross'] = pdt.cross(df.close, df.mid, above=self.is_above())
        df = df[df['cross'] == 1]

        threshold = self.configs['threshold']
        df['signals'] = (df['bandwidth'] <= threshold)

        df = df[df['signals'] == True]
        self.signals = df




# %%

from caissa_lab.data import load_from_file, resample_data
from caissa_lab.signalmaker import DataHolder

base = load_from_file('/mnt/work/caissa_lab/BTC-USDT-SWAP_1m_since_2022-01-01_till_2023-11-24.csv')
d5m = resample_data(base, '5m')

datas = [
    DataHolder(name='base', data=base, timeframe='1m', reset_count=1),
    DataHolder(name='main', data=d5m, timeframe='15m', reset_count=1),
]

gen = BBCross(datas, init_price_schema='close', step=5 * 10, signal_type='long', configs={'length': 20, 'std': 2, 'mamode': 'sma', 'threshold': 0.3})
# gen.run()

# %%
gen.step = 15 * 20
gen.quantify()
gen.make_analysis_quality()
# %%
df = resample_data(base, '1d')
