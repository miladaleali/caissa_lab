# %%
import pandas as pd
import pandas_ta as pdt

try:
    from caissa_lab.signalmaker import SignalGeneratorBase
except ModuleNotFoundError:
    import sys
    sys.path.append('/mnt/work/caissa_lab')
    from caissa_lab.signalmaker import SignalGeneratorBase

from caissa_lab.tv_indicators.overlap import vwma

class RsiVwma(SignalGeneratorBase):
    def generate(self) -> None:
        # Data
        main = self.get_data('main')

        # Indicators
        main['slow_sma'] = pdt.sma(main.close, length=self.configs.get('slow_sma_length', 200))
        main['fast_sma'] = pdt.sma(main.close, length=self.configs.get('fast_sma_length', 12))
        main['rsi'] = pdt.rsi(main.close, length=self.configs.get('rsi_length', 9))
        main['rsi_ok'] = pdt.below_value(main.rsi, 50)
        main['vwma_rsi'] = vwma(main.rsi, main.volume, period=self.configs.get('vwma_length', 20))
        main['cross'] = pdt.cross(main['close'], main['fast_sma'])
        main['above_fast'] = pdt.above(main['rsi'], main['vwma_rsi'])
        main['above_slow'] = pdt.above(main['close'], main['slow_sma'])

        # Signals
        main['signals'] = (
            (main['cross'] == 1)
            & (main['above_fast'] == 1)
            & (main['above_slow'] == 1)
            & (main['rsi_ok'] == 1) 
        )
        main = main[main['signals'] == True]
        self.signals = main


# %%
from caissa_lab.data import load_from_file, resample_data
from caissa_lab.signalmaker import DataHolder

base = load_from_file('/mnt/work/caissa_lab/BTC-USDT-SWAP_1m_since_2022-01-01_till_2023-11-24.csv')
d5m = resample_data(base, '1d')

datas = [
    DataHolder(name='base', data=base, timeframe='1m', reset_count=1),
    DataHolder(name='main', data=d5m, timeframe='1h', reset_count=1),
]

gen = RsiVwma(
    datas,
    init_price_schema='close',
    step=30 * 24 * 60,
    signal_type='long', 
    configs={}
)
gen.run()

# %%
gen.step = 60 * 24 * 7
gen.quantify()
gen.make_analysis_quality()
# %%
