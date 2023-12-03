# %%
import pandas as pd
import pandas_ta as pdt

try:
    from caissa_lab.signalmaker import SignalGeneratorBase
except ModuleNotFoundError:
    import sys
    sys.path.append('/mnt/work/caissa_lab')
    from caissa_lab.signalmaker import SignalGeneratorBase
from caissa_lab.tv_indicators.mix import qqe

class QQEGenerator(SignalGeneratorBase):
    def generate(self) -> None:
        # Data
        main = self.get_data('main')

        # Calculate Indicator
        _qqe = qqe(main.close, length=self.configs.get('length', 14), SSF=self.configs.get('SSF', 5))
        signals = pd.concat([main, _qqe], axis=1)
        signals['trend'] = pdt.ema(signals.close, self.configs.get('trend', 200))
        signals['cross'] = pdt.cross(signals['fast'], signals['slow'], above=self.is_above())
        signals['confirm_trend'] = pdt.above(signals['close'], signals['trend']) if self.is_above() else pdt.below(signals['close'], signals['trend'])

        # Calculate Signal
        signals['signals'] = (
            (signals['cross'] == 1)
            & (signals['confirm_trend'] == 1)
        )
        signals = signals[signals['signals'] == True]
        self.signals = signals

# %%
from caissa_lab.data import load_from_file, resample_data
from caissa_lab.signalmaker import DataHolder

base = load_from_file('/mnt/work/caissa_lab/DOGE-USDT-SWAP_1m_since_2022-01-01_till_2023-11-24.csv')
d5m = resample_data(base, '15m')

datas = [
    DataHolder(name='base', data=base, timeframe='1m', reset_count=1),
    DataHolder(name='main', data=d5m, timeframe='1h', reset_count=1),
]

gen = QQEGenerator(
    datas,
    init_price_schema='close',
    step=5,
    signal_type='long', 
    configs={'trend': 200, 'length': 14, 'SSF': 5}
)
gen.run()

# %%
gen.step = 5
gen.quantify()
gen.make_analysis_quality()
# %%
