# %%
from typing import Literal

import pandas as pd
import numpy as np
import pandas_ta as pdt
from tqdm import tqdm

try:
    from caissa_lab.signalmaker import SignalGeneratorBase
except ModuleNotFoundError:
    import sys
    sys.path.append('/mnt/work/caissa_lab')
    from caissa_lab.signalmaker import SignalGeneratorBase

from caissa_lab.signalmaker import DataHolder, StaticOrders

class SRBreakout(SignalGeneratorBase):
    def __init__(self, datas: list[DataHolder], init_price_schema: Literal['close', 'ohlc', 'hl'], step: int, srl: SRL, signal_type: Literal['long', 'short'] = None, configs: dict = None) -> None:
        super().__init__(datas, init_price_schema, step, signal_type, configs)
        self.srl = srl
        self.sl_impact = 0.007

    def calculate_slope(self, line: pd.Series, look_back_period: int, signal_timestamp: pd.Timestamp, positive: bool = True) -> float:
        line = line.loc[:signal_timestamp]
        slope, _ = np.polyfit(np.arange(1, look_back_period + 1), line.iloc[-look_back_period:], 1)
        return slope > 0 if positive else slope < 0

    def reset_pullback_setup(self) -> None:
        # PULLBACK SETUPS
        self.pulback_above_fast = False
        self.pulback_below_fast = False
        self.mid_cross_up_slow = False
        self.pulback_price_trigger = None

    def locate_pulback_stage(self, fast: float, low: float, close: float) -> None:
        if fast <= low <= fast * (1.001) and not self.pulback_above_fast:
            self.mid_cross_up_slow = False
            self.pulback_above_fast = True
            self.pulback_price_trigger = close
        elif low < fast:
            self.mid_cross_up_slow = False
            self.pulback_above_fast = False
            self.pulback_below_fast = True
            self.pulback_price_trigger = close

    def generate(self) -> None:
        # Data
        main_dh = self.get_data_holder('main')
        main = main_dh.data
        upper = self.get_data('upper')

        # Indicators
        main['fast'] = fast = pdt.ema(main.close, (_fl := self.configs.get('fast', 12)))
        main['mid'] = mid = pdt.ema(main.close, (_ml := self.configs.get('mid', 36)))
        main['slow'] = pdt.ema(main.close, self.configs.get('slow', 100))
        main['fast_cross_mid'] = pdt.cross(main.fast, main.mid, above=self.is_above())
        main['mid_cross_slow'] = pdt.cross(main.mid, main.slow, above=self.is_above())
        upper['trend'] = pdt.ema(upper.close, self.configs.get('trend', 50))

        is_above = self.is_above()
        potential_signals = main.iloc[2*_ml:][((main.fast_cross_mid == 1) | (main.mid_cross_slow == 1)) & ((main.fast > main.mid) & (main.mid > main.slow))]
        for timestamp, data in tqdm(potential_signals.iterrows(), total=len(potential_signals), unit='candle', desc='Generating signals...'):
            if self.check_upper_timeframe_trend((timestamp, data), upper):
                if data.fast_cross_mid == 1:
                    self.reset_pullback_setup()
                    if (
                        self.calculate_slope(fast, _fl, timestamp, positive=is_above)
                        and self.calculate_slope(mid, _ml, timestamp, positive=is_above)
                    ):
                        potential_signals.loc[timestamp, ['signals', 'signal_type']] = [True, 'fast_cross']
                        continue
                elif not self.mid_cross_up_slow and data.mid_cross_slow == 1:
                    self.reset_pullback_setup()
                    self.mid_cross_up_slow = True
                    self.locate_pulback_stage(data.fast, data.low, data.close)
                elif self.mid_cross_up_slow:
                    self.locate_pulback_stage(data.fast, data.low, data.close)
                elif self.pulback_above_fast:
                    if data.close > self.pullback_price_trigger:
                        self.reset_pullback_setup()
                        if (
                            self.calculate_slope(fast, _fl, timestamp, is_above)
                            and self.calculate_slope(mid, _ml, timestamp, is_above)
                        ):
                            potential_signals.loc[timestamp, ['signals', 'signal_type']] = [True, 'pullback_above_fast']
                            continue
                    self.locate_pulback_stage(data.fast, data.low, data.close)
                    if data.close < data.mid:
                        self.reset_pullback_setup()
                        continue
                elif self.pulback_below_fast:
                    if (
                        data.close > data.fast
                        and data.close > self.pulback_price_trigger
                    ):
                        self.reset_pullback_setup()
                        if (
                            self.calculate_slope(fast, _fl, timestamp, is_above)
                            and self.calculate_slope(mid, _ml, timestamp, is_above)
                        ):
                            potential_signals.loc[timestamp, ['signals', 'signal_type']] = [True, 'pullback_below_fast']
                    if data.close < data.mid:
                        self.reset_pullback_setup()
                        continue
            else:
                self.reset_pullback_setup()
        self.signals = potential_signals[potential_signals.signals == True]

# %%
from caissa_lab.srl import WilliamSRL, load_from_file, resample_data

base = load_from_file('/mnt/work/caissa_lab/BTC-USDT-SWAP_1m_since_2022-01-01_till_2023-11-24.csv')
main = resample_data(base, '15m')
upper = resample_data(base, '2h')


datas = [
    DataHolder(name='base', data=base, timeframe='1m', reset_count=1),
    DataHolder(name='main', data=main, timeframe='15m', reset_count=1),
    DataHolder(name='upper', data=upper, timeframe='2h', reset_count=1),
]


srl = WilliamSRL(mean_look_back_bar=546)
srl.load('/mnt/work/caissa_lab/BTC-USDT-SWAP_srlevels_archive.pkl')

pdt.stochrsi()
gen = SRBreakout(
    datas=datas,
    init_price_schema='close',
    step=15 * 5,
    srl=srl,
    signal_type='long',
    configs={}
)

# %%
gen.run()
# %%
gen.evaluate(0.5, 0.05, 15 * 3)
gen.performance_report(5)
# %%
sigs = gen.signals[gen.signals.up_trend == True]
lsig = len(sigs)
print(f"there is {len(sigs[sigs.signal == False]) / lsig} percent of candle that has no signal.")
print(f"there is {len(sigs[sigs.why == 'no_signal']) / lsig:.4f} candle that has no signal.")
print(f"there is {len(sigs[sigs.why == 'limit_no_signal']) / lsig:.4f} candle that has no signal, because limit_no_signal")
print(f"there is {len(sigs[sigs.why == 'not_up_trend']) / lsig:.4f} candle that has no signal, because not_up_trend")
print(f"there is {len(sigs[sigs.why == 'not_bearish']) / lsig:.4f} candle that has no signal, because not_bearish")
print(f"there is {len(sigs[sigs.why == 'limit_60']) / lsig:.4f} candle that has no signal, because limit_30")


# %%
sigs