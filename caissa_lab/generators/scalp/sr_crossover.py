# %%
from typing import Literal

import pandas as pd
import pandas_ta as pdt
from tqdm import tqdm

try:
    from caissa_lab.signalmaker import SignalGeneratorBase
except ModuleNotFoundError:
    import sys
    sys.path.append('/mnt/work/caissa_lab')
    from caissa_lab.signalmaker import SignalGeneratorBase

from caissa_lab.signalmaker import DataHolder
from caissa_lab.srl import SRL

class SRCrossover(SignalGeneratorBase):
    def __init__(self, datas: list[DataHolder], init_price_schema: Literal['close', 'ohlc', 'hl'], step: int, srl: SRL, signal_type: Literal['long', 'short'] = None, configs: dict = None) -> None:
        super().__init__(datas, init_price_schema, step, signal_type, configs)
        self.srl = srl
        self.sl_impact = 0.007

    def generate(self) -> None:
        main_dh = self.get_data_holder('main')
        main = main_dh.data
        tf = main_dh.timeframe
        past_signal_limit = 5
        main['up_trend'] = main.close > pdt.ema(main.close, 200)

        up_trend = main
        for timestamp, data in tqdm(up_trend.iterrows(), total=len(up_trend), unit='candle', desc='Generating signals...'):
            if past_signal_limit != 0:
                past_signal_limit += 1
                if past_signal_limit != 6:
                    main.loc[timestamp, ['signals', 'why']] = [False, 'limit_open_signal']
                    continue
                past_signal_limit = 0
            if not data.up_trend:
                main.loc[timestamp, ['signals', 'why']] = [False, 'not_up_trend']
                continue

            if data.open < data.close:
                main.loc[timestamp, ['signals', 'why']] = [False, 'not_bearish']
                continue

            if (loc := main.index.get_loc(timestamp)) < 30:
                main.loc[timestamp, ['signals', 'why']] = [False, 'limit_60']
                continue

            levels = self.srl.get_level_archive(timestamp, tf)
            levels = levels[levels.quality >= 20]
            if levels.empty:
                main.loc[timestamp, ['signals', 'why']] = [False, 'limit_level_quality']
                continue

            levels['signals'] = (
                (levels.average_price < data.open)
                & (levels.average_price > data.low)
                & (levels.average_price < data.close)
            )
            levels = levels[levels['signals'] == True]
            if levels.empty:
                main.loc[timestamp, ['signals', 'why']] = [False, 'no_signal']
                continue

            levels.sort_values('quality', ascending=False, inplace=True)
            level = levels.iloc[0]
            past_60 = main.iloc[loc-30: loc-1]
            above_one_pct = len(past_60[past_60.close >= (level.average_price * 1.01)]) / 30
            if above_one_pct < 0.5:
                main.loc[timestamp, ['signals', 'why']] = [False, 'limit_above_0.8']
                continue

            iline = level.average_price - level.std_price
            sl_impact = (data.close - iline)/data.close
            if sl_impact > self.sl_impact:
                main.loc[timestamp, ['signals', 'why']] = [False, f'sl__{sl_impact} > {self.sl_impact}']
                continue

            # this is when everything go right.
            past_signal_limit += 1
            main.loc[timestamp, ['signals', 'support', 'sl', 'quality']] = [True, level.average_price, level.average_price - level.std_price, level.quality]
            
        main['signals'] = main.signals.fillna(False)
        self.no_signals = main[main.signals == False]
        main = main[main.signals == True]
        self.signals = main.copy()

# %%
from caissa_lab.srl import WilliamSRL, load_from_file, resample_data

base = load_from_file('/mnt/work/caissa_lab/BTC-USDT-SWAP_1m_since_2022-01-01_till_2023-11-24.csv')
d5m = resample_data(base, '15m')

datas = [
    DataHolder(name='base', data=base, timeframe='1m', reset_count=1),
    DataHolder(name='main', data=d5m, timeframe='15m', reset_count=1),
]


srl = WilliamSRL(mean_look_back_bar=546)
srl.load('/mnt/work/caissa_lab/BTC-USDT-SWAP_srlevels_archive.pkl')

gen = SRCrossover(
    datas=datas,
    init_price_schema='close',
    step=15 * 5,
    srl=srl,
    signal_type='long'
)

# %%
gen.run()
# %%
gen.evaluate(0.5, 0.05, 15 * 5)
gen.performance_report(1)
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