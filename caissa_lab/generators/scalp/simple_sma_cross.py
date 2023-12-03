# %%
import pandas as pd
import pandas_ta as pdt

try:
    from caissa_lab.signalmaker import SignalGeneratorBase
except ModuleNotFoundError:
    import sys
    sys.path.append('/mnt/work/caissa_lab')
    from caissa_lab.signalmaker import SignalGeneratorBase


class SimpleSMACross(SignalGeneratorBase):
    def generate(self) -> None:
        main = self.get_data('main')

        fast = pdt.sma(main.close, self.configs.get('fast', 9))
        slow = pdt.sma(main.close, self.configs.get('slow', 18))
        main['fast_above_slow'] = pdt.above(fast, slow) if self.is_above() else pdt.below(fast, slow)
        main['cross'] = pdt.cross(main.close, slow, above=self.is_above())
        main['signals'] = (
            (main['cross'] == 1)
            & (main['fast_above_slow'] == 1)
        )

        main = main[main['signals']]
        self.signals = main


# %%
from caissa_lab.data import load_from_file, resample_data
from caissa_lab.signalmaker import DataHolder

base = load_from_file('/mnt/work/caissa_lab/BTC-USDT-SWAP_1m_since_2022-01-01_till_2023-11-24.csv')
d5m = resample_data(base, '5m')

datas = [
    DataHolder(name='base', data=base, timeframe='1m', reset_count=1),
    DataHolder(name='main', data=d5m, timeframe='5m', reset_count=1),
]

gen = SimpleSMACross(
    datas,
    init_price_schema='close',
    step=5 * 12,
    signal_type='long', 
    configs={})
gen.run()

# %%
p = gen.evaluate(0.8, 0.5, 5 * 12)


# %%
p = gen.real_performance.copy()
p['pnl'] = p['pnl']
compound_pnl = (1 + p['pnl']/100).prod() - 1

print(f"Compound PNL: {compound_pnl:.4%}")


# %%
# Calculate cumulative returns
cum = (1 + p['pnl'] / 100).cumprod()
cum.loc['2020-01-01'] = 1
cum = cum.sort_index()
cumulative_returns = cum

# Calculate the cumulative maximum
cumulative_max = cumulative_returns.cummax()

# Calculate the drawdown
drawdown = cumulative_returns / cumulative_max - 1

# Find the maximum drawdown
max_drawdown = drawdown.min()

print(f"Max Drawdown: {max_drawdown:.4%}")
# %%
