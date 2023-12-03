# %%
import pandas as pd
import pandas_ta as pdt

try:
    from caissa_lab.signalmaker import SignalGeneratorBase
    from caissa_lab.tv_indicators.mix import bbands_on_macd, trend_magic
except ModuleNotFoundError:
    import sys
    sys.path.append('/mnt/work/caissa_lab')
    from caissa_lab.signalmaker import SignalGeneratorBase
    from caissa_lab.tv_indicators.mix import bbands_on_macd


class BollingerBandsOnMacd(SignalGeneratorBase):
    def generate(self) -> None:
        # Get Data
        main = self.get_data('main')

        # Calculate Indicators
        bb = bbands_on_macd(main.close, rapida=self.configs.get('rapida', 8), lenta=self.configs.get('lenta', 26), stdv=self.configs.get('stdv', 0.8))
        trend = pdt.ema(main.close, length=self.configs.get('trend', 200))

        # Calculate Signals
        signals = pd.concat([main, bb], axis=1)
        signals['trend'] = trend

        # Long
        if self.is_above():
            signals['macd_reverse'] = pdt.cross_value(signals['bb_macd'], 0, above=True)
            signals['macd_cross'] = pdt.cross(signals['bb_macd'], signals['banda_supe'], above=True)
            signals['signals'] = (
                (signals['close'] > signals['trend'])
                & ((signals['macd_reverse'] == 1) | (signals['macd_cross'] == 1))
                & (signals['bb_macd'] > signals['banda_supe'])
            )
        # Short
        else:
            signals['macd_cross'] = pdt.cross(signals['bb_macd'], signals['banda_inf'], above=False)
            signals['signals'] = (
                (signals['close'] < signals['trend'])
                & (signals['macd_cross'] == 1)
                & (signals['bb_macd'] > 0)
            )

        df = signals
        signals = signals[signals['signals'] == True]
        self.signals = signals
        return df

class BBandsOnMacdMagicTrend(SignalGeneratorBase):
    def generate(self) -> None:
        main = self.get_data('main')

        bb_macd = bbands_on_macd(main.close, rapida=self.configs.get('rapida', 8), lenta=self.configs.get('lenta', 26), stdv=self.configs.get('stdv', 0.8))
        t_magic = trend_magic(
            main,
            cci_length=self.configs.get('cci_length', 20),
            atr_multiplier=self.configs.get('atr_multiplier', 1),
            atr_length=self.configs.get('atr_length', 5)
        )
        signals = pd.concat([main, bb_macd, t_magic], axis=1)
        signals.loc[:, 'trend'] = pdt.ema(signals['close'], 200)
        signals.loc[:, 'is_on_trend'] = pdt.above(signals['close'], signals['trend']) if self.is_above() else pdt.below(signals['close'], signals['trend'])
        signals.loc[:, 'cross_bb'] = pdt.cross(signals['bb_macd'], signals['banda_supe'])
        signals.loc[:, 'cross_tm'] = pdt.cross(signals['close'], signals['trend_magic'])
        signals.loc[:, 'tm_is_long'] = pdt.above_value(signals['cci'], 0)
        signals.loc[:, 'bb_is_long'] = pdt.above(signals['bb_macd'], signals['banda_supe'])
        signals['signals'] = (
            (
                (signals['cross_bb'] == 1)
                & (signals['tm_is_long'] == 1)
                & (signals['is_on_trend'] == 1)
            )
            | (
                (signals['cross_tm'] == 1)
                & (signals['bb_is_long'] == 1)
                & (signals['is_on_trend'] == 1)
            )
        )
        signals = signals[signals['signals'] == True]
        self.signals = signals



# %%
from caissa_lab.data import load_from_file, resample_data
from caissa_lab.signalmaker import DataHolder

base = load_from_file('/mnt/work/caissa_lab/BTC-USDT-SWAP_1m_since_2022-01-01_till_2023-11-24.csv')
d5m = resample_data(base, '15m')

datas = [
    DataHolder(name='base', data=base, timeframe='1m', reset_count=1),
    DataHolder(name='main', data=d5m, timeframe='15m', reset_count=1),
]

step = 15 * 4 * 4
gen = BollingerBandsOnMacd(
    datas,
    init_price_schema='close',
    step=step,
    signal_type='long',
    configs={})
gen.run()

# %%
gen.evaluate(0.8, 0.1, step)
gen.performance_report(1)
# %%

gen.performance_report(5)
# %%
