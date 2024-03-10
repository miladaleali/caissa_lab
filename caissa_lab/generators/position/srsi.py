# %%
from typing import Literal

import pandas as pd
import numpy as np
import pandas_ta as pdt
from tqdm import tqdm

try:
    from caissa_lab.signalmaker import SignalGeneratorBase, StaticOrders
except ModuleNotFoundError:
    import sys
    sys.path.append('/mnt/work/caissa_lab')
    from caissa_lab.signalmaker import SignalGeneratorBase, StaticOrders


class SRSICrossRSI(SignalGeneratorBase):
    def generate(self) -> None:
        # Data
        main = self.get_data('main')
        upper = self.get_data('upper')

        # Indicators
        srsi = pdt.stochrsi(
            close=main.close,
            length=self.configs.get('stoch_length', 14),
            rsi_length=self.configs.get('rsi_length', 14),
            k=self.configs.get('stoch_k', 1),
            d=self.configs.get('stoch_d', 1),
        )
        srsi.columns = ['k', 'd']
        main = pd.concat([main, srsi], axis=1)
        main['rsi'] = pdt.rsi(
            close=main.close,
            length=self.configs.get('rsi_length', 14)
        )
        main['cross'] = pdt.cross(main['k'], main['rsi'], above=self.is_above())
        upper['trend'] = pdt.ema(upper.close, self.configs.get('trend', 50))

        main = main[main.cross == 1]
        for timestamp, data in tqdm(main.iterrows(), total=len(main), unit='candle', desc='Generating signals...'):
            if self.check_upper_timeframe_trend((timestamp, data), upper):
                main.loc[timestamp, 'signals'] = True
        self.signals = main[main['signals'] == True]

    def evaluate_static(self, tp: float, sl: float) -> pd.DataFrame:
        self.real_performance = pd.DataFrame(
            columns=[
                'pnl',
                'sl',
                'tp',
                'close_timestamp',
                'open_orders',
                'done_orders',
            ]
        )
        long = self.is_above()

        for timestamp, signal in tqdm(self.signals.iterrows(), total=len(self.signals), unit='signal', desc='Running evaluate...'):
            orders = StaticOrders(signal, long, tp, sl)
            orders.make_orders()
            ohlcvs = self.get_ohlcvs_for_simulate(timestamp)
            done_orders = []
            count = 0
            for otime, ohlcv in ohlcvs.iterrows():
                if count != self.step:
                    for order in orders.orders:
                        if orders.remain_portion > 0:
                            if orders.is_order_filled(order, ohlcv, long):
                                done_orders.append(order)
                                orders.order_filled(order)
                        else:
                            break
                    orders.update_orders(done_orders)
                    done_orders = []
                    count += 1
                    if orders.remain_portion <= 0:
                        break
                else:
                    orders.exit_price += ohlcv.close * orders.remain_portion
                    orders.remain_portion = 0
                    break
            if not orders.remain_portion:
                pnl = (orders.exit_price / signal.close) - 1 if self.is_above() else 1 - (orders.exit_price / signal.close)
                pnl = round(pnl * 100, 4)
                self.real_performance.loc[timestamp, :] = np.array([pnl, orders.sl_remain, orders.tp_remain, otime, orders.orders, orders.done_orders], dtype=object)

        self.real_performance = self.real_performance.convert_dtypes()
        return self.real_performance

# %%
from caissa_lab.srl import load_from_file, resample_data
from caissa_lab.signalmaker import DataHolder

base = load_from_file('/mnt/work/caissa_lab/BTC-USDT-SWAP_1m_since_2022-01-01_till_2023-11-24.csv')
main = resample_data(base, '15m')
upper = resample_data(base, '2h')


datas = [
    DataHolder(name='base', data=base, timeframe='1m', reset_count=1),
    DataHolder(name='main', data=main, timeframe='15m', reset_count=1),
    DataHolder(name='upper', data=upper, timeframe='2h', reset_count=1),
]


step = 15 * 5
gen = SRSICrossRSI(
    datas=datas,
    init_price_schema='close',
    step=step,
    signal_type='long',
    configs={'stoch_k': 1, 'stock_d': 1, 'trend': 50, 'rsi_length': 14, 'stoch_length': 14}
)

# %%
gen.generate()
gen.evaluate_static(1, 0.25)
# gen.run()
# %%
gen.run()
gen.evaluate(0.8, 0.05, step)
gen.performance_report(5)
# %%
gen.performance_report(10)
# %%
gen.evaluate(0.7, 0.05, step)
gen.performance_report(5)
# %%
