from typing import Literal
from datetime import timedelta

import pandas as pd
import numpy as np
from pydantic import BaseModel
import pandas_ta as pdt


from caissa_lab.quantifier import QuantifyCalculator

class DataHolder(BaseModel):
    name: str  # base is 1m and each SignalGenerator must have a base in it.
    data: pd.DataFrame
    timeframe: str
    reset_count: int

    class Config:
        arbitrary_types_allowed = True

SignalRow = tuple[pd.DatetimeIndex, pd.Series]

class Order(BaseModel):
    price: float
    amount: float
    type: Literal['tp', 'sl']

    def pnl_impact(self) -> float:
        return self.price * self.amount

class Orders:
    def __init__(
        self,
        signal: pd.Series,
        analysis_quality: pd.Series,
        long: bool,
    ) -> None:
        self.signal = signal
        self.analysis_quality = analysis_quality
        self.long = long
        self.remain_portion = 1
        self.tp_remain = 1
        self.sl_remain = 1
        self.orders: list[Order] = []
        self.done_orders: list[Order] = []
        self.exit_price = 0

    def make_orders(self) -> None:
        up, down = self.analysis_quality.get(['mean_rise', 'rise_std']), self.analysis_quality.get(['mean_fall', 'fall_std'])
        up = up.rename({'mean_rise': 'mean_', 'rise_std': 'std_'})
        down = down.rename({'mean_fall': 'mean_', 'fall_std': 'std_'})
        enter_price = self.signal.close
        if not self.long:
            up, down = down, up

        # SL
        side_adjuster = +1 if self.long else -1
        # self.orders.append(
        #     Order(
        #         price=(enter_price)*(1-(side_adjuster * down.mean_/100)),
        #         amount=0.5,
        #         type='sl'
        #     )
        # )
        # self.orders.append(
        #     Order(
        #         price=(enter_price)*(1-(side_adjuster)*(down.mean_/100 + down.std_/100)),
        #         amount=0.5,
        #         type='sl'
        #     )
        # )

        self.orders.append(
            Order(
                price=(enter_price)*(1-(side_adjuster)*(down.mean_/100 + down.std_/100)),
                amount=1,
                type='sl'
            )
        )

        # self.orders.append(
        #     Order(
        #         price=(enter_price)*0.99,
        #         amount=1,
        #         type='sl'
        #     )
        # )

        # # TP
        # self.orders.append(
        #     Order(
        #         price=(enter_price)*1.02,
        #         amount=1,
        #         type='tp'
        #     )
        # )

        self.orders.append(
            Order(
                price=(enter_price)*(1+(side_adjuster * (up.mean_/100 - (up.std_/200)))),
                amount=0.25,
                type='tp'
            )
        )
        self.orders.append(
            Order(
                price=(enter_price)*(1+(side_adjuster * up.mean_/100)),
                amount=0.5,
                type='tp'
            )
        )
        self.orders.append(
            Order(
                price=(enter_price)*(1+(side_adjuster * (up.mean_/100 + (up.std_/200)))),
                amount=0.25,
                type='tp'
            )
        )

    def update_orders(self, done_orders: list) -> None:
        for order in done_orders:
            self.orders.remove(order)
            self.done_orders.append(order)

    def is_order_filled(self, order: Order, ohlcv: pd.Series, long: bool) -> bool:
        price = order.price
        if (price in np.arange(ohlcv.low, ohlcv.high)) or (price == ohlcv.high):
            return True
        elif order.type == 'sl':
            if long and price > ohlcv.open:
                order.price = ohlcv.open
                return True
            elif not long and price < ohlcv.open:
                order.price = ohlcv.open
                return True
        elif order.type == 'tp':
            if long and price < ohlcv.open:
                order.price = ohlcv.open
                return True
            elif not long and price > ohlcv.open:
                order.price = ohlcv.open
                return True
        return False

    def order_filled(self, order: Order) -> None:
        if order.type == 'tp':
            adjust_amount = order.amount * (self.remain_portion/self.tp_remain)
            self.exit_price += order.price * adjust_amount
            self.tp_remain -= order.amount
        else:
            adjust_amount = order.amount * (self.remain_portion/self.sl_remain)
            self.exit_price += order.price * (adjust_amount)
            self.sl_remain -= order.amount
        
        self.remain_portion -= adjust_amount

class SignalGeneratorBase:
    SIGNAL_TYPE: Literal['long', 'short']

    def __init__(
        self,
        datas: list[DataHolder],
        init_price_schema: Literal['close', 'ohlc', 'hl'],
        step: int,
        signal_type: Literal['long', 'short'] = None,
        configs: dict = None
    ) -> None:
        self.datas: dict[str, DataHolder] = self.safe_load_datas(datas)
        self.init_price_schema: Literal['close', 'ohlc', 'hl'] = init_price_schema
        self.step: int = step
        self.signal_type: Literal['long', 'short'] = signal_type or self.SIGNAL_TYPE
        self.signals: pd.DataFrame = None
        self.qualities: pd.DataFrame = None
        self.analysis_quality: dict = None
        self.configs = configs
        self.main_timeframe_min = self.convert_timeframe_to_min(self.get_data_holder('main').timeframe)

    def convert_timeframe_to_min(self, timeframe: str) -> int:
        m, c = int(timeframe[:-1]), timeframe[-1]
        match c:
            case 'm':
                m *= 1
            case 'h' | 'H':
                m *= 60
            case 'd' | 'D':
                m *= 24 * 60
            case 'w' | 'W':
                m *= 7 * 24 * 60
            case 'M':
                m *= 30 * 24 * 60
        return m

    def check_upper_timeframe_trend(self, signal_row: SignalRow, upper_data: pd.DataFrame) -> bool:
        data = upper_data[upper_data.index <= signal_row[0]]
        if data.index[-1] == signal_row[0]:
            data = data.iloc[:-1]
        return data.trend[-1] < signal_row[1]['close'] if self.signal_type == 'long' else data.trend[-1] > signal_row[1]['close']

    def is_above(self) -> bool:
        return self.signal_type == 'long'

    def safe_load_datas(self, datas: list[DataHolder]) -> dict[str, DataHolder]:
        dct = {data.name: data for data in datas}
        if 'base' not in dct:
            raise ValueError('please give base data for calculating quality phase.')
        return dct

    def get_data_holder(self, name: str) -> DataHolder:
        return self.datas.get(name)
        
    def get_data(self, name: str) -> pd.DataFrame:
        return self.get_data_holder(name).data.copy()

    def generate(self) -> None:
        raise NotImplementedError

    def get_quality_calculator(self) -> QuantifyCalculator:
        q_calc = QuantifyCalculator(
            data=self.get_data('base'),
            init_price_schema=self.init_price_schema,
            step=self.step,
            signal_type=self.signal_type,
            base_timeframe=self.get_data_holder('main').timeframe
        )
        return q_calc

    def quantify(self) -> None:
        q_calc = self.get_quality_calculator()
        self.qualities = self.signals[['open', 'high', 'low', 'close', 'volume']].copy()
        # for signal in self.signals.iterrows():
        #     quality = q_calc.calculate_signal_quality(signal)
        #     self.qualities.loc[signal[0], list(quality.keys())] = quality
        self.qualities.loc[:, ['score', 'rise_mean_price_pct', 'rise_max_price_pct', 'fall_mean_price_pct', 'fall_max_price_pct']] = self.qualities.apply(lambda x: q_calc.calculate_signal_quality(x), axis=1, result_type='expand')
        self.qualities.fillna(0, inplace=True)

    def make_analysis_quality(self) -> dict:
        # q_calc = self.get_quality_calculator()
        # self.analysis_quality = q_calc.calculate_analysis_quality(self.qualities.copy())
        # return self.analysis_quality
        P_VALUES = list(range(5, 100, 5))
        P_VALUES = [_p / 100 for _p in P_VALUES]
        q = self.qualities
        df = pd.DataFrame(columns=['score', 'mean_rise', 'mean_fall', 'max_rise', 'max_fall', 'rise_std', 'fall_std', 'total_signal'], index=P_VALUES)
        is_above = self.is_above()
        for p_value in P_VALUES:
            _q = q[q.score >= p_value]
            base, obase = _q.rise_mean_price_pct, _q.fall_mean_price_pct
            if is_above:
                base, obase = obase, base

            mean = base.mean()
            std = base.std()
            base_outlier_limit = mean + std

            omean = obase.mean()
            ostd = obase.std()
            obase_outlier_limit = omean + ostd

            _q = _q[(base <= base_outlier_limit) & (obase <= obase_outlier_limit)]
            df.loc[p_value] = {
                'score': round(len(_q)/len(q), 4),
                'mean_rise': _q.rise_mean_price_pct.mean(),
                'mean_fall': _q.fall_mean_price_pct.mean(),
                'max_rise': _q.rise_mean_price_pct.max(),
                'max_fall': _q.fall_mean_price_pct.max(),
                'rise_std': _q.rise_mean_price_pct.std(),
                'fall_std': _q.fall_mean_price_pct.std(),
                'total_signal': len(_q)
            }
        df['rr'] = df.mean_rise.divide((2*df.mean_fall + df.fall_std)/2) if self.is_above() else df.mean_fall.divide((2*df.mean_rise + df.rise_std)/2)
        df['expected_rr'] = (df.score * df.mean_rise) / ((1-df.score) * df.mean_fall) if self.is_above() else (df.score * df.mean_fall) / ((1-df.score) * df.mean_rise)
        self.analysis_quality = df
        return self.analysis_quality

    def set_signals(self, main_data: pd.DataFrame, signals: list[bool]) -> pd.DataFrame:
        main_data.loc[main_data.index, 'signals'] = signals
        self.signals = main_data[main_data.signals == True]
        return main_data

    def run(self) -> dict:
        self.generate()
        self.quantify()
        q = self.make_analysis_quality()
        return q

    def recalculate_analysis_quality(self, qualities: pd.DataFrame, raw_signal_len: int):
        base, obase = qualities.rise_mean_price_pct, qualities.fall_mean_price_pct
        if self.is_above():
            base, obase = obase, base
        mean = base.mean()
        std = base.std()
        omean = obase.mean()
        ostd = obase.std()

        qualities = qualities[(base <= (mean + std)) & (obase <= (omean + ostd))]
        return pd.Series(
            {
                'score': round(len(qualities)/raw_signal_len, 4),
                'mean_rise': qualities.rise_mean_price_pct.mean(),
                'mean_fall': qualities.fall_mean_price_pct.mean(),
                'max_rise': qualities.rise_mean_price_pct.max(),
                'max_fall': qualities.fall_mean_price_pct.max(),
                'rise_std': qualities.rise_mean_price_pct.std(),
                'fall_std': qualities.fall_mean_price_pct.std(),
                'total_signal': len(qualities)
            }
        )

    def evaluate(self, portion: float, score: float, step: int):
        q = self.qualities[self.qualities.score >= score].copy()
        separate = int(round(len(q) * portion, 0))

        train = q.iloc[:separate]
        test_start = train.iloc[-1].name

        test: pd.DataFrame = self.signals[self.signals.index > test_start].get(['open', 'high', 'low', 'close', 'volume']).copy()
        raw_signal_len = len(self.signals) - len(test)
        pending_signal_quality: dict[pd.Timestamp, pd.Series] = {}
        done_signal_quality: list[pd.Timestamp] = []
        self.real_performance = pd.DataFrame(
            columns=[
                'pnl',
                'sl',
                'tp',
                'close_timestamp',
                'open_orders',
                'done_orders',
                'analysis_quality'
            ]
        )
        analysis_quality = self.recalculate_analysis_quality(train, raw_signal_len)
        long = self.is_above()
        for timestamp, signal in test.iterrows():

            # update analysis quality
            for ptimestamp, psignal in pending_signal_quality.items():
                if ptimestamp + timedelta(minutes=self.step) <= timestamp:
                    sig_quality = self.get_quality_calculator().calculate_signal_quality(psignal)
                    raw_signal_len += 1
                    if sig_quality['score'] >= score:
                        train.loc[ptimestamp, :] = sig_quality
                    analysis_quality = self.recalculate_analysis_quality(train, raw_signal_len)
                    done_signal_quality.append(ptimestamp)

            # reset done signals
            for done_signal in done_signal_quality:
                pending_signal_quality.pop(done_signal)
            done_signal_quality = []

            if not pending_signal_quality:
                # calculate signal real performance
                orders = Orders(signal=signal, analysis_quality=analysis_quality, long=long)
                orders.make_orders()
                ohlcvs = self.get_ohlcvs_for_simulate(timestamp)
                done_orders = []
                count = 0
                for otime, ohlcv in ohlcvs.iterrows():
                    if count != step:
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
                    pending_signal_quality[timestamp] = signal
                    pnl = (orders.exit_price / signal.close) - 1 if self.is_above() else 1 - (orders.exit_price / signal.close)
                    pnl = round(pnl * 100, 4)
                    self.real_performance.loc[timestamp, :] = np.array([pnl, orders.sl_remain, orders.tp_remain, otime, orders.orders, orders.done_orders, analysis_quality], dtype=object)
        self.real_performance = self.real_performance.convert_dtypes()
        return self.real_performance

    def get_ohlcvs_for_simulate(self, signal_timestamp: pd.Timedelta) -> pd.DataFrame:
        start = signal_timestamp + timedelta(minutes=self.main_timeframe_min)
        return self.get_data('base').loc[start:]

    def performance_report(self, leverage: int) -> None:
        p = self.real_performance.copy()
        p['pnl'] = p['pnl'] * leverage
        compound_pnl = (1 + p['pnl']/100).prod() - 1

        print(f"Compound PNL: {compound_pnl:.4%}")
        
        cum = (1 + p['pnl'] / 100).cumprod()
        cum.loc['2010-01-01'] = 1
        cum = cum.sort_index()
        cumulative_returns = cum

        # Calculate the cumulative maximum
        cumulative_max = cumulative_returns.cummax()

        # Calculate the drawdown
        drawdown = cumulative_returns / cumulative_max - 1

        # Find the maximum drawdown
        max_drawdown = drawdown.min()

        print(f"Max Drawdown: {max_drawdown:.4%}")

