from datetime import timedelta
from typing import Literal

import pandas as pd
from pydantic import BaseModel

class QualityBaseModel(BaseModel):
    mean: float
    max: float
    std: float = None

    def get_quality_dict(self, mode: Literal['rise', 'fall']) -> dict:
        dct = self.dict()
        return {f"{mode}_{k}_price_pct": round(v, 4) for k, v in dct.items() if v is not None}

class QuantifyCalculator:
    '''this class is just a quantify calculator, it doesn't have any logic'''

    def __init__(
        self,
        data: pd.DataFrame,
        init_price_schema: Literal['close', 'ohlc', 'hl'],
        step: int,
        signal_type: Literal['long', 'short'],
        base_timeframe: str
    ):
        self.uid = 'quantify_calculator'
        self.data = data
        self.init_price_schema = init_price_schema
        self.step = step
        self.signal_type = signal_type
        self.base_timeframe = base_timeframe

    def get_safe_data(self) -> pd.DataFrame:
        return self.data.copy()

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

    def get_stamp(
        self,
        signal_timestamp: pd.Timestamp,
        step: int,
    ) -> pd.DataFrame:
        data = self.get_safe_data()
        base = self.convert_timeframe_to_min(self.base_timeframe)
        end_timestamp = signal_timestamp + timedelta(minutes=step + base)
        start = signal_timestamp + timedelta(minutes=base)
        return data[(data.index >= start) & (data.index < end_timestamp)]

    def make_init_price(self, signal: pd.Series, init_price_schema: Literal['close', 'ohlc', 'hl']) -> float:
        match init_price_schema:
            case 'close':
                return signal['close']
            case 'hl':
                return signal.drop({'open', 'close', 'volume'}).mean()
            case 'ohlc':
                return signal.drop('volume').mean()

    def calculate_signal_quality(
        self,
        signal:  pd.Series,
    ) -> dict:
        # NOTE base candle is calculated base on stamp_step interval not generated signal interval
        stamps = self.get_stamp(signal_timestamp=signal.name, step=self.step)
        base = self.make_init_price(signal, init_price_schema=self.init_price_schema)
        up = True if self.signal_type == 'long' else False
        score = self.calculate_signal_score(base, stamps, up)
        rising = self.calculate_signal_rising_price_params(base, stamps.high)
        falling = self.calculate_signal_falling_price_params(base, stamps.low)

        return dict(
            score=score,
            **rising,
            **falling
        )

    def calculate_signal_score(self, base: float, stamps: pd.DataFrame, up: bool) -> float:
        lstamps = len(stamps)
        score = len(stamps[stamps.close > base] if up else stamps[stamps.close < base])
        return round(score / lstamps, 4)

    def _calculate_signal_rise_fall_params(self, base: float, stamps: pd.Series, which: str) -> dict:
        _stamps = stamps > base if which == 'rise' else stamps < base
        stamps = stamps[_stamps]

        if which == 'rise':
            max_ = 100 * abs(stamps.max() - base) / base
        else:
            max_ = 100 * abs(stamps.min() - base) / base
        mean_ = 100 * abs(stamps.mean() - base) / base
        return QualityBaseModel(max=max_, mean=mean_).get_quality_dict(which)

    def calculate_signal_rising_price_params(self, base: float, stamps: pd.Series) -> dict:
        return self._calculate_signal_rise_fall_params(base, stamps, 'rise')

    def calculate_signal_falling_price_params(self, base: float, stamps: pd.Series) -> dict:
        return self._calculate_signal_rise_fall_params(base, stamps, 'fall')

    def calculate_analysis_quality(self, signal_quality_df: pd.DataFrame) -> dict:
        score = round(signal_quality_df.score.mean(), 4)
        risk = round(signal_quality_df.score.std(), 4)
        rising = self.calculate_analysis_rise_fall_params(signal_quality_df, 'rise')
        falling = self.calculate_analysis_rise_fall_params(signal_quality_df, 'fall')
        timestamp = signal_quality_df.index.max()
        return dict(
            score=score,
            timestamp=timestamp, 
            risk=risk,
            **rising,
            **falling
        )

    def _analysis_dataframe_column(self, which: str) -> dict:
        dct = dict(mean_price_pct='mean_', max_price_pct='max_')
        return dict(map(lambda x: (f"{which}_{x[0]}", x[1]), dct.items()))

    def _prepare_analysis_rise_fall_dataframe(self, signals: pd.DataFrame, which: str) -> pd.DataFrame:
        col = self._analysis_dataframe_column(which)
        sigs = signals.rename(col, axis=1)
        return sigs.get(col.values())

    def calculate_analysis_rise_fall_params(self, signals: pd.DataFrame, which: str) -> dict:
        signals = self._prepare_analysis_rise_fall_dataframe(signals, which)
        return QualityBaseModel(
            max=signals.max_.max(),
            mean=signals.mean_.mean(),
            std=signals.mean_.std(),
        ).get_quality_dict(which)
