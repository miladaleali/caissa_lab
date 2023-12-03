# %%
from datetime import timedelta
from collections import namedtuple

from tqdm import tqdm
import pandas as pd
import numpy as np
from pydantic import BaseModel

from caissa_lab.data import resample_data, load_from_file

class SRLConfig(BaseModel):
    timeframe: str
    calc_params: dict
    srlevel_std_multiplier: float = 1.05
    first_std_srl: float = 0.025

Line = namedtuple('Line', ['timestamp', 'ohlcv'])

class SRLBase:
    def __init__(self, data: pd.DataFrame = None, symbol: str = None, configs: list[SRLConfig] = None, output_path: str = None) -> None:
        self._data = data
        self.symbol = symbol
        self.configs = configs
        self.output_path = output_path
        self._raw_lines = pd.DataFrame(columns=['timestamp', 'index_price', 'srlevel_std_multiplier', 'first_std_srl', 'open_', 'high', 'low', 'close', 'volume'])
        self._lines = pd.DataFrame(columns=['srline_raw_id', 'srlevel_id'])
        self._levels = pd.DataFrame(columns=['average_price', 'std_price', 'quality'])
        self._levels_archive = pd.DataFrame(columns=['levels'])

    def calculate(self, data: pd.DataFrame, configs: dict) -> pd.DataFrame:
        raise NotImplementedError

    def calculate_index_price(self, ohlcv: pd.Series) -> float:
        return ohlcv.drop('volume').mean()

    def add_new_raw_line(self, line: Line, adjust_timestamp: timedelta, srlevel_std_multiplier: float, first_std_srl: float) -> int:
        """return raw_line_id"""
        index_price = self.calculate_index_price(line.ohlcv)
        rid = len(self._raw_lines)
        timestamp = line.timestamp + adjust_timestamp
        self._raw_lines.iloc[rid, ['timestamp', 'index_price', 'srlevel_std_multiplier', 'first_std_srl', 'open_', 'high', 'low', 'close', 'volume']] = [timestamp, index_price, srlevel_std_multiplier, first_std_srl, *line.ohlcv]
        return rid

    def add_new_line(self, raw_line_id: int, level_id: int) -> int:
        """return line_id"""
        lid = len(self._lines)
        self._lines.loc[lid, ['srline_raw_id', 'srlevel_id']] = [raw_line_id, level_id]
        return lid

    def add_new_level(self, average_price: float, std_price: float) -> int:
        """return level_id"""
        lid = len(self._levels)
        self._levels.iloc[lid, ['average_price', 'std_price', 'quality']] = [average_price, std_price, 1]
        return lid

    def update_level(self, level_id: int) -> None:
        lines_id = self._lines[self._lines.srlevel_id == level_id].srline_raw_id.values
        lines = self._raw_lines[self._raw_lines.index.isin(lines_id)]
        self._levels.loc[level_id, ['average_price', 'std_price', 'quality']] = [lines.index_price.mean(), lines.std_price.mean(), len(lines)]

    def locate_level(self, index_price: float, mult: float, band_mult: float) -> int | None:
        """return level_id if any level exist, otherwise return None"""
        df = self._levels.copy()
        if df.empty:
            return

        left = abs(1-mult) * index_price
        right = abs(1+mult) * index_price
        df = df[df.average_price.between(left, right)]
        if df.empty:
            return

        df['upper_band'] = df.average_price + (df.std_price * band_mult)
        df['lower_band'] = df.average_price - (df.std_price * band_mult)
        lower = df['lower_band'] <= index_price
        upper = df['upper_band'] >= index_price 
        df = df[lower & upper]
        df.sort_values(by='quality', ascending=False, inplace=True)
        if df.empty:
            return

        res = df.iloc[0]
        df = df[df.quality == res.quality]
        if df.shape[0] == 1:
            return res.name
        df['std_pct'] = df['average_price'].div(df['std_price'])
        df.sort_values(by='std_pct', ascending=False, inplace=True)
        return df.iloc[0].name

    def timestamp_reality_diff(self, timeframe: str) -> int:
        """return tiemstamp reality difference in minute"""
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

    def run(self) -> None:
        for configs in self.configs:
            data = resample_data(self.data, configs.timeframe)
            raw_lines = self.calculate(data, configs=configs.calc_params)
            adjust_timestamp = self.timestamp_reality_diff(configs.timeframe)
            std_mult = configs.srlevel_std_multiplier
            first_std = configs.first_std_srl
            for raw_line in raw_lines.iterrows():
                raw_line = Line(raw_line)
                self.add_new_raw_line(raw_line, adjust_timestamp, std_mult, first_std)

        for raw_line_id, line in tqdm(self._raw_lines.iterrows(), desc='Calculate SRLevels...', unit='line'):
            level_exist = True
            level_id = self.locate_level(
                index_price=line.index_price,
                mult=(1.1 * (lmult := line.srlevel_std_multiplier)),
                band_mult=lmult
            )
            if not level_id:
                level_id = self.add_new_level(
                    average_price=line.index_price,
                    std_price=line.index_price * line.first_std_srl
                )
                level_exist = False
            self.add_new_line(raw_line_id, level_id)
            if level_exist:
                self.update_level(level_id)
            self.set_level_archive(line.timestamp, self._levels.copy())

        self._levels_archive.to_csv(f'{self.output_path}/{self.symbol}_srlevels_archive.csv')
        self._raw_lines.to_csv(f"{self.output_path}/{self.symbol}_raw_srline.csv")

    def load(self, level_archive_file_path: str) -> None:
        self._levels_archive = pd.read_csv(level_archive_file_path, index_col=0)

    def set_level_archive(self, timestamp: pd.Timestamp, levels: pd.DataFrame) -> None:
        self._levels_archive.loc[timestamp, 'levels'] = [levels]

    def get_level_archive(self, timestamp: pd.Timestamp) -> pd.DataFrame:
        level = self._levels_archive[self._levels_archive.index <= timestamp].iloc[-1].level
        return level

class WilliamSRL(SRLBase):
    def calculate(self, data: pd.DataFrame, configs: dict) -> pd.DataFrame:
        """Indicate bearish and bullish fractal patterns using shifted Series.

        :param df: OHLC data
        :param period: number of lower (or higher) points on each side of a high (or low)
        :return: pd.DataFrame (bearish, bullish) where True marks a fractal pattern
        """
        period = configs['period']
        # default [-2, -1, 1, 2]
        periods = [p for p in range(-period, period + 1) if p != 0]

        highs = [data['high'] > data['high'].shift(p) for p in periods]
        bears = pd.Series(np.logical_and.reduce(highs), index=data.index)

        lows = [data['low'] < data['low'].shift(p) for p in periods]
        bulls = pd.Series(np.logical_and.reduce(lows), index=data.index)

        return data[bears | bulls]

# CONFIGS = {
#     '5m': CTSettings(
#         timeframe='5m',
#         srlevel_std_multiplier=1.05,
#         data_settings=data_settings,
#         first_std_srl=0.02,
#         other=dict(
#             period=20
#         ),
#     ),
#     '15m': CTSettings(
#         timeframe='15m',
#         srlevel_std_multiplier=1.05,
#         data_settings=data_settings,
#         other=dict(
#             period=20
#         ),
#     ),
#     '30m': CTSettings(
#         timeframe='30m',
#         srlevel_std_multiplier=1.05,
#         data_settings=data_settings,
#         other=dict(
#             period=20
#         ),
#     ),
#     '1h': CTSettings(
#         timeframe='1h',
#         srlevel_std_multiplier=1.05,
#         data_settings=data_settings,
#         other=dict(
#             period=15
#         ),
#     ),
#     '2h': CTSettings(
#         timeframe='2h',
#         srlevel_std_multiplier=1.05,
#         data_settings=data_settings,
#         other=dict(
#             period=10
#         ),
#     ),
#     '4h': CTSettings(
#         timeframe='4h',
#         srlevel_std_multiplier=1.05,
#         data_settings=data_settings,
#         other=dict(
#             period=10
#         ),
#     ),
#     '6h': CTSettings(
#         timeframe='6h',
#         srlevel_std_multiplier=1.05,
#         data_settings=data_settings,
#         other=dict(
#             period=10
#         ),
#     ),
#     '12h': CTSettings(
#         timeframe='12h',
#         srlevel_std_multiplier=1.05,
#         data_settings=data_settings,
#         other=dict(
#             period=10
#         ),
#     ),
#     '1d': CTSettings(
#         timeframe='1d',
#         srlevel_std_multiplier=1.05,
#         data_settings=data_settings,
#         other=dict(
#             period=7
#         ),
#     ),
#     '3d': CTSettings(
#         timeframe='3d',
#         srlevel_std_multiplier=1.05,
#         data_settings=data_settings,
#         other=dict(
#             period=5
#         ),
#     ),
#     '1w': CTSettings(
#         timeframe='1w',
#         srlevel_std_multiplier=1.05,
#         data_settings=data_settings,
#         other=dict(
#             period=3
#         ),
#     ),
#     '1M': CTSettings(
#         timeframe='1M',
#         srlevel_std_multiplier=1.05,
#         data_settings=data_settings,
#         other=dict(
#             period=2
#         ),
#     ),
# }
