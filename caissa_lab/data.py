# %%
from typing import Literal
from datetime import datetime, date
import asyncio
import os

import pandas as pd
from ccxt.async_support import okx, Exchange

class DataNotFetched(Exception):
    pass

def get_data_provider() -> Exchange:
    configs = {}
    if os.uname()[1] == 'her':
        configs={
            'aiohttp_trust_env': {
                'HTTP_PROXY': 'http://127.0.0.1:10809',
                'HTTPS_PROXY': 'https://127.0.0.1:10809'
            }
        }
    return okx(config=configs)

def chunk_fetch_period(
    start: int,
    end: int,
    limit: int = 100
) -> list[int]:
    """
    return only since timestamp
    """
    step = limit * 60 * 1000
    tmp_end = start + step
    periods = []
    while tmp_end < end:
        periods.append(start)
        start = tmp_end
        tmp_end += step
    periods.append(start)
    return periods

def chunkit(list_, size):
    for i in range(0, len(list_), size):
        yield list_[i: i + size]

def today_timestamp() -> int:
    today = datetime.now().date()
    return date_timestamp(today)

def date_timestamp(date: date) -> int:
    return int(datetime(date.year, date.month, date.day).timestamp())

def safe_input_datetime(dtime: str) -> int:
    dtime = datetime.strptime(dtime, '%Y-%m-%d').date()
    dtime = date_timestamp(dtime) * 1000
    return dtime

async def _fetch_ohlcv(
    data_provider: Exchange,
    symbol: str,
    timeframe: str,
    since: int,
    limit: int = 100,
) -> list:
    repeat = 5
    _e = None
    while repeat:
        try:
            return await data_provider.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            repeat -= 1
            _e = e
    raise DataNotFetched(f"symbol: {symbol} -- timeframe: {timeframe} -- since: {since} not fetched") from _e

async def fetch_data(
    data_provider: Exchange,
    symbol: str,
    timeframe: str,
    start: str,
    output_path: str,
    output_format: Literal['pickle', 'csv'] = 'csv',
    end: str | None = None,
    limit: int = 100,
    api_limit: tuple = (10, 2)  # this mean 10 request in 2 second
) -> pd.DataFrame:
    if not end:
        _end = datetime.now().date().strftime('%Y-%m-%d')
        end = today_timestamp() * 1000
    else:
        _end = end
        end = safe_input_datetime(end)
    fname = f'{symbol}_{timeframe}_since_{start}_till_{_end}'
    start = safe_input_datetime(start)

    sinces = chunk_fetch_period(start, end, limit=limit)
    sinces = list(chunkit(sinces, api_limit[0]))

    datas = []
    for _sinces in sinces:
        _tmp = await asyncio.gather(*[_fetch_ohlcv(data_provider=data_provider, symbol=symbol, timeframe=timeframe, since=since, limit=limit) for since in _sinces])
        _tmp = [d for sub_d in _tmp for d in sub_d]
        datas.extend(_tmp)
        # await asyncio.sleep(api_limit[1])
    df = pd.DataFrame(datas, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df.set_index('timestamp', drop=True, inplace=True)
    # df['volume'] = df['volume'] * df['close']
    df = df[df.index <= end]
    match output_format:
        case 'csv':
            fname += '.csv'
            output_path += fname
            df.to_csv(output_path)
        case 'pickle':
            fname += '.pkl'
            output_path += fname
            df.to_pickle(output_path)
        case _:
            pass
    df.index = pd.to_datetime(df.index, unit='ms')
    return df

def convert_resample_timeframe(timeframe: str) -> str:
    c = timeframe[:-1]
    t = timeframe[-1]
    if t == 'm':
        t = "T"
    else:
        t = t.upper()
    tmp = f"{c}{t}"
    return tmp

def resample_data(
    base_data: pd.DataFrame,
    resample_timeframe: str,
) -> pd.DataFrame:
    resample_tm = convert_resample_timeframe(resample_timeframe)
    dfr = base_data.resample(resample_tm).agg(
        {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
    ).reset_index()
    dfr.set_index('timestamp', drop=True, inplace=True)
    return dfr.iloc[:-1]

def load_from_file(
    file_path: str,
    file_ext: Literal['csv', 'pickle'] = 'csv'
) -> pd.DataFrame:
    match file_ext:
        case 'csv':
            df = pd.read_csv(file_path)
        case 'pickle':
            df = pd.read_pickle(file_path)
        case _:
            raise ValueError('file_ext must be csv or pickle.')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', drop=True, inplace=True)
    return df

# %%
