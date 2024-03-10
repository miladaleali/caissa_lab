import asyncio

from caissa_lab.data import fetch_data, get_data_provider

okx = get_data_provider()

# repeat = 3
# try:
#     while repeat:
#         try:
print(asyncio.run(okx.fetch_ohlcv('BTC-USDT-SWAP', '1m')))
asyncio.run(
    fetch_data(
        okx,
        symbol='BTC-USDT-SWAP',
        start='2022-12-01',
        timeframe='1m',
        output_path='./',
        end='2023-01-01'
    )
)
#             break
#         except Exception:
#             repeat -= 1
#     else:
#         print('ETH cant fetch.')
# except Exception:
#     pass


# symbols = [
#     'ADA-USDT-SWAP',
#     'XRP-USDT-SWAP',
#     'FTM-USDT-SWAP',
# ]
# for sym in symbols:
#     repeat = 3
#     while repeat:
#         try:
#             asyncio.run(
#                 fetch_data(
#                     okx,
#                     symbol=sym,
#                     start='2022-01-01',
#                     timeframe='1m',
#                     output_path='./',
#                 )
#             )
#             break
#         except Exception:
#             repeat -= 1
#     else:
#         print(f'{sym} cant fetch.')
