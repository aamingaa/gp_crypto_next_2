# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np
import aiohttp
import asyncio
from datetime import datetime, timedelta


async def get_market_data_futures_v2(symbol, time_interval):
    urls = {
        'sum_toptrader_long_short_ratio': 'https://fapi.binance.com/futures/data/topLongShortPositionRatio',
        'count_toptrader_long_short_ratio': 'https://fapi.binance.com/futures/data/topLongShortAccountRatio',
        'sum_open_interest': 'https://fapi.binance.com/futures/data/openInterestHist',
        'sum_taker_long_short_vol_ratio': 'https://fapi.binance.com/futures/data/takerlongshortRatio',
        'count_long_short_ratio': 'https://fapi.binance.com/futures/data/globalLongShortAccountRatio',
        'candle_data': 'https://fapi.binance.com/fapi/v1/klines'
    }

    # 通用的请求参数
    params_ratios = {'symbol': symbol.upper(), 'period': time_interval, 'limit': 1}

    # 异步会话并发请求数据
    async with aiohttp.ClientSession() as session:
        # 创建任务
        tasks = {
            'sum_open_interest': fetch_json(session, urls['sum_open_interest'], params_ratios),
            'count_toptrader_long_short_ratio': fetch_json(session, urls['count_toptrader_long_short_ratio'], params_ratios),
            'sum_toptrader_long_short_ratio': fetch_json(session, urls['sum_toptrader_long_short_ratio'], params_ratios),
            'count_long_short_ratio': fetch_json(session, urls['count_long_short_ratio'], params_ratios),
            'sum_taker_long_short_vol_ratio': fetch_json(session, urls['sum_taker_long_short_vol_ratio'], params_ratios)
        }
        
        # 并发执行所有任务并获取结果
        results = await asyncio.gather(*tasks.values())
        
        # 使用字典解析返回结果
        data_map = {
            'oi': results[0][0].get('sumOpenInterest') if results[0] else None,
            'oi_ccy': results[0][0].get('sumOpenInterestValue') if results[0] else None,
            'toptrader_count_lsr': results[1][0].get('longShortRatio') if results[1] else None,
            'toptrader_oi_lsr': results[2][0].get('longShortRatio') if results[2] else None,
            'count_lsr': results[3][0].get('longShortRatio') if results[3] else None,
            'taker_vol_lsr': results[4][0].get('buySellRatio') if results[4] else None
        }

        # 计算过去时间
        now = datetime.now()
        since_time = now - timedelta(hours=24 * 10)
        params_candle = {'symbol': symbol, 'interval': time_interval, 'limit': 1500}

        # 分两次请求蜡烛图数据
        candle_data_1 = await fetch_json(session, urls['candle_data'], {**params_candle, 'endTime': int(now.timestamp()) * 1000})
        candle_data_2 = await fetch_json(session, urls['candle_data'], {**params_candle, 'startTime': int((now - timedelta(minutes=1500 * 5)).timestamp()) * 1000})

    # 验证并处理蜡烛图数据
    if candle_data_1 and candle_data_2:
        df_1 = pd.DataFrame(
            candle_data_1,
            columns=[
                'timestamp', 'o', 'h', 'l', 'c', 'vol', 'close_time', 
                'vol_ccy', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ]
        ).astype({
            'o': 'float', 'h': 'float', 'l': 'float', 'c': 'float', 
            'vol': 'float', 'vol_ccy': 'float', 'trades': 'int'
        })
        
        df_2 = pd.DataFrame(
            candle_data_2,
            columns=[
                'timestamp', 'o', 'h', 'l', 'c', 'vol', 'close_time', 
                'vol_ccy', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ]
        ).astype({
            'o': 'float', 'h': 'float', 'l': 'float', 'c': 'float', 
            'vol': 'float', 'vol_ccy': 'float', 'trades': 'int'
        })
        
        # 合并两个DataFrame
        df = pd.concat([df_1, df_2]).drop_duplicates().reset_index(drop=True)
        
        # 添加并调整时区
        df['candle_begin_time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['candle_begin_time_GMT4'] = df['candle_begin_time'] + timedelta(hours=4)
        
        # 选择需要的列
        df = df[['candle_begin_time_GMT4', 'o', 'h', 'l', 'c', 'vol', 'vol_ccy', 'trades']]
        
        # 添加其他比率数据列
        for col, value in data_map.items():
            df[col] = value
        
        print('df:', df.tail())
        print('len(df):', len(df))
        
        return df
    else:
        print("蜡烛图数据获取失败。")
        return None

async def fetch_json(session, url, params):
    """并发获取JSON数据的辅助函数"""
    async with session.get(url, params=params) as response:
        if response.status == 200:
            return await response.json()
        else:
            print(f"获取数据失败：{url}，状态码：{response.status}")
            return None