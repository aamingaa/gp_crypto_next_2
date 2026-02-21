'''
数据读取、降频处理和计算收益率模块
'''

import pandas as pd
import numpy as np
from pathlib import Path
import originalFeature
from scipy.optimize import minimize
import time
import talib as ta
from enum import Enum
import re

import sys
import matplotlib.pyplot as plt
from scipy.stats import zscore, kurtosis, skew, yeojohnson, boxcox
from scipy.stats import tukeylambda, mstats
from sklearn.preprocessing import RobustScaler
from typing import Dict, List, Optional, Tuple, Any
import time
import talib as ta
from enum import Enum
import re
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import normDataCheck

import sys
import matplotlib.pyplot as plt
from scipy.stats import zscore, kurtosis, skew, yeojohnson, boxcox
from scipy.stats import tukeylambda, mstats
import zipfile


def data_load(sym: str) -> pd.DataFrame:
    '''数据读取模块'''
    file_name = '/home/etern/crypto/data/merged/merged/' + sym + '-merged-without-rfr-1m.csv'  
    z = pd.read_csv(file_name, index_col=1)[
        ['o', 'h', 'l', 'c', 'vol', 'vol_ccy', 'trades',
               'oi', 'oi_ccy', 'toptrader_count_lsr', 'toptrader_oi_lsr', 'count_lsr',
               'taker_vol_lsr']]
    return z



def removed_zero_vol_dataframe(df):
    """
    打印并且返回-
    1. volume这一列为0的行组成的df
    2. low这一列的最小值
    3. volume这一列的最小值
    5. 去除掉volume=0的行的dataframe
    -------

    """
    # 将DataFrame的索引列设置为'datetime'
    df.index = pd.to_datetime(df.index)

    # 1. volume这一列为0的行组成的df
    volume_zero_df = df[df['vol'] == 0]
    print(f"Volume为0的行组成的DataFrame: {len(volume_zero_df)}")

    # 2. low这一列的最小值
    min_low = df['l'].min()
    print(f"Low这一列的最小值: {min_low}")

    # 3. volume这一列的最小值
    min_volume = df['vol'].min()
    print(f"Volume这一列的最小值: {min_volume}")

    # 5. 去除掉volume=0的行的dataframe
    removed_zero_vol_df = df[df['vol'] != 0]
    print(f"去除掉Volume为0的行之前的DataFrame length: {len(df)}")
    print(f"去除掉Volume为0的行之后的DataFrame length: {len(removed_zero_vol_df)}")

    return removed_zero_vol_df


def resample(z: pd.DataFrame, freq: str) -> pd.DataFrame:
    '''
    这是不支持vwap的，默认读入的数据是没有turnover信息，自然也没有vwap的信息，不需要获取sym的乘数
    '''
    if freq != '1min' or freq != '1m':
        z.index = pd.to_datetime(z.index)
        # 注意closed和label参数
        z = z.resample(freq, closed='left', label='left').agg({'o': 'first',
                                                               'h': 'max',
                                                               'l': 'min',
                                                               'c': 'last',
                                                               'vol': 'sum',
                                                               'vol_ccy': 'sum',
                                                               'trades': 'sum',
                                                               'oi': 'last', 
                                                               'oi_ccy': 'last', 
                                                               'toptrader_count_lsr':'last', 
                                                               'toptrader_oi_lsr':'last', 
                                                               'count_lsr':'last',
                                                               'taker_vol_lsr':'last'})
        # 注意resample后,比如以10min为resample的freq，9:00的数据是指9:00到9:10的数据~~
        z = z.fillna(method='ffill')   
        z.columns = ['o', 'h', 'l', 'c', 'vol', 'vol_ccy','trades',
               'oi', 'oi_ccy', 'toptrader_count_lsr', 'toptrader_oi_lsr', 'count_lsr',
               'taker_vol_lsr']

        # 重要，这个删掉0成交的操作，不能给5分钟以内的freq进行操作，因为这种情况还是挺容易出现没有成交的，这会改变本身的分布
        # 使用正则表达式提取开头的数值部分, 判断freq的周期
        match = re.match(r"(\d+)", freq)
        if match:
            int_freq = int(match.group(1))
        if int_freq > 5:
            z = removed_zero_vol_dataframe(z)
    return z

def data_prepare_coarse_grain_rolling_offset(
        sym: str, 
        freq: str,  # 预测周期，例如 '2h' 表示预测未来2小时收益
        start_date_train: str, 
        end_date_train: str,
        start_date_test: str, 
        end_date_test: str,
        coarse_grain_period: str = '2h',  # 粗粒度特征桶周期
        feature_lookback_bars: int = 8,    # 特征回溯桶数（8个2h = 16小时）
        rolling_time_step: str = '15min',       # 滚动步长
        y_train_ret_period: int = 8,
        rolling_w: int = 2000,
        output_format: str = 'ndarry',
        data_dir: str = '',
        read_frequency: str = '',
        timeframe: str = '',
        file_path: Optional[str] = None,
        use_parallel: bool = True,  # 是否使用并行处理
        n_jobs: int = -1,  # 并行进程数，-1表示使用所有CPU核心
        use_fine_grain_precompute: bool = True,  # 是否使用细粒度预计算优化
        include_categories: List[str] = None,
        remove_warmup_rows: bool = True,  # 是否删除rolling窗口未满的前rolling_w-1行
        predict_label: str = 'norm'  # 预测标签类型
    ):
    """
    粗粒度特征 + 细粒度滚动的数据准备方法（使用offset参数版本）
    
    核心思想：
    - 特征使用粗粒度周期（如2小时）聚合，减少噪声
    - 特征窗口使用固定时间长度（如8个2小时 = 16小时），预测起点以细粒度步长滚动（如15分钟），产生高频样本，**关键改进**：每个滚动时间点都独立计算其专属的滑动窗口特征，避免多个样本重复使用相同的粗粒度桶
    - 预测目标是未来N个粗粒度周期的收益（如未来2小时）
    
    参数说明：
    - sym: 交易对符号
    - freq: 用于兼容，实际预测周期由 y_train_ret_period * coarse_grain_period 决定
    - coarse_grain_period: 粗粒度特征桶周期，如 '2h', '1h', '30min'
    - feature_lookback_bars: 特征回溯的粗粒度桶数量（如8表示8个2h桶）
    - rolling_step: 滚动步长，如 '15min', '10min', '5min'
    - y_train_ret_period: 预测周期数（以rolling_step为单位）
    - remove_warmup_rows: 是否删除rolling窗口未满的前rolling_w-1行（默认False保留所有数据）
    
    示例场景（滑动窗口）：
    - coarse_grain_period='2h', feature_lookback_bars=8, rolling_step='15min'
    优势：
    - 每个时间点的特征窗口都是独立的，避免了数据泄露和样本相关性问题
    - 滚动步长可以任意设置，不受粗粒度周期限制
    - 特征更加精细，更能反映实时市场状态
    - 使用offset参数，避免时间偏移问题
    
    返回与 data_prepare 相同的接口
    """
    
    print(f"\n{'='*60}")
    print(f"粗粒度特征 + 细粒度滚动数据准备（offset参数版本）")
    print(f"品种: {sym}")
    print(f"回溯的周期: {coarse_grain_period}, 特征窗口 {feature_lookback_bars} × {coarse_grain_period} = {feature_lookback_bars * pd.Timedelta(coarse_grain_period).total_seconds() / 3600:.1f}小时")
    print(f"预测的周期: {y_train_ret_period} × {rolling_time_step} = {y_train_ret_period * pd.Timedelta(rolling_time_step).total_seconds() / 3600:.1f}小时")
    print(f"{'='*60}\n")
    
    # ========== 第一步：读取原始数据（细粒度） ==========
    z_raw = data_load_v2(sym, data_dir=data_dir, start_date=start_date_train, end_date=end_date_test,
                         timeframe=timeframe, read_frequency=read_frequency, file_path=file_path)
    z_raw.index = pd.to_datetime(z_raw.index)


    z_raw = z_raw[(z_raw.index >= pd.to_datetime(start_date_train)) 
                  & (z_raw.index <= pd.to_datetime(end_date_test))]
    
    print(f"读取原始数据: {len(z_raw)} 行，时间范围 {z_raw.index.min()} 至 {z_raw.index.max()}")
    
    # ========== 第二步：使用offset参数预计算粗粒度桶特征 ==========
    coarse_features_dict = {}
    
    # 计算需要多少组不同偏移的resample
    coarse_period_minutes = pd.Timedelta(coarse_grain_period).total_seconds() / 60
    rolling_time_step_td = pd.Timedelta(rolling_time_step)
    rolling_step_minutes = rolling_time_step_td.total_seconds() / 60
    num_offsets = int(coarse_period_minutes / rolling_step_minutes)


    if use_fine_grain_precompute:
        print(f"滚动步长: {rolling_step_minutes}分钟")
        print(f"需要预计算 {num_offsets} 组不同偏移的粗粒度桶")
        
        samples = []
        prediction_horizon_td = rolling_step_minutes * y_train_ret_period
        
        for i in range(num_offsets):
            offset = pd.Timedelta(minutes=i * rolling_step_minutes)
            print(f"\n组{i}: 偏移 {offset} ...")
            
            if i >= 1 :
                print("hello")

            z_raw_copy = z_raw.copy()
            original_start = z_raw_copy.index.min()
            original_end = z_raw_copy.index.max()

            coarse_bars = resample_with_offset(
                z = z_raw_copy, 
                resample_freq = coarse_grain_period, 
                offset=offset,  # 直接使用offset参数
                closed='left', 
                label='left'
            )
            
            # 🔧 过滤掉超出原始数据范围的桶
            coarse_bars = coarse_bars [
                (coarse_bars.index >= original_start) & 
                (coarse_bars.index <= original_end)
            ]

            # 计算特征
            base_feature = originalFeature.BaseFeature(
                coarse_bars.copy(), 
                include_categories=include_categories, 
                rolling_zscore_window=int(rolling_w / num_offsets)
            )

            features_df = base_feature.init_feature_df
            
            # 说明：
            # - z_raw 是 15m K 线（index 为 open_time）。
            # - coarse_bars 是对 z_raw 按 coarse_grain_period（如 2h）resample(closed='left', label='left', offset=...) 得到的粗粒度桶；
            #   其 index 同样是桶的 left label（也就是桶的 open_time / 起始时刻 t0）。
            # - features_df.index (= row_timestamps) 表示粗桶起始时刻 t0（例如 10:00）。
            #   该行特征使用的是区间 [t0, t0 + coarse_grain_period) 的聚合结果；并且 BaseFeature 内的 rolling_zscore_window 是“粗桶行数”，
            # - decision_timestamps = t0 + coarse_grain_period（例如 12:00），代表桶结束/决策时刻。
            # - prediction_timestamps = decision_timestamps + rolling_step * y_train_ret_period（例如 rolling_step=15m 且 y_train_ret_period=8 时为 14:00）。
            # - 当前价(用于 label 分母)取决策点刚结束的那根 15m K 线的 close：current_data_timestamps = decision_timestamps - 15m
            #   （例如 11:45，对应区间 [11:45, 12:00) 的 close）。
           
            # 1. 计算决策时间 (物理时间 12:00)
            # decision_timestamps = features_df.index + pd.to_timedelta(coarse_grain_period)

            # 2. 【核心修正】计算在 z_raw (Open Time Index) 中对应的“当前行”
            # 如果决策时间是 12:00，我们需要取 11:45 开始的那根 K 线 (因为它在 12:00 结束)
            # lookup_offset = pd.Timedelta(rolling_step) # 例如 15min
            # current_data_timestamps = decision_timestamps - lookup_offset
            current_data_timestamps = features_df.index + pd.to_timedelta(coarse_grain_period)
            
            # 向量化计算未来时刻
            prediction_timestamps = current_data_timestamps + prediction_horizon_td

            # ==============================================================================
            # 🆕 第二步：向量化获取预计算好的 "平滑 Label"
            # ==============================================================================
            
            # 1. 获取当前时刻的价格 和 波动率
            t_prices = z_raw['c'].reindex(current_data_timestamps)
            o_prices = z_raw['o'].reindex(current_data_timestamps)
            
            t_future_smooth = None
            scaled_label = None
            return_p = None
            
            t_future_smooth = z_raw['c'].reindex(prediction_timestamps).values
            return_p = t_future_smooth / t_prices.values
            scaled_label = np.log(return_p)
                    
            # 将标签添加到features_df
            features_df['feature_offset'] = offset.total_seconds() / 60  # 转换为分钟
            # features_df['decision_timestamps'] = current_data_timestamps
            features_df['current_data_timestamps'] = current_data_timestamps
            features_df['prediction_timestamps'] = prediction_timestamps
            features_df['t_price'] = t_prices.values
            features_df['t_future_price'] = t_future_smooth
            features_df['o_price'] = o_prices.values
            features_df['return_f'] = scaled_label
            features_df['return_p'] = return_p

            valid_mask = ~np.isnan(features_df['return_f'])
            features_df_mask = features_df[valid_mask]

            coarse_features_dict[offset] = features_df_mask
            samples.append(features_df_mask)

            print(f"  ✓ 组{i}完成: {len(features_df)} 个桶, {len(features_df.columns)} 个特征")
        
        print(f"\n✓ 预计算完成: {num_offsets} 组粗粒度特征")
        print(f"优化策略: 每个时间点根据其offset选择对应组的预计算特征")
    
    # ========== 第三步：生成细粒度滚动时间网格 ==========    
    if len(samples) > 0 and isinstance(samples[0], pd.DataFrame):
        df_samples = pd.concat(samples, axis=0, ignore_index=False, copy=False)
                
        # 检查并处理重复的时间戳
        if df_samples.index.duplicated().any():
            num_duplicates = df_samples.index.duplicated().sum()
            df_samples = df_samples[~df_samples.index.duplicated(keep='first')]
            print(f"  发现 {num_duplicates} 个重复时间戳 ✓ 去重后保留 {len(df_samples)} 行（保留first）")

        df_samples.sort_index(inplace=True)
        df_samples.dropna(inplace=True)
    else:
        print(f"  使用pd.DataFrame合并{len(samples)}个样本...")
        df_samples = pd.DataFrame(samples)
        df_samples.set_index('timestamp', inplace=True)
        df_samples.sort_index(inplace=True)
    
    print(f"样本时间范围: {df_samples.index.min()} 至 {df_samples.index.max()}, 样本数量: {len(df_samples)}")
    
    df_samples['ret_rolling_zscore'] = normDataCheck.norm(df_samples['return_f'].values, window=rolling_w)
    # df_samples['ret_rolling_zscore'] = norm_ret(df_samples['return_f'].values, window=rolling_w)
    remove_warmup_rows = True
            
    # ========== 删除rolling窗口未满的行（可选） ==========
    if remove_warmup_rows and len(df_samples) > rolling_w:
        print(f"\n删除前 {rolling_w-1} 行（rolling窗口预热期）")
        original_len = len(df_samples)
        df_samples = df_samples.iloc[rolling_w:]
        print(f"   数据行数: {original_len} → {len(df_samples)}")
        print(f"   新的时间范围: {df_samples.index.min()} 至 {df_samples.index.max()}")

    print(f"return_f - 偏度: {df_samples['return_f'].skew():.4f}, 峰度: {df_samples['return_f'].kurtosis():.4f}")
    print(f"ret_rolling_zscore - 偏度: {df_samples['ret_rolling_zscore'].skew():.4f}, 峰度: {df_samples['ret_rolling_zscore'].kurtosis():.4f}")
    
    # ========== 分割训练集和测试集 ==========
    # 说明：
    # - 训练集样本的标签使用 [decision_timestamps, prediction_timestamps] 区间的未来收益
    # - 为避免训练样本在计算标签时跨越到测试区间，这里在 end_date_train 前预留 2 * prediction_horizon_td 的安全边界
    effective_end_train = pd.to_datetime(end_date_train) - 2 * prediction_horizon_td
    train_mask = (df_samples.index >= pd.to_datetime(start_date_train)) & \
                 (df_samples.index < pd.to_datetime(effective_end_train))
    
    test_mask = (df_samples.index >= pd.to_datetime(start_date_test)) & \
                (df_samples.index <= pd.to_datetime(end_date_test))
    
    # 提取特征列
    exclude_cols = ['t_price', 'o_price', 't_future_price', 'return_p', 'return_f', 
                   'prediction_timestamps', 'ret_rolling_zscore', 'feature_offset', 'decision_timestamps', 'current_data_timestamps']
    feature_cols = [col for col in df_samples.columns if col not in exclude_cols]
    
    X_all = df_samples[feature_cols]
    X_train = df_samples.loc[train_mask, feature_cols]
    X_test = df_samples.loc[test_mask, feature_cols]
    
    y_train = df_samples.loc[train_mask, 'ret_rolling_zscore']
    y_test = df_samples.loc[test_mask, 'ret_rolling_zscore']
    
    ret_train = df_samples.loc[train_mask, 'return_f']
    ret_test = df_samples.loc[test_mask, 'return_f']
    
    y_p_train_origin = df_samples.loc[train_mask, 'return_p']
    y_p_test_origin = df_samples.loc[test_mask, 'return_p']
    
    open_train = df_samples.loc[train_mask, 'o_price']
    close_train = df_samples.loc[train_mask, 't_price']
    open_test = df_samples.loc[test_mask, 'o_price']
    close_test = df_samples.loc[test_mask, 't_price']
    
    feature_names = feature_cols
    
    # 保存训练集 / 测试集对应的时间索引，供后续因子模块精确对齐使用
    train_index = df_samples.index[train_mask]
    test_index = df_samples.index[test_mask]
    
    # 格式转换
    if output_format == 'ndarry':
        X_all = X_all.values
        X_train = X_train.values
        X_test = X_test.values
    else:
        raise ValueError(f"output_format 应为 'ndarry' 或 'dataframe'，当前为 {output_format}")
    
    # 构建 ohlc DataFrame
    ohlc_aligned = pd.DataFrame({
        'c': df_samples['t_price'],
        'close': df_samples['t_price']
    }, index=df_samples.index)
    
    print('检查x all是不是等于 x train和y train相加，再检查train和test以及close和open的形状是否一致')
    print(f'检查X_all的形状 {X_all.shape}')
    print(f'检查x dataset train的形状 {X_train.shape}')
    print(f'检查y dataset train的形状 {y_train.shape}')
    print(f'检查x all是不是等于train和test相加 {len(X_all)},{len(X_test)+len(X_train)}')
    print(f'检查open train的形状 {open_train.shape}')
    print(f'检查close train的形状 {close_train.shape}')
    print(f'检查x dataset test的形状 {X_test.shape}')
    print(f'检查y dataset test的形状 {y_test.shape}')
    print(f'检查open test的形状 {open_test.shape}')
    print(f'检查close test的形状 {close_test.shape}')
    print(f'检查ohlc_aligned的形状 {ohlc_aligned.shape}')
    print(f'检查y_p_train_origin的形状 {y_p_train_origin.shape}')
    print(f'检查y_p_test_origin的形状 {y_p_test_origin.shape}')

    # 返回接口与 data_prepare 基本保持一致，新增 train_index / test_index 用于后续对齐因子
    return (X_all, X_train, y_train, ret_train, X_test, y_test, ret_test,
            feature_names, open_train, open_test, close_train, close_test,
            df_samples.index, ohlc_aligned, y_p_train_origin, y_p_test_origin,
            train_index, test_index)


def data_prepare_coarse_grain_rolling_offset_v2(
        sym: str,
        freq: str,
        start_date_train: str,
        end_date_train: str,
        start_date_test: str,
        end_date_test: str,
        coarse_grain_period: str = '2h',
        feature_lookback_bars: int = 8,
        rolling_time_step: str = '15min',
        y_train_ret_period: int = 8,
        rolling_w: int = 2000,
        output_format: str = 'ndarry',
        data_dir: str = '',
        read_frequency: str = '',
        timeframe: str = '',
        file_path: Optional[str] = None,
        use_parallel: bool = True,
        n_jobs: int = -1,
        use_fine_grain_precompute: bool = True,
        include_categories: List[str] = None,
        remove_warmup_rows: bool = True,
        predict_label: str = 'norm'
    ):
    """
    粗粒度特征 + 细粒度滚动（修正版）

    修复点：
    1) 时间锚点统一为“决策时刻”，避免按桶起点切分带来的边界歧义。
    2) current/future 价格都用 15min open_time index 对齐到“已收盘的K线”。
    3) 预测周期严格使用 Timedelta 计算，避免分钟数值直接加到时间戳。
    4) 仅在必要列做缺失处理，避免对 OHLC 全量 ffill 造成伪K线。
    """
    del freq, feature_lookback_bars, use_parallel, n_jobs, use_fine_grain_precompute, predict_label

    print(f"\n{'='*60}")
    print("粗粒度特征 + 细粒度滚动数据准备（v2修正版）")
    print(f"品种: {sym}")
    print(f"特征周期: {coarse_grain_period}, 滚动步长: {rolling_time_step}")
    print(f"预测周期: {y_train_ret_period} × {rolling_time_step}")
    print(f"{'='*60}\n")

    # ========== 第一步：读取细粒度原始数据 ==========
    z_raw = data_load_v2(
        sym,
        data_dir=data_dir,
        start_date=start_date_train,
        end_date=end_date_test,
        timeframe=timeframe,
        read_frequency=read_frequency,
        file_path=file_path
    )
    z_raw = z_raw.copy()
    z_raw.index = pd.to_datetime(z_raw.index)
    z_raw.sort_index(inplace=True)
    z_raw = z_raw[(z_raw.index >= pd.to_datetime(start_date_train)) &
                  (z_raw.index <= pd.to_datetime(end_date_test))]
    print(f"读取原始数据: {len(z_raw)} 行，时间范围 {z_raw.index.min()} 至 {z_raw.index.max()}")

    # ========== 第二步：参数检查与时间常量 ==========
    coarse_td = pd.Timedelta(coarse_grain_period)
    rolling_step_td = pd.Timedelta(rolling_time_step)
    prediction_horizon_td = rolling_step_td * y_train_ret_period

    if coarse_td <= pd.Timedelta(0) or rolling_step_td <= pd.Timedelta(0):
        raise ValueError("coarse_grain_period 与 rolling_time_step 必须大于0")
    if coarse_td % rolling_step_td != pd.Timedelta(0):
        raise ValueError("coarse_grain_period 必须能被 rolling_time_step 整除")

    num_offsets = int(coarse_td / rolling_step_td)
    print(f"需要预计算 {num_offsets} 组 offset（{coarse_grain_period} / {rolling_time_step}）")

    # 与原方法保持一致：固定使用这7列做聚合，便于后续特征扩展
    agg_map = {
        'o': 'first',
        'h': 'max',
        'l': 'min',
        'c': 'last',
        'vol': 'sum',
        'vol_ccy': 'sum',
        'trades': 'sum',
    }

    # ========== 第三步：按 offset 预计算 2h 特征 ==========
    samples = []
    for i in range(num_offsets):
        offset = i * rolling_step_td
        print(f"\n组{i}: offset={offset}")

        coarse_bars = z_raw.resample(
            coarse_grain_period,
            closed='left',
            label='left',
            offset=offset
        ).agg(agg_map)

        # 对OHLC缺失的桶直接丢弃，成交量类缺失填0
        coarse_bars = coarse_bars.dropna(subset=['o', 'h', 'l', 'c'])
        for col in ('vol', 'vol_ccy', 'trades'):
            coarse_bars[col] = coarse_bars[col].fillna(0.0)

        # 防止resample产生范围外空桶
        coarse_bars = coarse_bars[(coarse_bars.index >= z_raw.index.min()) &
                                  (coarse_bars.index <= z_raw.index.max())]
        if coarse_bars.empty:
            continue

        base_feature = originalFeature.BaseFeature(
            coarse_bars.copy(),
            include_categories=include_categories,
            rolling_zscore_window=max(1, int(np.ceil(rolling_w / max(1, num_offsets))))
        )
        features_df = base_feature.init_feature_df.copy()
        if features_df.empty:
            continue

        # 时间轴定义（严格按 15min open_time 索引对齐）：
        # 1) features_df.index = 粗粒度桶起点 t0，对应特征来自区间 [t0, t0 + coarse_td)
        # 2) decision_timestamps = t0 + coarse_td，表示该粗桶刚结束、可做预测的决策时刻
        #    例：t0=10:00, coarse_td=2h -> decision=12:00
        block_start = pd.to_datetime(features_df.index)
        decision_timestamps = block_start + coarse_td

        # 3) current_data_timestamps = decision - rolling_step：
        #    在 open_time 索引下，decision 对应的是“下一根K线的起点”，
        #    因此当前可用价格应取上一根已收盘K线的 close，避免前视泄露
        #    例：rolling_step=15min -> current_data_timestamp=11:45（取11:45这根K线的close）
        current_data_timestamps = decision_timestamps
        # 4) prediction_timestamps = 未来预测终点（物理时间）
        #    例：prediction_horizon_td=2h -> prediction_timestamps=14:00
        prediction_timestamps = decision_timestamps + prediction_horizon_td
        # 5) prediction_data_timestamps = prediction_end - rolling_step：
        #    与当前价同理，取未来终点前一根已收盘K线的 close 作为标签分子
        #    例：prediction_data_timestamp=13:45（取13:45这根K线的close）
        prediction_data_timestamps = prediction_timestamps

        t_prices = z_raw['c'].reindex(current_data_timestamps)
        o_prices = z_raw['o'].reindex(current_data_timestamps)
        t_future_prices = z_raw['c'].reindex(prediction_data_timestamps)

        return_p = t_future_prices.values / t_prices.values
        return_f = np.log(return_p)

        features_df['feature_offset'] = offset.total_seconds() / 60.0
        features_df['decision_timestamps'] = decision_timestamps
        features_df['current_data_timestamps'] = current_data_timestamps
        features_df['prediction_timestamps'] = prediction_timestamps
        features_df['prediction_data_timestamps'] = prediction_data_timestamps
        features_df['t_price'] = t_prices.values
        features_df['t_future_price'] = t_future_prices.values
        features_df['o_price'] = o_prices.values
        features_df['return_f'] = return_f
        features_df['return_p'] = return_p

        # 统一把索引设为决策时刻，后续切分按该锚点执行
        # features_df.index = decision_timestamps

        valid_mask = (
            ~np.isnan(features_df['return_f']) &
            ~np.isnan(features_df['t_price']) &
            ~np.isnan(features_df['t_future_price'])
        )
        features_df = features_df[valid_mask]
        samples.append(features_df)

        print(f"  ✓ 样本数: {len(features_df)}")

    if len(samples) == 0:
        raise ValueError("未生成有效样本，请检查时间范围与参数设置")

    # ========== 第四步：合并样本并做标签标准化 ==========
    df_samples = pd.concat(samples, axis=0, ignore_index=False, copy=False)
    if df_samples.index.duplicated().any():
        dup_cnt = int(df_samples.index.duplicated().sum())
        df_samples = df_samples[~df_samples.index.duplicated(keep='first')]
        print(f"发现重复决策时间戳 {dup_cnt} 个，已按 first 去重")
    df_samples.sort_index(inplace=True)

    # 因果滚动zscore：仅使用当前及历史收益，不使用未来信息
    ret_series = pd.Series(df_samples['return_f'].values, index=df_samples.index)
    rolling_mean = ret_series.rolling(window=rolling_w, min_periods=rolling_w).mean()
    rolling_std = ret_series.rolling(window=rolling_w, min_periods=rolling_w).std(ddof=0)
    df_samples['ret_rolling_zscore'] = (ret_series - rolling_mean) / rolling_std.replace(0.0, np.nan)

    if remove_warmup_rows:
        before_len = len(df_samples)
        df_samples = df_samples.dropna(subset=['ret_rolling_zscore'])
        print(f"删除rolling预热期后: {before_len} -> {len(df_samples)}")
    else:
        df_samples = df_samples.dropna(subset=['return_f', 'ret_rolling_zscore'])

    # ========== 第五步：按决策时刻切分训练/测试 ==========
    # 训练样本要求其标签终点不超过训练区间结束
    effective_end_train = pd.to_datetime(end_date_train) - prediction_horizon_td
    train_mask = (df_samples.index >= pd.to_datetime(start_date_train)) & \
                 (df_samples.index <= effective_end_train)
    test_mask = (df_samples.index >= pd.to_datetime(start_date_test)) & \
                (df_samples.index <= pd.to_datetime(end_date_test))

    exclude_cols = [
        't_price', 'o_price', 't_future_price', 'return_p', 'return_f',
        'prediction_timestamps', 'prediction_data_timestamps',
        'ret_rolling_zscore', 'feature_offset', 'decision_timestamps',
        'current_data_timestamps'
    ]
    feature_cols = [col for col in df_samples.columns if col not in exclude_cols]

    X_all = df_samples[feature_cols]
    X_train = df_samples.loc[train_mask, feature_cols]
    X_test = df_samples.loc[test_mask, feature_cols]

    y_train = df_samples.loc[train_mask, 'ret_rolling_zscore']
    y_test = df_samples.loc[test_mask, 'ret_rolling_zscore']
    ret_train = df_samples.loc[train_mask, 'return_f']
    ret_test = df_samples.loc[test_mask, 'return_f']
    y_p_train_origin = df_samples.loc[train_mask, 'return_p']
    y_p_test_origin = df_samples.loc[test_mask, 'return_p']

    open_train = df_samples.loc[train_mask, 'o_price']
    close_train = df_samples.loc[train_mask, 't_price']
    open_test = df_samples.loc[test_mask, 'o_price']
    close_test = df_samples.loc[test_mask, 't_price']
    feature_names = feature_cols

    train_index = df_samples.index[train_mask]
    test_index = df_samples.index[test_mask]

    if output_format == 'ndarry':
        X_all = X_all.values
        X_train = X_train.values
        X_test = X_test.values
    else:
        raise ValueError(f"output_format 应为 'ndarry' 或 'dataframe'，当前为 {output_format}")

    ohlc_aligned = pd.DataFrame(
        {'c': df_samples['t_price'], 'close': df_samples['t_price']},
        index=df_samples.index
    )

    print("检查输出形状:")
    print(f"X_all: {X_all.shape}, X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
    print(f"train_index: {train_index.min() if len(train_index) else None} -> {train_index.max() if len(train_index) else None}")
    print(f"test_index: {test_index.min() if len(test_index) else None} -> {test_index.max() if len(test_index) else None}")

    return (X_all, X_train, y_train, ret_train, X_test, y_test, ret_test,
            feature_names, open_train, open_test, close_train, close_test,
            df_samples.index, ohlc_aligned, y_p_train_origin, y_p_test_origin,
            train_index, test_index)

# 应用滚动标准化到标签
def norm_ret(x, window=2000):
    x = np.log1p(np.asarray(x))
    factors_data = pd.DataFrame(x, columns=['factor'])
    factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
    factors_std = factors_data.rolling(window=window, min_periods=1).std()
    factor_value = factors_data / factors_std
    factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
    return np.nan_to_num(factor_value).flatten()
    

def data_prepare(sym, freq, start_date_train, end_date_train, start_date_test, end_date_test, y_train_ret_period=1,
                 rolling_w=2000, output_format='ndarry', _compute_transformed_series=False, _check_zscore_window_series=False):
    '''
    内置了一些对于label的分析，比较关键, 但只需要研究和对比时才会开启

        # 对于Label的分析的指导目标，是希望它能够接近正态分布，偏度，峰度接近于0
    # 方案1，先对log_return做clip，完全去除了outlier，再看偏度峰度，决定后续是否rolling_zscore.
    # 方案2，先对log_return做rolling_zscore,
    # (rolling窗口值是2000，暂时当做经验性的参数，取值的自由度来源于詹森不等式和大数定理，都是用数据算出来的)
        # 1. 参数和单点夏普的关系不明确，但是和几万个因子的夏普只和，他们的关系应该存在一定的凸性；
        # 2. 参数的设置应该在维持1的前提下兼顾大数定理；
        # 3. samples拆分成为几个class后样本量仍然符合大数定理；
    # 如上两种方案的对比，当前认为是应该第二种方式，应该是能够保留一部分outlier的信息，相对平衡的减轻outlier的影响

    Note - 最终要把窗口还没积累完全的部分，删除掉这些样本，否则会影响训练的结果。往往是都生成了feature之后，最后处理好label，再做切割。
    '''

    # -----------------------------------------
    z = data_load(sym)
    # 切分数据，只取需要的部分 - train and test
    z.index = pd.to_datetime(z.index)
    print(f'开始处理 {sym} 的历史数据')   
    print(f'len of z before select = {len(z)}')
    z = z[(z.index >= pd.to_datetime(start_date_train)) & (
        z.index <= pd.to_datetime(end_date_test))]  # 只截取参数指定部分dataframe
    print(f'len of z after select = {len(z)}')
    ohlcva_df = resample(z, freq)

    print(f'len of resample_z = {len(ohlcva_df)}')
    # --------------------------------------------------------
    if _compute_transformed_series:
        # 分析label的分布，画出label的各类处理后的 三阶矩，四阶矩
        compute_transformed_series(z.c)
    if _check_zscore_window_series:
        # 画图，展现各种窗口下的label的log_return
        check_zscore_window_series(z, 'c')
    # --------------------------------------------------------
    print('开始生成初始特征')
    base_feature = originalFeature.BaseFeature(ohlcva_df.copy())
    z = base_feature.init_feature_df

    # -----------------------------------------

    # 关键 - 生成ret这一列，这是label数值，整个因子评估体系的基础，要注意分析label分布的skewness, kurtosis等.
    # note - 需要把空值处理掉，因为测试集中的最后的几个空值可能刚好影响测试的持仓效果.
    # 注意使用滑动窗口时，对于没填满的区域，和最后空空值区域，也要有类似的考量，防止刚好碰到极值label引起失真影响。
    print('开始生成ret')
    z['return_f'] = np.log(z['c']).diff(
        y_train_ret_period).shift(-y_train_ret_period)
    z['return_f'] = z['return_f'].fillna(0)
    z['r'] = np.log(z['c']).diff()
    z['r'] = z['r'].fillna(0)


    # ---方案2， 先对label做rolling_zscore---------------
    def norm_ret(x, window=rolling_w):  # 不再用L2 norm，恢复到之前的zscore，然后这里需要做的是给他增加一个周期

        # 注意这个函数是用在return上面的，log1p最小的数值是-1，用于return合适
        x = np.log1p(np.asarray(x))
        factors_data = pd.DataFrame(x, columns=['factor'])
        factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
        # factors_mean = factors_data.rolling(
        #     window=window, min_periods=1).mean()
        factors_std = factors_data.rolling(window=window, min_periods=1).std()
        factor_value = factors_data / factors_std
        factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
        # factor_value = factor_value.clip(-6, 6)
        return np.nan_to_num(factor_value).flatten()


    # Note - 先强行使用norm_ret看效果
    z['ret_rolling_zscore'] = norm_ret(z['return_f'])

    # 此时，所有的features和label，都用相同窗口做完了rolling处理，为了训练模型的准确性，可以开始删除掉还没有存满窗口的那些行了。去除前window行
    # 重要！！ 如果执行如下这句，会z与上面的ohlcva_df不一致，导致originalFeature.BaseFeature(ohlcva_df)初始化的ohlcva_df与做eval的feature_data不一致
    # z = z.iloc[window-1:]


    kept_index = z.index
    ohlcva_df = ohlcva_df.loc[kept_index]
    open_train = ohlcva_df['o'][(ohlcva_df['o'].index >= pd.to_datetime(start_date_train)) & (
            ohlcva_df['o'].index < pd.to_datetime(end_date_train))]
    open_test = ohlcva_df['o'][(ohlcva_df['o'].index >= pd.to_datetime(start_date_test)) & (
            ohlcva_df['o'].index <= pd.to_datetime(end_date_test))]
    close_train = ohlcva_df['c'][(ohlcva_df['c'].index >= pd.to_datetime(start_date_train)) & (
            ohlcva_df['c'].index < pd.to_datetime(end_date_train))]
    close_test = ohlcva_df['c'][(ohlcva_df['c'].index >= pd.to_datetime(start_date_test)) & (
            ohlcva_df['c'].index <= pd.to_datetime(end_date_test))]

    print('ret_rolling_zscore skew = {}'.format(
        z['ret_rolling_zscore'].skew()))
    print('ret_rolling_zscore kurtosis = {}'.format(
        z['ret_rolling_zscore'].kurtosis()))

    print('return skew = {}'.format(z['return_f'].skew()))
    print('return kurtosis = {}'.format(z['return_f'].kurtosis()))


    # 切分为train和test两个数据集，但是注意，test数据集其实带入了之前的数据的窗口, 是要特意这么做的。
    z_train = z[(z.index >= pd.to_datetime(start_date_train)) & (
        z.index < pd.to_datetime(end_date_train))]  # 只截取参数指定部分dataframe
    z_test = z[(z.index >= pd.to_datetime(start_date_test)) & (
        z.index <= pd.to_datetime(end_date_test))]
    # ------------<label 分析>-------------------------

    # 对于Label的分析的指导目标，是希望它能够接近正态分布，偏度，峰度接近于0
    # 方案1，先对log_return做clip，完全去除了outlier，再看偏度峰度，决定后续是否rolling_zscore.
    # 方案2，先对log_return做rolling_zscore,
    # (rolling窗口值是2000，暂时当做经验性的参数，取值的自由度来源于詹森不等式和大数定理，都是用数据算出来的)
    # 1. 参数和单点夏普的关系不明确，但是和几万个因子的夏普只和，他们的关系应该存在一定的凸性；
    # 2. 参数的设置应该在维持1的前提下兼顾大数定理；
    # 3. samples拆分成为几个class后样本量仍然符合大数定理；
    # 如上两种方案的对比，当前认为是应该第二种方式，应该是能够保留一部分outlier的信息，相对平衡的减轻outlier的影响



    if output_format == 'ndarry':
        y_dataset_train = z_train['ret_rolling_zscore'].values
        y_dataset_test = z_test['ret_rolling_zscore'].values
        ret_dataset_train = z_train['return_f'].values
        ret_dataset_test = z_test['return_f'].values
        # 重要！要删除掉包含未来信息的字段，ret，ret_rolling_zscore
        z_train.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)
        z_test.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)
        z.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)

        X_all = np.where(np.isnan(z), 0, z)
        X_dataset_train = np.where(np.isnan(z_train), 0, z_train)
        X_dataset_test = np.where(np.isnan(z_test), 0, z_test)

    elif output_format == 'dataframe':
        y_dataset_train = z_train['ret_rolling_zscore']
        y_dataset_test = z_test['ret_rolling_zscore']
        ret_dataset_train = z_train['return_f']
        ret_dataset_test = z_test['return_f']
        # 重要！要删除掉包含未来信息的字段，ret，ret_rolling_zscore
        z_train.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)
        z_test.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)
        z.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)

        X_all = z.fillna(0)
        X_dataset_train = z_train.fillna(0)
        X_dataset_test = z_test.fillna(0)

    else:
        print(
            'output_format of data_prepare should be "ndarry" or "dataframe", pls check it ')
        exit(1)

    feature_names = z_train.columns

    print('检查x all是不是等于 x train和y trian相加，再检查trian和test以及close和open的形状是否一致')
    # X_all 专门是为做batch prediction的时候，要用X_all生成test集要用到的factor_df, 因为factor的计算需要之前一段window中的feature值
    print(f'检查X_all的形状 {X_all.shape}')
    print(f'检查x dataset train的形状 {X_dataset_train.shape}')
    print(f'检查y dataset train的形状 {y_dataset_train.shape}')
    print(f'检查x all是不是等于train和test相加 {len(X_all)},{len(X_dataset_test)+len(X_dataset_train)}')
    print(f'检查open train的形状 {open_train.shape}')
    print(f'检查close train的形状 {close_train.shape}')
    print(f'检查x dataset test的形状 {X_dataset_test.shape}')
    print(f'检查y dataset test的形状 {y_dataset_test.shape}')
    print(f'检查open test的形状 {open_test.shape}')
    print(f'检查close test的形状 {close_test.shape}')

    # X_all 专门是为做batch prediction的时候，要用X_all生成test集要用到的factor_df, 因为factor的计算需要之前一段window中的feature值
    return X_all, X_dataset_train, y_dataset_train,ret_dataset_train, X_dataset_test, y_dataset_test,ret_dataset_test, feature_names,open_train,open_test,close_train,close_test, z.index ,ohlcva_df


def compute_transformed_series(column):
    """
    输入的是一个dataframe的一列，series.
    计算得到如下几个ndarry-

    1. log_return: 取log return。
    2. log_log_return：对log_return再做一次log.
    3. boxcox_transformed: 使用Box-Cox变换。
    4. yeo_johnson_transformed: 使用Yeo-Johnson变换。
    5. winsorized_log_return: 对log_return进行Winsorizing。
    6. scaled_log_return: 对log_return进行RobustScaler缩放。

    plot一个直方图，上面4种颜色显示如上四类数值各自的直方分布图.
    并且在图上画出如上4个序列各自的skewness和kurtosis
    -------

    """
    log_return = (np.log(column).diff(1).fillna(0)*1).shift(-1)
    log_return = np.where(np.isnan(log_return), 0, log_return)

    # ---------尝试对log return做滚动标准化--------
    log_return = _rolling_zscore(log_return, 300)

    # 计算 log_log_return
    log_log_return = np.log(log_return + 1)
    # log_log_return_2 = np.log(np.log(column)).diff().fillna(0).shift(-1)

    # 平移数据使其为正值
    log_return_shifted = log_return - np.min(log_return) + 1
    # 应用 Box-Cox 变换
    boxcox_transformed, _ = boxcox(log_return_shifted)

    # 应用 Yeo-Johnson 变换
    yeo_johnson_transformed, _ = yeojohnson(log_return)

    # 应用 Winsorizing
    winsorized_log_return = mstats.winsorize(log_return, limits=[0.05, 0.05])

    # # 应用 RobustScaler
    # scaler = RobustScaler()
    # scaled_log_return = scaler.fit_transform(log_return).flatten()

    # 绘制直方图
    plt.figure(figsize=(12, 6))

    # 绘制 log return 的直方图
    plt.hist(log_return, bins=160, alpha=0.3, color='blue', label='Log Return')

    # 绘制 log_log_return 的直方图
    plt.hist(log_log_return, bins=160, alpha=0.3,
             color='orange', label='Log Log Return')

    # 绘制 boxcox_transformed 的直方图
    plt.hist(boxcox_transformed, bins=160, alpha=0.3,
             color='green', label='Box-Cox Transformed')

    # 绘制 yeo_johnson_transformed 的直方图
    plt.hist(yeo_johnson_transformed, bins=160, alpha=0.3,
             color='red', label='Yeo-Johnson Transformed')

    # 绘制 winsorized_log_return 的直方图
    plt.hist(winsorized_log_return, bins=160, alpha=0.3,
             color='red', label='Winsorized Transformed')

    # 计算并显示 skewness 和 kurtosis
    log_return_skewness = skew(log_return)
    log_return_kurtosis = kurtosis(log_return)
    log_log_return_skewness = skew(log_log_return)
    log_log_return_kurtosis = kurtosis(log_log_return)
    boxcox_skewness = skew(boxcox_transformed)
    boxcox_kurtosis = kurtosis(boxcox_transformed)
    yeo_johnson_skewness = skew(yeo_johnson_transformed)
    yeo_johnson_kurtosis = kurtosis(yeo_johnson_transformed)
    winsorized_skewness = skew(winsorized_log_return)
    winsorized_kurtosis = kurtosis(winsorized_log_return)

    # 在图上显示 skewness 和 kurtosis
    plt.text(0.05, 0.95,
             f'Log Return Skewness: {log_return_skewness:.2f}\nLog Return Kurtosis: {log_return_kurtosis:.2f}',
             fontsize=10, transform=plt.gca().transAxes, va='top', ha='left', bbox=dict(facecolor='blue', alpha=0.5))
    plt.text(0.05, 0.85,
             f'Log Log Return Skewness: {log_log_return_skewness:.2f}\nLog Log Return Kurtosis: {log_log_return_kurtosis:.2f}',
             fontsize=10, transform=plt.gca().transAxes, va='top', ha='left', bbox=dict(facecolor='orange', alpha=0.5))
    plt.text(0.05, 0.75,
             f'Box-Cox Transformed Skewness: {boxcox_skewness:.2f}\nBox-Cox Transformed Kurtosis: {boxcox_kurtosis:.2f}',
             fontsize=10, transform=plt.gca().transAxes, va='top', ha='left', bbox=dict(facecolor='green', alpha=0.5))
    plt.text(0.05, 0.65,
             f'Yeo-Johnson Transformed Skewness: {yeo_johnson_skewness:.2f}\nYeo-Johnson Transformed Kurtosis: {yeo_johnson_kurtosis:.2f}',
             fontsize=10, transform=plt.gca().transAxes, va='top', ha='left', bbox=dict(facecolor='red', alpha=0.5))
    plt.text(0.05, 0.55,
             f'Winsorized Skewness: {winsorized_skewness:.2f}\nWinsorized Kurtosis: {winsorized_kurtosis:.2f}',
             fontsize=10, transform=plt.gca().transAxes, va='top', ha='left', bbox=dict(facecolor='purple', alpha=0.5))

    # 添加图例和标签
    plt.legend()
    plt.title('Histogram of Transformed Series')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 显示图形
    plt.show()


def check_zscore_window_series(df, column, n_values=[50, 100, 200, 250, 300, 450, 600, 1200, 2400, 4800, 9600]):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    if column == 'c':
        # 计算原始对数收益率，此时 column 应为 NumPy ndarray
        log_return = (np.log(df[column]).diff(1).fillna(0)*1).shift(-1)
        # 将 NaN 值替换为0
        log_return = np.where(np.isnan(log_return), 0, log_return)
    else:
        log_return = df[column].values

    # 绘制原始对数收益率的直方图
    ax1.hist(log_return, bins=100, alpha=0.5,
             color='blue', label='Original Log Return')

    # 计算偏度和峰度
    skewness_orig = skew(log_return)
    kurtosis_orig = kurtosis(log_return)
    ax1.text(0.01, 0.9, f'Original Skew: {skewness_orig:.2f}, Kurtosis: {kurtosis_orig:.2f}',
             transform=ax1.transAxes, fontsize=10, color='blue')

    # 颜色生成器
    color_cycle = plt.cm.viridis(np.linspace(0, 1, len(n_values)))

    # 计算并绘制每个n值的滚动标准化对数收益率
    for n, color in zip(n_values, color_cycle):
        # rolling_mean = np.convolve(log_return, np.ones(n) / n, mode='valid')
        # # 填充使长度一致
        # rolling_mean = np.concatenate(
        #     (np.full(n - 1, np.nan), rolling_mean, np.full(len(log_return) - len(rolling_mean) - (n - 1), np.nan)))
        # rolling_std = np.sqrt(np.convolve((log_return - rolling_mean) ** 2, np.ones(n) / n, mode='valid'))
        # rolling_std = np.concatenate(
        #     (np.full(n - 1, np.nan), rolling_std, np.full(len(log_return) - len(rolling_std) - (n - 1), np.nan)))
        # norm_log_return = (log_return - rolling_mean) / rolling_std

        norm_log_return = _rolling_zscore_np(log_return, n)

        # 绘制直方图
        ax1.hist(norm_log_return, bins=100, alpha=0.5,
                 color=color, label=f'Norm Log Return n={n}')

        # 计算偏度和峰度
        skewness = skew(norm_log_return[~np.isnan(norm_log_return)])
        kurtosis_val = kurtosis(norm_log_return[~np.isnan(norm_log_return)])
        ax1.text(0.01, 0.8 - 0.07 * n_values.index(n),
                 f'n={n} Skew: {skewness:.2f}, Kurtosis: {kurtosis_val:.2f}',
                 transform=ax1.transAxes, fontsize=10, color=color)

    # 在第二个子图上设置两个y轴
    ax2_2 = ax2.twinx()
    for n, color in zip(n_values, color_cycle):
        norm_log_return = _rolling_zscore_np(log_return, n)
        ax2.plot(np.arange(len(df[column])), norm_log_return.cumsum(
        ), color=color, label=f'Cumulative Norm Log Ret n={n}')

    # 绘制原始数值
    ax2_2.plot(np.arange(len(df[column])), df[column], color='black',
               linewidth=2, label='Original Values', alpha=0.7)
    ax2.set_ylabel('Cumulative Norm Log Ret')
    ax2_2.set_ylabel('Original Values')

    ax1.set_title('Histogram of Log Returns and Normalized Log Returns')
    ax1.set_xlabel('Log Return Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax2.legend(loc='upper left')
    ax2_2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def _rolling_zscore(x1, n=300):  # 标准差标准化
    x1 = x1.flatten().astype(np.double)
    x1 = np.nan_to_num(x1)
    x1_rolling_avg = ta.SMA(x1, n)  # 使用TA-Lib中的简单移动平均函数SMA
    x_value = _DIVP(x1, ta.STDDEV(x1, n))
    # x_value = np.clip(x_value, -6, 6)
    return np.nan_to_num(x_value)


def _rolling_zscore_np(x1, n=300):  # 标准差标准化
    x = np.asarray(x1, dtype=np.float64)
    x1 = np.nan_to_num(x1)
    x1_rolling_avg = ta.SMA(x1, n)  # 使用TA-Lib中的简单移动平均函数SMA
    x_value = _DIVP(x1, ta.STDDEV(x1, n))
    # x_value = np.clip(x_value, -6, 6)
    return np.nan_to_num(x_value)


def _DIVP(x1, x2):  # 零分母保护的除法
    x1 = x1.flatten().astype(np.double)
    x2 = x2.flatten().astype(np.double)
    x = np.nan_to_num(np.where(x2 != 0, np.divide(x1, x2), 0))

    return x


def cal_ret(sym: str, freq: str, n: int) -> pd.Series:
    '''计算未来n个周期的收益率
    params
    sym:品种
    freq:降频周期
    n:第几个周期后的收益率'''
    z = data_load(sym)
    z = resample(z, freq)

    ret = (np.log(z.c).diff(n)*1).shift(-n)  # 计算对数收益率
    ret = np.where(np.isnan(ret), 0, ret)

    # 关键 - 对label进行了rolling_zscore处理！！
    ret_ = _rolling_zscore_np(ret, n)
    return ret_


def data_load_v2(sym: str = 'ETHUSDT', data_dir: str = '/Users/aming/data/ETHUSDT', start_date: str = '2025-01-01', end_date: str = '2025-12-31', 
                 timeframe: str = '1h', read_frequency: str = 'monthly',
                 file_path: Optional[str] = None) -> pd.DataFrame:
    """
    数据读取模块 V2 - 支持从多种时间粒度的数据文件读取
    参数:
    sym: 交易对符号，例如 'BTCUSDT'
    data_dir: 数据目录路径，例如 '/Volumes/Ext-Disk/data/futures/um/monthly/klines/BTCUSDT/1m'
    start_date: 起始日期
        - 月度格式: 'YYYY-MM' (如 '2020-01')
        - 日度格式: 'YYYY-MM-DD' (如 '2020-01-01')
    end_date: 结束日期，格式同上
    timeframe: 时间周期，默认 '1m'，可选 '5m', '1h' 等
    frequency: 数据频率，'monthly'（月度）或 'daily'（日度）
    file_path: 直接指定文件路径（支持 .feather / .zip / .csv），指定后将忽略其他参数
    
    返回:
    包含标准化列名的 DataFrame
    """

    # 解析频率参数
    try:
        freq_enum = DataFrequency(read_frequency.lower())
    except ValueError:
        raise ValueError(f"不支持的数据频率: {read_frequency}，仅支持 'monthly' 或 'daily'")
    
    # 生成日期范围
    date_list = _generate_date_range(start_date, end_date, freq_enum)
    
    # 读取所有时间段的数据
    df_list = []
    success_count = 0
    failed_count = 0
    
    for date_str in date_list:
        df = _read_single_period_data(sym, date_str, data_dir, timeframe, freq_enum)
        if df is not None:
            df_list.append(df)
            success_count += 1
        else:
            failed_count += 1
    
    # 检查是否成功读取到数据
    if not df_list:
        raise ValueError(f"未能成功读取任何数据文件，请检查路径和日期范围\n路径: {data_dir}\n日期: {start_date} ~ {end_date}")

    
    # 合并所有数据
    merged_df = pd.concat(df_list, ignore_index=True)
    
    # 标准化列名和索引
    standardized_df = _standardize_dataframe_columns(merged_df)
    
    print(f"数据时间范围: {standardized_df.index.min()} 至 {standardized_df.index.max()}")
    print(f"{'='*60}\n")
    
    return standardized_df

class DataFrequency(Enum):
    """数据频率枚举"""
    MONTHLY = 'monthly'  # 月度数据
    DAILY = 'daily'      # 日度数据


def _generate_date_range(start_date: str, end_date: str, read_frequency: DataFrequency = DataFrequency.MONTHLY) -> List[str]:
    
    if read_frequency == DataFrequency.MONTHLY:
        # 兼容 'YYYY-MM' 和 'YYYY-MM-DD' 两种格式 如果是 'YYYY-MM-DD' 格式，自动截取为 'YYYY-MM'
        new_start_date = start_date
        new_end_date = end_date
        if len(start_date) == 10:  # 'YYYY-MM-DD' 格式
            new_start_date = start_date[:7]
        if len(end_date) == 10:
            new_end_date = end_date[:7]
            
        start_dt = datetime.strptime(new_start_date, '%Y-%m')
        end_dt = datetime.strptime(new_end_date, '%Y-%m')
        
        date_list = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_list.append(current_dt.strftime('%Y-%m'))
            # 移动到下一个月
            if current_dt.month == 12:
                current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
            else:
                current_dt = current_dt.replace(month=current_dt.month + 1)
        
        return date_list
    
    elif read_frequency == DataFrequency.DAILY:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        date_list = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_list.append(current_dt.strftime('%Y-%m-%d'))
            current_dt += timedelta(days=1)
        
        return date_list
    
def _standardize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化 DataFrame 列名并设置索引
    
    参数:
    df: 原始 DataFrame（包含 Binance 格式的列名）
    
    返回:
    标准化后的 DataFrame
    """
    # 将 open_time 转换为 datetime 并设置为索引
    df = df.copy()
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)

    # df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    # df.set_index('close_time', inplace=True)
    
    # 列名映射：新列名 -> 旧列名
    # 新列名: open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore
    # 旧列名: o, h, l, c, vol, vol_ccy, trades, oi, oi_ccy, toptrader_count_lsr, toptrader_oi_lsr, count_lsr, taker_vol_lsr
    column_mapping = {
        'open': 'o',
        'high': 'h',
        'low': 'l',
        'close': 'c',
        'volume': 'vol',
        'quote_volume': 'vol_ccy',
        'count': 'trades',
        'close_time': 'close_time',
    }
    
    df = df.rename(columns=column_mapping)
    
    # 选择需要的列，对于缺失的列用 0 填充
    required_columns = [
                            'o', 'h', 'l', 'c', 
                            'vol', 
                            'vol_ccy', 
                            'trades',
                        #    'oi', 'oi_ccy', 'toptrader_count_lsr', 'toptrader_oi_lsr', 'count_lsr',
                        #    'taker_vol_lsr', 
                            'close_time', 
                            'taker_buy_volume', 
                            'taker_buy_quote_volume'
                       ]
    
    # 为缺失的列添加默认值 0
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
            print(f"⚠ 警告：列 '{col}' 不存在，已填充为 0")
    
    return df[required_columns]


def _read_single_period_data(sym: str, date_str: str, data_dir: str, timeframe: str = '1m',
                             frequency: DataFrequency = DataFrequency.MONTHLY) -> Optional[pd.DataFrame]:
    
    file_base_name, feather_path, zip_path = _build_file_paths(sym, date_str, data_dir, timeframe, frequency)
    
    # # 优先读取 feather
    df = _read_feather_file(feather_path)
    if df is not None:
        return df
    
    # 如果 feather 不存在，读取 zip
    df = _read_zip_file(zip_path, file_base_name, save_feather=True)
    if df is not None:
        return df
    
    # 两种文件都不存在
    print(f"⚠ 警告：文件不存在，跳过: {file_base_name}")
    return None

def _build_file_paths(sym: str, date_str: str, data_dir: str, timeframe: str = '1m', 
                      frequency: DataFrequency = DataFrequency.MONTHLY) -> Tuple[str, str, str]:
    """
    构建文件路径
    
    参数:
    sym: 交易对符号 ETHUSDT
    date_str: 日期字符串 2025-01
    data_dir: /Users/aming/data/ETHUSDT/15m
    timeframe: 时间周期 (如 '1m', '5m', '1h')
    frequency: 数据频率
    
    返回:
    (file_base_name, feather_path, zip_path) 元组
    """
    if frequency == DataFrequency.MONTHLY:
        file_base_name = f"{sym}-{timeframe}-{date_str}"
    elif frequency == DataFrequency.DAILY:
        file_base_name = f"{sym}-{timeframe}-{date_str}"
    
    # /Users/aming/data/ETHUSDT/15m/ 2025/ ETHUSDT-15m-2025-01.feather
    year = date_str.split('-')[0]
    feather_path = os.path.join(f'{data_dir}/{year}', f"{file_base_name}.feather")
    zip_path = os.path.join(f'{data_dir}/{year}', f"{file_base_name}.zip")
    
    return file_base_name, feather_path, zip_path


def _read_zip_file(zip_path: str, file_base_name: str, save_feather: bool = True) -> Optional[pd.DataFrame]:
    """
    读取 zip 格式文件（内含 CSV）
    
    参数:
    zip_path: zip 文件路径
    file_base_name: 文件基础名称（不含扩展名）
    save_feather: 是否保存为 feather 格式以加速后续读取
    
    返回:
    DataFrame 或 None（如果读取失败）
    """
    if not os.path.exists(zip_path):
        return None
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取 zip 中的 csv 文件名
            csv_filename = f"{file_base_name}.csv"
            
            # if csv_filename not in zip_ref.namelist():
            #     # 如果找不到，尝试使用第一个 csv 文件
            #     csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            #     if csv_files:
            #         csv_filename = csv_files[0]
            #     else:
            #         print(f"✗ 在 {os.path.basename(zip_path)} 中找不到 CSV 文件")
            #         return None
            
            # 读取 CSV 数据
            with zip_ref.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file)
                # print(f"✓ 成功读取 zip: {os.path.basename(zip_path)}, 行数: {len(df)}")
                
                # 可选：保存为 feather 格式以加速后续读取
                if save_feather:
                    feather_path = zip_path.replace('.zip', '.feather')
                    try:
                        df.to_feather(feather_path)
                        print(f"  → 已缓存为 feather 格式")
                    except Exception as e:
                        print(f"  → 保存 feather 文件失败: {str(e)}")
                
                return df
    
    except Exception as e:
        print(f"✗ 读取 zip 文件失败: {os.path.basename(zip_path)}, 错误: {str(e)}")
        return None
    
def _read_feather_file(feather_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(feather_path):
        return None
    
    try:
        df = pd.read_feather(feather_path)
        print(f"✓ 成功读取 feather: {os.path.basename(feather_path)}, 行数: {len(df)}")
        return df
    except Exception as e:
        print(f"✗ 读取 feather 文件失败: {os.path.basename(feather_path)}, 错误: {str(e)}")
        return None

def resample_with_offset(z: pd.DataFrame, resample_freq: str, offset: pd.Timedelta = None, 
                        closed: str = 'left', label: str = 'left') -> pd.DataFrame:
    '''
    支持offset参数的resample函数 - 使用pandas原生offset参数，避免时间索引偏移的问题
    
    参数:
        z: 输入的DataFrame，必须有DatetimeIndex
        freq: 重采样频率，如 '1h', '2h', '30min'
        offset: 偏移量（pd.Timedelta），用于调整分桶起点
                例如：offset=pd.Timedelta(minutes=15) 会让1小时桶从 9:15, 10:15, 11:15... 开始
        closed: 区间闭合方式，'left' 或 'right'
        label: 标签位置，'left' 或 'right'
    
    返回:
        重采样后的DataFrame
    '''
    if resample_freq == '15m':
        return z
    
    if resample_freq != '1min' and resample_freq != '1m':
        z.index = pd.to_datetime(z.index)
        
        # 使用pandas原生的offset参数，而不是偏移索引
        if offset is not None:
            z_resampled = z.resample(
                resample_freq, 
                closed=closed, 
                label=label,
                offset=offset  # 🔑 关键：使用pandas原生offset参数
            ).agg({
                'o': 'first',
                'h': 'max',
                'l': 'min',
                'c': 'last',
                'vol': 'sum',
                'vol_ccy': 'sum',
                'trades': 'sum',
            })
        else:
            # 没有offset时，使用原有逻辑
            z_resampled = z.resample(freq, closed=closed, label=label).agg({
                'o': 'first',
                'h': 'max',
                'l': 'min',
                'c': 'last',
                'vol': 'sum',
                'vol_ccy': 'sum',
                'trades': 'sum',
            })
        
        # 前向填充NaN值
        z_resampled = z_resampled.fillna(method='ffill')
        z_resampled.columns = ['o', 'h', 'l', 'c', 'vol', 'vol_ccy', 'trades']
        
        return z_resampled
    
    return z