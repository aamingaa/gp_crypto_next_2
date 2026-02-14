import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.special import erf
import time
import warnings
warnings.filterwarnings('ignore')

"""
NormDataCheck模块使用说明

功能描述：
    该模块用于验证和分析数据标准化(Normalization)处理的有效性，主要关注标准化周期对结果的影响。

主要验证内容：
1. 标准化周期(norm period)的选择评估
   - 分析不同标准化周期对结果的影响
   - 验证过小的标准化周期可能导致的结果符号改变问题
   - 评估结果符号改变的比例

使用方法：
1. 导入模块：
   from NormDataCheck import *

2. 数据准备：
   - 确保输入数据为pandas DataFrame格式
   - 数据应包含必要的时间索引和待分析的数值列

3. 参数设置：
   - 设置不同的标准化周期进行对比
   - 设定评估指标的阈值

4. 结果分析：
   - 观察不同周期下的标准化结果
   - 统计符号改变的比例
   - 选择最优的标准化周期

注意事项：
1. 标准化周期不宜过小，需要平衡周期长度和结果稳定性
2. 建议结合具体业务场景选择合适的评估指标
3. 数据预处理时注意处理异常值和缺失值

"""

def norm(data, window = 500, clip=6):
    """
    使用numpy的random模块
    """
    x = np.asarray(data)
    n = len(x)
    result = np.zeros(n)
    
    if n <= window:
        return result
    
    np.random.seed(42)  # 设置固定种子
    random_seq = np.random.random(n)
    for i in range(window, n):
        window_data = x[i-window:i]
        current_val = x[i]
        
        rank = 0
        ties = 0
        for j in range(window):
            if window_data[j] < current_val:
                rank += 1
            elif window_data[j] == current_val:
                ties += 1
        
        # 使用预生成的随机数
        if ties > 0:
            rank += random_seq[i] * (ties + 1)
        else:
            rank += random_seq[i]
            
        quantile = rank / (window + 1)
        
        q = quantile if quantile < 0.5 else (1.0 - quantile)
        sign = -1.0 if quantile < 0.5 else 1.0
        
        r = np.sqrt(-2.0 * np.log(q))
        r2 = r * r
        r3 = r2 * r
        
        numerator = 2.515517 + 0.802853*r + 0.010328*r2
        denominator = 1.0 + 1.432788*r + 0.189269*r2 + 0.001308*r3
        
        result[i] = sign * (r - numerator / denominator)
    
    return np.clip(result, -clip, clip)

def vectorized_gauss_rank_norm(data, window=500, clip=6):
    """
    向量化的高斯秩变换
    """
    s = pd.Series(data)
    
    # 1. 滚动排序 (pct=True 直接得到 0-1 之间的 quantile)
    # method='average' 处理平局，或者可以用 'random' 模拟你的抖动逻辑
    # 但为了完全复刻你的随机抖动，我们需要手动加噪
    
    # 加入微小随机噪声进行 Dithering (替代原本复杂的 random_seq 逻辑)
    # 1e-6 的噪声不会改变原本的大小关系，只会打乱绝对相等的值
    np.random.seed(42)
    noise = np.random.randn(len(s)) * 1e-9 
    s_noisy = s + noise
    
    # 计算滚动 Rank百分比
    # 注意：pandas 的 rolling rank 比较慢，可以用 numba 优化，但比纯 python 快
    # 这里 rank 后的值在 [1/N, 1.0] 之间
    rolling_rank = s_noisy.rolling(window=window).rank(pct=True)
    
    # 2. 避免无穷大 (Inf)
    # 如果 rank 是 1.0，norm.ppf 会是 Inf。我们需要将其限制在 (0, 1) 开区间
    # 你的代码用的是 rank / (window + 1)，这很聪明
    # 这里我们做一个修正
    rolling_rank = rolling_rank * (window / (window + 1)) + (1 / (window + 1)) * 0.5
    
    # 3. 逆正态变换 (Inverse CDF)
    # scipy.stats.norm.ppf 就是标准正态分布的逆函数 (Percent point function)
    result = norm.ppf(rolling_rank)
    
    # 4. 填充 NaN (前 window 个数据) 并 Clip
    result = np.nan_to_num(result, nan=0.0)
    return np.clip(result, -clip, clip)


def inverse_norm(normalized_data, original_data, window=2000):
    """
    将标准化后的数据转换回原始分布的近似值
   
    参数:
    - normalized_data: 标准化后的数据
    - original_data: 原始数据序列
    - window: 与norm函数使用的相同窗口大小
   
    返回:
    - 还原后的数据序列
    """
    x = np.asarray(normalized_data)
    orig = np.asarray(original_data)
    n = len(x)
    result = np.zeros(n)
   
    if n <= window:
        return result
   
    for i in range(window+1, n):
        # 1. 从标准正态分布值计算出分位数
        norm_val = x[i]
        if norm_val == 0:
            quantile = 0.5
        else:
            # 计算CDF
            cdf = 0.5 * (1 + erf(norm_val / np.sqrt(2)))
            quantile = cdf
           
        # 2. 根据分位数在当前窗口中找到对应的值
        window_data = orig[i-window-1:i-1]
        sorted_window = np.sort(window_data)
       
        # 计算在窗口中的位置
        pos = int(quantile * window)
        # print(pos, '======', i)
        pos = min(max(pos, 0), window-1)  # 确保位置在有效范围内
        # print(normalized_data[i], '=========', quantile, '======inverse_norm之后的结果', sorted_window[pos], '======原始的value', orig[i], '===========差额', (sorted_window[pos] - orig[i])/orig[i])
       
        result[i] = sorted_window[pos]
   
    return result

def verify_inverse_transform(original_data, normalized_data, window=2000, is_plot=False):
    """
    验证逆变换的准确性
    参数:
    - original_data: 原始收益率数据
    - normalized_data: 归一化后的数据
    - window: 窗口大小
    """
    # 跳过前window个数据点
    original = original_data[window:]
    normalized = normalized_data[window:]
    
    # 进行逆变换
    recovered_data = inverse_norm(normalized_data, original_data, window)
    recovered_data = recovered_data[window:]  # 跳过前window个点
    diff = pd.Series((original - recovered_data) / original).replace([np.nan, np.inf, -np.inf], 0).sum() / len(original)
    # 计算相关系数
    pearson_corr = np.corrcoef(original, recovered_data)[0,1]
    spearman_corr = pd.Series(original).corr(pd.Series(recovered_data), method='spearman')
    
    # print('===========orginal data')
    # print(original[-20:])
    # print('===========recovered data')
    # print(recovered_data[-20:])
    print(diff, '==逆变换的误差')
    print("\n逆变换验证结果:")
    print(f"原始数据与恢复数据的pearson相关系数: {pearson_corr:.4f}")
    print(f"原始数据与恢复数据的spearman相关系数: {spearman_corr:.4f}")
    
    if is_plot:
        # 可视化
        plt.figure(figsize=(15, 10))
        
        # 1. 原始数据vs标准化数据的散点图
        plt.subplot(221)
        plt.scatter(original, normalized, alpha=0.5, c='blue', s=1)
        plt.xlabel('Raw Retuen')
        plt.ylabel('Normalized value')
        plt.title('rawdata vs normed data')
        plt.grid(True)
        
        # 2. 原始数据vs恢复数据的散点图
        plt.subplot(222)
        plt.scatter(original, recovered_data, alpha=0.5, c='red', s=1)
        plt.xlabel('raw return')
        plt.ylabel('recovered return')
        plt.title('rawdata vs recoverd data')
        plt.grid(True)
        
        # 3. 数据分布对比
        plt.subplot(223)
        plt.hist(original, bins=100, alpha=0.5, label='raw data', density=True, color='blue')
        plt.hist(recovered_data, bins=100, alpha=0.5, label='recovered data', density=True, color='red')
        plt.xlabel('values')
        plt.ylabel('density')
        plt.title('comparision of values')
        plt.legend()
        plt.grid(True)
        
        # 4. 标准化数据的分布
        plt.subplot(224)
        plt.hist(normalized, bins=100, density=True, alpha=0.7, color='green')
        plt.xlabel('normalized values')
        plt.ylabel('density')
        plt.title('normalized distribution')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def check_quantile_consistency(original_data, normalized_data, window=2000, is_plot=False):
    """
    检查原始数据和归一化后数据的分位数对应关系
    """
    # 去掉前window个数据点
    original = original_data[window:]
    normalized = normalized_data[window:]
    
    # 计算分位数点
    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    # 创建对比DataFrame
    comparison = pd.DataFrame({
        'Quantile': quantiles,
        'Original': np.quantile(original, quantiles),
        'Normalized': np.quantile(normalized, quantiles)
    })
    
    # 检查单调性
    orig_sorted = np.sort(original)
    norm_sorted = np.sort(normalized)
    rank_correlation = np.corrcoef(orig_sorted, norm_sorted)[0,1]
    
    print("quantile consistency check中orginal和normalized结果对比：")
    print(comparison)
    print(f"\n分位数单调性对比_采用分位数的秩相关系数：: {rank_correlation:.4f}")
    
    # 分别检查正负值的分位数
    pos_mask = original > 0
    neg_mask = original < 0
    
    # print("\n正值分位数：")
    # print(pd.DataFrame({
    #     'Quantile': quantiles,
    #     'Original': np.quantile(original[pos_mask], quantiles),
    #     'Normalized': np.quantile(normalized[pos_mask], quantiles)
    # }))
    
    # print("\n负值分位数：")
    # print(pd.DataFrame({
    #     'Quantile': quantiles,
    #     'Original': np.quantile(original[neg_mask], quantiles),
    #     'Normalized': np.quantile(normalized[neg_mask], quantiles)
    # }))
    
    if is_plot:
        # 可视化
        plt.figure(figsize=(15, 5))
        
        # QQ图
        plt.subplot(121)
        plt.scatter(orig_sorted, norm_sorted, alpha=0.5)
        plt.xlabel('rawdata quantiles')
        plt.ylabel('normdata quantiles')
        plt.title('QQ plot')
        
        # 分布对比
        plt.subplot(122)
        plt.hist(original, bins=50, alpha=0.5, label='rawdata', density=True)
        plt.hist(normalized, bins=50, alpha=0.5, label='normed data', density=True)
        plt.legend()
        plt.title('distributions')
        plt.tight_layout()
        plt.show()

def evaluate_position_performance(data, label_col, window=2000, fee_rate=0.0004, is_plot=False):
    """
    评估标准化后的label作为仓位的表现
    增加is_plot参数控制是否绘图
    """
    results = {}
    eval_data = data.copy()
    eval_data['position'] = eval_data[label_col].fillna(0)
    eval_data['position'] = np.where(eval_data['position'] > 0, 1, -1)
    eval_data['raw_pnl'] = eval_data['position'] * eval_data['ret']

    eval_data['position_change'] = eval_data['position'].diff().abs()
    eval_data['fee_cost'] = eval_data['position_change'] * fee_rate
    eval_data['net_pnl'] = eval_data['raw_pnl'] - eval_data['fee_cost']

    # 跳过前window个数据进行评估
    eval_period = eval_data.iloc[window:]
    
    # 计算累积收益和回撤
    cumulative_returns = 1 + eval_period['net_pnl'].cumsum()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns / rolling_max - 1

    if is_plot:
        # 降采样，减少绘图点数
        sample_size = min(1000, len(cumulative_returns))
        plot_idx = np.linspace(0, len(cumulative_returns)-1, sample_size, dtype=int)
        
        plt.figure(figsize=(15, 8))
        
        # 上半部分：累积收益曲线
        plt.subplot(2, 1, 1)
        plt.plot(cumulative_returns.index[plot_idx], cumulative_returns.iloc[plot_idx], 
                label='累积收益', color='blue')
        plt.plot(rolling_max.index[plot_idx], rolling_max.iloc[plot_idx], 
                label='历史最高点', color='red', linestyle='--', alpha=0.5)
        plt.title('累积收益曲线')
        plt.legend()
        plt.grid(True)
        
        # 下半部分：回撤曲线
        plt.subplot(2, 1, 2)
        plt.fill_between(drawdowns.index[plot_idx], drawdowns.iloc[plot_idx], 0, 
                        color='red', alpha=0.3)
        plt.plot(drawdowns.index[plot_idx], drawdowns.iloc[plot_idx], 
                color='red', label='回撤')
        plt.title('回撤曲线')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    hours_per_year = 24 * 365  # 一年的小时数
    n_years = len(eval_period) / hours_per_year  # 数据跨越的年数

    # 计算评估指标
    results['原始总收益率'] = eval_period['ret'].abs().sum()
    results['总收益率'] = eval_period['net_pnl'].sum()
    results['年化收益率'] = eval_period['net_pnl'].sum() / n_years  # 总收益除以年数
    results['年化夏普比率'] = (eval_period['net_pnl'].mean() / eval_period['net_pnl'].std() * np.sqrt(24 * 365))
    results['最大回撤'] = drawdowns.min()
    results['胜率'] = (eval_period['net_pnl'] > 0).mean()
    results['换仓成本总计'] = eval_period['fee_cost'].sum()
    results['方向准确率'] = (np.sign(eval_period['position']) == np.sign(eval_period['ret'])).mean()
    results['仓位波动率'] = eval_period['position'].std()
    results['仓位翻转频率'] = (eval_period['position'].diff().abs() > 1).mean()
    results['年化收益-回撤比'] = results['年化收益率'] / abs(results['最大回撤'])
    results['总收益-手续费比'] = results['总收益率'] / results['换仓成本总计']
    
    return results


def _generate_date_range(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m')
    end = datetime.strptime(end_date, '%Y-%m')
    date_list = []
    current_dt = start
    while current_dt <= end:
        date_list.append(current_dt.strftime('%Y-%m'))
        # current += timedelta(days=1)
        if current_dt.month == 12:
                current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
        else:
                current_dt = current_dt.replace(month=current_dt.month + 1)
    return date_list


# start = time.time()
# # file_path = 'H:/QuantofKL/CryptosFuturesModel/raw_data/DOGEUSDT_1h_250108.pkl'
# start_date = '2025-01'
# end_date = '2025-03'
# df_list = []
# date_list = _generate_date_range(start_date, end_date)
# raw_data = []
# for date in date_list:
#     file_path = f'/Volumes/Ext-Disk/data/futures/um/monthly/klines/ETHUSDT/1h/2025/ETHUSDT-1h-{date}.zip'
#     df = pd.read_csv(file_path)
#     # df['date'] = date
#     df_list.append(df)
#     print(f'df{len(df)}, read df_list {len(df_list)} done')

# raw_data = pd.concat(df_list)
# # raw_data = pd.read_csv(file_path)
# raw_data['ret'] = (raw_data['close'].shift(-1) / raw_data['close'] - 1).fillna(0)
# # raw_data['ret'] = (raw_data['close'] / raw_data['close'].shift(1) - 1).fillna(0)
# range_start = 5
# range_end = 15

# window_list = [100, 200, 300, 500, 800, 1000, 1200, 1500, 1800, 2000, 2500, 3000]
# window_list = [100, 500, 1000, 2000]
# window_list = [100, 200, 300, 500]
# for window in window_list:
#     for i in range(range_start, range_end, 2):
#         print(f'Start of step {i} at window {window}')
#         raw_data[f'ret_{i}'] = (raw_data['close'].shift(-i) / raw_data['close'] - 1).fillna(0)
#         print(f'mean of term {i} with norm window of {window} cumsum of returns')
#         print(f'mean of term {i} with norm window of {window} absolute return values is {raw_data[f"ret_{i}"].abs().mean()}')
#         print(raw_data[f"ret_{i}"].describe())
#         raw_data[f'ret_{i}_norm'] = norm(raw_data[f'ret_{i}'].values, window=window, clip=6)


#         print(f"\n检查 ret_{i} 且norm 的window为{window}的逆变换效果:")
#         print('1==Norm之后的变号状况及变号部分return的分布')
#         raw_data[f'ret_{i}_direct_inequals'] = np.where(raw_data[f'ret_{i}'] * raw_data[f'ret_{i}_norm'] < 0, 0, 1)
#         print(f'step为{i}且window为{window}时，直接变号比例为',  1 - raw_data[f'ret_{i}_direct_inequals'].sum()/len(raw_data), '形状参数如下：')
#         print(raw_data[raw_data[f'ret_{i}_direct_inequals'] == 0][f'ret'].describe())

#         verify_inverse_transform(
#             raw_data[f'ret_{i}'].values,
#             raw_data[f'ret_{i}_norm'].values,
#             window=2000,
#             is_plot=False
#         )

#         check_quantile_consistency(
#             raw_data[f'ret_{i}'].values,
#             raw_data[f'ret_{i}_norm'].values,
#             window=2000,
#             is_plot=False
#         )

#         print(f"\n评估 ret_{i}_norm且window为{window}的表现:")
#         performance = evaluate_position_performance(
#             raw_data, 
#             f'ret_{i}_norm',
#             window=window,
#             is_plot=False
#         )
        
#         for metric, value in performance.items():
#             print(f"{metric}: {value:.4f}")
#         print(f'End of step {i} at window {window} \n\n')

# print(f'time cost :{time.time() - start}S')
