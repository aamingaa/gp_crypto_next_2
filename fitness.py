"""Metrics to evaluate the fitness of a program.

The :mod:`gplearn.fitness` module contains some metric with which to evaluate
the computer programs created by the :mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause


import numbers
import numpy as np
from joblib import wrap_non_picklable_objects
from scipy.stats import rankdata,pearsonr,spearmanr, wasserstein_distance
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
__all__ = ['make_fitness']




def calculate_annual_bars(freq: str) -> int:
    # 创建一个 pandas 定时器（timestamp），并获取 freq 参数所表示的时间间隔
    freq_timedelta = pd.to_timedelta(freq)
    
    # 24小时等于86400秒
    hours_in_a_day = pd.Timedelta(hours=24)
    
    # 计算 24 小时内包含多少bar
    multiples_of_freq = hours_in_a_day // freq_timedelta
    
    #计算年化多少bar
    annual_bars = 365 * multiples_of_freq
    
    return annual_bars


# freq 频率对应的参数
rolling_w = 2000
freq = '30min'
x_clip = 20
y_clip = 0.2
times_per_year = calculate_annual_bars(freq)




class _Fitness(object):

    """A metric to measure the fitness of a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting floating point score quantifying the quality of the program's
    representation of the true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """

    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args):
        return self.function(*args)


def make_fitness(*, function, greater_is_better, wrap=True):
    """Make a fitness measure, a metric scoring the quality of a program's fit.

    This factory function creates a fitness measure object which measures the
    quality of a program's fit and thus its likelihood to undergo genetic
    operations into the next generation. The resulting object is able to be
    called with NumPy vectorized arguments and return a resulting floating
    point score quantifying the quality of the program's representation of the
    true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom metrics is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """

    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'
                         % type(greater_is_better))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))
    if function.__code__.co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w),'
                         ' got %d.' % function.__code__.co_argcount)
    if not isinstance(function(np.array([1, 1]),
                      np.array([2, 2]),
                      np.array([1, 1])), numbers.Number):
        raise ValueError('function must return a numeric.')

    if wrap:
        return _Fitness(function=wrap_non_picklable_objects(function),
                        greater_is_better=greater_is_better)
    return _Fitness(function=function,
                    greater_is_better=greater_is_better)





# --------------------------需要norm_y的评价函数-----------------------------------


def _weighted_pearson(y, y_pred, w):
    """Calculate the weighted Pearson corr coefficient."""
    with np.errstate(divide='ignore', invalid='ignore'):
        y_pred_demean = y_pred - np.average(y_pred, weights=w)
        y_demean = y - np.average(y, weights=w)
        corr = ((np.sum(w * y_pred_demean * y_demean) / np.sum(w)) /
                np.sqrt((np.sum(w * y_pred_demean ** 2) *
                         np.sum(w * y_demean ** 2)) /
                        (np.sum(w) ** 2)))
        
    if np.isfinite(corr):
        return np.abs(corr)
    return 0.


def _weighted_spearman(y, y_pred, w):
    """Calculate the weighted Spearman corr coefficient."""
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y)
    
    return _weighted_pearson(y_pred_ranked, y_ranked, w)


def _mean_absolute_error(y, y_pred, w):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=w)


def _mean_square_error(y, y_pred, w):
    """Calculate the mean square error."""
    return np.average(((y_pred - y) ** 2), weights=w)


def _root_mean_square_error(y, y_pred, w):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))


def _log_loss(y, y_pred, w):
    """Calculate the log loss."""
    eps = 1e-15
    inv_y_pred = np.clip(1 - y_pred, eps, 1 - eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    score = y * np.log(y_pred) + (1 - y) * np.log(inv_y_pred)
    
    return np.average(-score, weights=w)


def _discretize_by_quantile(values, n_bins=16):
    """Discretize a 1D array into quantile bins."""
    arr = np.nan_to_num(values).flatten().astype(float)
    if arr.size == 0:
        return np.array([], dtype=int)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(arr, quantiles)
    edges = np.unique(edges)
    if edges.size <= 2:
        return np.zeros(arr.shape[0], dtype=int)

    # digitize returns [1, n_bins], convert to [0, n_bins-1]
    bins = np.digitize(arr, edges[1:-1], right=True)
    return bins.astype(int)


def _mutual_information_from_bins(x_bin, y_bin):
    """Compute MI and entropies from two discrete arrays."""
    if x_bin.size == 0 or y_bin.size == 0 or x_bin.size != y_bin.size:
        return 0.0, 0.0, 0.0

    x_bin = x_bin.astype(int)
    y_bin = y_bin.astype(int)
    n_x = int(np.max(x_bin)) + 1
    n_y = int(np.max(y_bin)) + 1
    if n_x <= 0 or n_y <= 0:
        return 0.0, 0.0, 0.0

    joint_counts = np.zeros((n_x, n_y), dtype=float)
    np.add.at(joint_counts, (x_bin, y_bin), 1.0)
    total = joint_counts.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0

    p_xy = joint_counts / total
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    eps = 1e-12
    h_x = -np.sum(p_x * np.log(p_x + eps))
    h_y = -np.sum(p_y * np.log(p_y + eps))

    denominator = (p_x[:, None] * p_y[None, :]) + eps
    mi = np.sum(p_xy * np.log((p_xy + eps) / denominator))
    return float(mi), float(h_x), float(h_y)


def _calculate_mi_score(y, y_pred, w, n_bins=16):
    """
    Normalized mutual information score in [0, 1].
    Higher is better.
    """
    y_pred = np.nan_to_num(y_pred).flatten()
    y = np.nan_to_num(y).flatten()
    if len(y) != len(y_pred) or len(y) < 20:
        return 0.0

    x_bin = _discretize_by_quantile(y_pred, n_bins=n_bins)
    y_bin = _discretize_by_quantile(y, n_bins=n_bins)
    mi, h_x, h_y = _mutual_information_from_bins(x_bin, y_bin)
    norm = np.sqrt(max(h_x, 0.0) * max(h_y, 0.0))
    if norm <= 1e-12:
        return 0.0

    score = mi / norm
    return float(np.clip(score, 0.0, 1.0))


def _calculate_entropy_gain_score(y, y_pred, w, n_bins=16):
    """
    Conditional-entropy reduction ratio in [0, 1].
    Equivalent to MI / H(Y): larger means stronger uncertainty reduction.
    """
    y_pred = np.nan_to_num(y_pred).flatten()
    y = np.nan_to_num(y).flatten()
    if len(y) != len(y_pred) or len(y) < 20:
        return 0.0

    x_bin = _discretize_by_quantile(y_pred, n_bins=n_bins)
    y_bin = _discretize_by_quantile(y, n_bins=n_bins)
    mi, _h_x, h_y = _mutual_information_from_bins(x_bin, y_bin)
    if h_y <= 1e-12:
        return 0.0

    score = mi / h_y
    return float(np.clip(score, 0.0, 1.0))


def _calculate_average_pic(y, y_pred, w, n_chunk = 5):
    # 确保x和y长度相同
    y_pred = np.nan_to_num(y_pred).flatten()
    y = np.nan_to_num(y).flatten()
    assert len(y) == len(y_pred), "x and y must have the same length"

    # 使用np.split切分x和y
    x_segments = np.array_split(y_pred, n_chunk)
    y_segments = np.array_split(y, n_chunk)

    # 计算每个段的IC（Spearman相关系数）
    ics = [pearsonr(x_seg, y_seg)[0] for x_seg, y_seg in zip(x_segments, y_segments)]

    # 计算平均IC
    average_ic = np.mean(ics)

    return average_ic

def _calculate_average_sic(y, y_pred, w, n_chunk = 5):
    # 确保x和y长度相同
    y_pred = np.nan_to_num(y_pred).flatten()
    y = np.nan_to_num(y).flatten()
    assert len(y) == len(y_pred), "x and y must have the same length"

    # 使用np.split切分x和y
    x_segments = np.array_split(y_pred, n_chunk)
    y_segments = np.array_split(y, n_chunk)

    # 计算每个段的IC（Spearman相关系数）
    ics = [spearmanr(x_seg, y_seg)[0] for x_seg, y_seg in zip(x_segments, y_segments)]

    # 计算平均IC
    average_ic = np.mean(ics)

    return average_ic

def _calculate_max_ic_chunk(y, y_pred, w, n_chunk = 5):
    # 确保x和y长度相同
    y_pred = np.nan_to_num(y_pred).flatten()
    y = np.nan_to_num(y).flatten()
    assert len(y) == len(y_pred), "x and y must have the same length"

    # 将x和y根据x的大小进行排序
    sorted_indices = np.argsort(y_pred)
    sorted_x = y_pred[sorted_indices]
    sorted_y = y[sorted_indices]

    # 计算分割点，将数据分为n组
    n = int(n_chunk)  # 确保n是整数
    split_indices = np.array_split(np.arange(len(y_pred)), n)

    # 按照排序后的数据计算每组的相关性
    group_ics = []
    
    for indices in split_indices:
        if len(indices) > 1:  # 至少需要两个数据点来计算相关性
            x_group = sorted_x[indices]
            y_group = sorted_y[indices]
            corr, _ = spearmanr(x_group, y_group)
        else:
            corr = np.nan  # 如果数据量太小无法计算，返回NaN
        group_ics.append(corr)

    # 计算最大IC
    arr = np.array(group_ics)
    # 找到最大值的位置
    max_index = np.argmax(arr) 
    # 将最大值设为一个较小的值，以便找到第二大值
    arr[max_index] = -np.inf    
    # 找到第二大值的位置，并转换为记录的数据
    second_max_index = np.argmax(arr)
    mid_index = n // 2
    # 判断最大值和第二大值的位置是否关于中点位置对称
    is_symmetric = int(((max_index + second_max_index) <= 2 * mid_index + 1) & ((max_index + second_max_index) >= 2 * mid_index - 1))

    max_ic = round(np.nanmax(group_ics), 4)
    #添加最大值位置信息
    max_index = max_index + 1
    max_ic = max_ic + max_index*1e-6
    #添加对称信息
    max_ic = max_ic + is_symmetric*1e-8
    #添加分层数信息
    max_ic = max_ic + n_chunk*1e-7
   
    return max_ic

###

def _calculate_max_ic_chunk_train(y, y_pred, w, n_chunk = 5):
    # 确保x和y长度相同
    y_pred = np.nan_to_num(y_pred).flatten()
    y = np.nan_to_num(y).flatten()
    assert len(y) == len(y_pred), "x and y must have the same length"

    # 将x和y根据x的大小进行排序
    sorted_indices = np.argsort(y_pred)
    sorted_x = y_pred[sorted_indices]
    sorted_y = y[sorted_indices]

    # 计算分割点，将数据分为n组
    n = int(n_chunk)  # 确保n是整数
    split_indices = np.array_split(np.arange(len(y_pred)), n)

    # 按照排序后的数据计算每组的相关性
    group_ics = []
    up_xs = []
    dn_xs = []
    
    for indices in split_indices:
        if len(indices) > 1:  # 至少需要两个数据点来计算相关性
            x_group = sorted_x[indices]
            y_group = sorted_y[indices]
            corr, _ = spearmanr(x_group, y_group)
            up_x = np.max(x_group)
            dn_x = np.min(x_group)
        else:
            corr = np.nan  # 如果数据量太小无法计算，返回NaN
            up_x = np.nan
            dn_x = np.nan
        group_ics.append(corr)
        up_xs.append(up_x)
        dn_xs.append(dn_x)


    # 计算最大IC
    arr = np.array(group_ics)
    # 找到最大值的位置
    max_index = np.argmax(arr) 
    up_r = up_xs[max_index]
    dn_r = dn_xs[max_index]
    # 将最大值设为一个较小的值，以便找到第二大值
    arr[max_index] = -np.inf    
    # 找到第二大值的位置，并转换为记录的数据
    second_max_index = np.argmax(arr)
    mid_index = n // 2
    # 判断最大值和第二大值的位置是否关于中点位置对称
    is_symmetric = int(((max_index + second_max_index) <= 2 * mid_index + 1) & ((max_index + second_max_index) >= 2 * mid_index - 1))

    max_ic = round(np.nanmax(group_ics), 4)
    #添加最大值位置信息
    max_index = max_index + 1
    max_ic = max_ic + max_index*1e-6
    #添加对称信息
    max_ic = max_ic + is_symmetric*1e-8
    #添加分层数信息
    max_ic = max_ic + n_chunk*1e-7
   
    return max_ic, up_r, dn_r 


def _calculate_given_range_ic(y, y_pred, w, up_r, dn_r, method='spearman'):
    # 确保x和y长度相同
    y_pred = np.nan_to_num(y_pred).flatten()
    y = np.nan_to_num(y).flatten()
    assert len(y) == len(y_pred), "x and y must have the same length"
    
    mask = (y_pred >= dn_r) & (y_pred <= up_r)
    x_selected = y_pred[mask]
    y_selected = y[mask]

    # 检查是否有足够数据计算相关性
    if len(x_selected) > 20:
    # 计算相关性
        if method == 'pearson':
            corr, _ = pearsonr(x_selected, y_selected)
        elif method == 'spearman':
            corr, _ = spearmanr(x_selected, y_selected)
    else:
        corr = np.nan
   
    return corr

def _risk_constrained_chunk(y, y_pred, w, n_chunk=5):
    """
    适配时序因子挖掘的风险约束fitness函数
    核心逻辑：fitness = 时序预测收益能力 - 加权风险惩罚（越大越好）
    输入：
        y: 一维数组，真实的时序收益率（t+1时刻）
        y_pred: 一维数组，因子值（t时刻），即时序预测信号
        w: 样本权重，默认不用
    输出：
        标量fitness得分，越大越好（无效因子返回-∞）
    """

    # -------------------------- 1. 时序收益基础项（核心） --------------------------
    # 时序因子核心评价：方向准确率 + 收益贡献（兼顾胜率和盈亏比）
    # a. 方向准确率：预测方向与实际收益方向一致的比例
    dir_correct = np.sum((y_pred > 0) == (y > 0)) / len(y)
    # b. 收益贡献：预测值与真实收益的相关系数（放大正确方向的收益权重）
    corr, _ = pearsonr(y_pred, y)
    corr = 0 if np.isnan(corr) else corr
    # c. 基础得分：方向准确率*（1+相关系数），归一化到[0,1]
    base_score = dir_correct * (1 + abs(corr)) / 2  # 范围[0,1]

    # -------------------------- 2. 时序分布一致性惩罚pen_EMD（跨窗口稳定性） --------------------------
    # 适配逻辑：将时间序列切分为3个滚动窗口，用EMD衡量因子预测分布的差异（差异大=惩罚重）
    try:
        # 切分滚动窗口（保证每个窗口至少10个样本）
        np.array_split(y_pred, n_chunk)

        y_pre_segments = np.array_split(y_pred, n_chunk)
        y_segments = np.array_split(y, n_chunk)

        emd_list = []

        for y_pre_segment, y_segment in zip(y_pre_segments, y_segments):
            emd = wasserstein_distance(y_pre_segment, y_segment)
            emd = 0 if np.isnan(emd) else emd
            emd_list.append(emd)

        emd_list_np = np.array(emd_list)
        avg_emd = np.nanmean(emd_list_np)
        
        # 归一化：用因子值的全距作为最大EMD，将惩罚归一到[0,1]
        emd_max = np.ptp(y_pred)  # 全距=最大值-最小值
        pen_EMD = avg_emd / emd_max if emd_max > 1e-8 else 1.0
    except:
        pen_EMD = 1.0  # 计算异常则满额惩罚

    # -------------------------- 3. 时序尾部风险惩罚pen_tail（极端时段失效） --------------------------
    # a. 定义时序极端行情窗口
    extreme_thresh = 0.02  # 涨/跌超2%视为极端行情
    extreme_mask = (y > extreme_thresh) | (y < -extreme_thresh)
    
    if extreme_mask.sum() < 5:  # 极端样本不足，不惩罚
        pen_tail = 0.0
    else:
        # 极端行情下的方向准确率（越低，惩罚越重）
        extreme_dir_correct = np.sum((y_pred[extreme_mask] > 0) == (y[extreme_mask] > 0)) / extreme_mask.sum()
        # 平方项放大极端失效的惩罚，归一到[0,1]
        pen_tail = (1 - extreme_dir_correct) ** 2
        
    # -------------------------- 4. 时序回撤惩罚pen_MDD（持仓净值回撤） --------------------------
    # 适配逻辑：模拟因子时序持仓的净值曲线，计算最大回撤（回撤大=惩罚重）
    try:
        # 时序持仓逻辑：因子值>0做多，<0做空，仓位大小=因子值归一化后的绝对值
        norm_pred = y_pred / np.max(np.abs(y_pred))  # 仓位归一到[-1,1]
        portfolio_return = norm_pred * y  # 每日持仓收益=仓位*实际收益率
        # 计算净值曲线和最大回撤
        net_value = np.cumprod(1 + portfolio_return)
        peak = np.maximum.accumulate(net_value)  # 滚动峰值
        drawdown = (peak - net_value) / peak  # 回撤率
        max_dd = drawdown.max()
        pen_MDD = min(max_dd, 1.0)  # 归一到[0,1]
    except:
        pen_MDD = 1.0  # 计算异常则满额惩罚

    # -------------------------- 5. 融合所有项，输出最终fitness --------------------------
    # 时序场景的惩罚权重（可根据需求调整）
    w_EMD = 0.25   # 跨窗口分布稳定性（时序核心约束）
    w_tail = 0.25  # 极端时段失效惩罚
    w_MDD = 0.2    # 时序回撤惩罚

    # 总fitness = 基础收益项 - 加权总惩罚（保证非负，无效因子返回-∞）
    total_penalty = w_EMD * pen_EMD + w_tail * pen_tail + w_MDD * pen_MDD
    final_fitness = base_score - total_penalty

    # 边界处理：fitness<0视为无效因子，直接淘汰
    final_fitness = final_fitness if final_fitness > 0 else -np.inf
    return final_fitness


# --------------------------需要原始y的评价函数-----------------------------------


def _cal_pnl(y,y_pred,w):
    
    y_pred = np.nan_to_num(y_pred).flatten()
    y = np.nan_to_num(y).flatten()
    y = y.clip(-y_clip,y_clip)
    y_pred = y_pred.clip(-x_clip, x_clip)
    pnl = (y*y_pred*w).cumsum()
    
    return pnl


def _cal_rets(y,y_pred,w):
    
    y_pred = np.nan_to_num(y_pred).flatten()
    y = np.nan_to_num(y).flatten()
    y = y.clip(-y_clip,y_clip)
    y_pred = y_pred.clip(-x_clip, x_clip)
    pnl = y*y_pred*w
    
    return pnl


def _prepare_margin_inputs(y, y_pred, w, clip_low_q=0.01, clip_high_q=0.99):
    """Prepare arrays for margin-based directional fitness."""
    y_pred = np.nan_to_num(y_pred).flatten().astype(float)
    y = np.nan_to_num(y).flatten().astype(float)
    w = np.nan_to_num(w).flatten().astype(float)

    if len(y_pred) != len(y):
        raise ValueError("y and y_pred must have the same length")
    if len(w) != len(y):
        w = np.ones_like(y, dtype=float)

    w = np.where(w > 0, w, 0.0)
    if np.sum(w) <= 0:
        w = np.ones_like(y, dtype=float)

    # clip only inside loss/fitness to stabilize extreme outliers
    low = np.nanquantile(y, clip_low_q)
    high = np.nanquantile(y, clip_high_q)
    y_for_loss = np.clip(y, low, high)

    return y, y_for_loss, y_pred, w


def _margin_and_extreme_weight(y_for_loss, y_pred, extreme_q=0.95):
    """Compute margin and emphasize extreme absolute-return samples."""
    margin = y_pred * y_for_loss
    abs_y = np.abs(y_for_loss)
    extreme_thr = np.nanquantile(abs_y, extreme_q)
    is_extreme = abs_y >= extreme_thr

    scale = np.nanquantile(abs_y, extreme_q) + 1e-12
    extreme_weight = np.where(is_extreme, 1.0 + abs_y / scale, 1.0)

    return margin, extreme_weight, is_extreme


def _directional_reward(y, y_pred, w):
    """Directional PnL-like reward, higher is better."""
    pos = np.sign(np.nan_to_num(y_pred))
    return np.average(y * pos, weights=w)


def _extreme_wrong_sign_penalty(y, y_pred, w, lam=1.0, extreme_q=0.95):
    """
    Score = directional reward - lambda * weighted wrong-sign penalty.
    """
    y_raw, y_for_loss, y_pred, w = _prepare_margin_inputs(y, y_pred, w)
    margin, extreme_weight, is_extreme = _margin_and_extreme_weight(
        y_for_loss, y_pred, extreme_q=extreme_q
    )

    wrong_sign = (margin < 0).astype(float)
    penalty_t = wrong_sign * extreme_weight * is_extreme.astype(float)
    penalty = np.average(penalty_t, weights=w)
    reward = _directional_reward(y_raw, y_pred, w)

    return reward - lam * penalty


def _weighted_margin_score(y, y_pred, w, lam=1.0, extreme_q=0.95):
    """
    Score = directional reward - lambda * weighted negative-margin loss.
    """
    y_raw, y_for_loss, y_pred, w = _prepare_margin_inputs(y, y_pred, w)
    margin, extreme_weight, _ = _margin_and_extreme_weight(
        y_for_loss, y_pred, extreme_q=extreme_q
    )

    penalty_t = extreme_weight * np.maximum(0.0, -margin)
    penalty = np.average(penalty_t, weights=w)
    reward = _directional_reward(y_raw, y_pred, w)

    return reward - lam * penalty


def _asym_extreme_direction_score(y, y_pred, w, a=3.0, b=0.3, m0=0.2, lam=1.0, extreme_q=0.95):
    """
    Asymmetric margin score with hard wrong-sign penalty and soft-margin penalty.
    """
    y_raw, y_for_loss, y_pred, w = _prepare_margin_inputs(y, y_pred, w)
    margin, extreme_weight, _ = _margin_and_extreme_weight(
        y_for_loss, y_pred, extreme_q=extreme_q
    )

    hard_wrong = np.maximum(0.0, -margin)
    soft_margin = np.maximum(0.0, m0 - margin)
    penalty_t = extreme_weight * (a * hard_wrong + b * soft_margin)
    penalty = np.average(penalty_t, weights=w)
    reward = _directional_reward(y_raw, y_pred, w)

    return reward - lam * penalty

def _calculate_max_drawdown(y,y_pred,w):
    
    pnl = _cal_pnl(y,y_pred,w)
    peak_values = np.maximum.accumulate(pnl)
    drawdowns = pnl - peak_values

    return np.min(drawdowns)

def _calculate_average_max_drawdown(y,y_pred,w,chunks_n = 5):
    def cal_max_dd(s):
        peak_values = np.maximum.accumulate(s)
        drawdowns = s - peak_values
        return np.min(drawdowns)

    pnl = _cal_pnl(y,y_pred,w)
    chunks = np.array_split(pnl, chunks_n)

    # 计算每段的最大回撤
    max_drawdowns = [cal_max_dd(segment) for segment in chunks]

    # 计算平均最大回撤
    average_max_drawdown = np.mean(max_drawdowns)

    return average_max_drawdown


def _calculate_calmar_ratio(y,y_pred,w, periods_per_year=times_per_year):
    
    rets = _cal_rets(y,y_pred,w)    
    annual_return = np.nanmean(rets) * periods_per_year    
    max_drawdown = _calculate_max_drawdown(y, y_pred, w)
    calmar_ratio = annual_return / -max_drawdown if max_drawdown != 0 else np.inf
    
    return calmar_ratio


def _calculate_sharpe_ratio(y, y_pred, w, periods_per_year =times_per_year ):
    
    rets = _cal_rets(y,y_pred,w)
    sharp_ratio = np.nanmean(rets)/ np.nanstd(rets) * np.sqrt(periods_per_year)
    
    return sharp_ratio





def _calculate_rolling_sharp(y, y_pred,w,t=rolling_w, period_per_year=times_per_year):
    '''计算滚动的夏普比率'''

    rets = _cal_rets(y,y_pred,w)
    rets = pd.Series(rets)
    rolling_returns = rets.rolling(window=t).mean()
    annual_std = rets.rolling(window=t).std()
    sharpe_ratio = rolling_returns / annual_std * np.sqrt(period_per_year)

    return np.nan_to_num(sharpe_ratio)






# 分位数分箱
def _rolling_percentile_position(y_pred, up_q, dn_q, window_size =rolling_w):

    y_pred = np.nan_to_num(y_pred).flatten()
    y_pred = pd.Series(y_pred)

    # 计算每个滚动窗口的 75% 和 25% 分位数
    percentile_75 = y_pred.rolling(window=window_size).quantile(up_q)
    percentile_25 = y_pred.rolling(window=window_size).quantile(dn_q) 

    pos = np.select(
        [y_pred > percentile_75, y_pred < percentile_25],  # 条件
        [1, -1],  # 对应的替换值
        default=0  # 默认值
    )                          

    return pos

def _sharpe_fixed_threshold(y, y_pred, w, up_q = 0.9, dn_q =0.1, periods_per_year=times_per_year,window_size =rolling_w):
    #固定常数阈值的仓位转化回测

    y_pred = np.nan_to_num(y_pred).flatten()
    position = _rolling_percentile_position(y_pred, up_q, dn_q, window_size)
    sharpe = _calculate_sharpe_ratio(y, position, w, periods_per_year)

    return sharpe

# std分箱
def _rolling_std_position(y_pred, n_std = 2, window_size =rolling_w):

    y_pred = np.nan_to_num(y_pred).flatten()
    y_pred = pd.Series(y_pred)

    y_pred_mean = y_pred.rolling(window=window_size).mean()
    y_pred_std = y_pred.rolling(window=window_size).std()
    up_r = y_pred_mean + n_std*y_pred_std
    dn_r = y_pred_mean - n_std*y_pred_std

    pos = np.select(
        [y_pred > up_r, y_pred < dn_r],  # 条件
        [1, -1],  # 对应的替换值
        default=0  # 默认值
    )                          

    return pos

def _sharpe_std_threshold(y, y_pred,w ,n_std = 2, window_size =rolling_w, periods_per_year=times_per_year):
    #固定n倍std的仓位转化回测

    y_pred = np.nan_to_num(y_pred).flatten()
    position = _rolling_std_position(y_pred, n_std, window_size)
    sharpe = _calculate_sharpe_ratio(y, position, w, periods_per_year)

    return sharpe









weighted_pearson = _Fitness(function=_weighted_pearson,
                            greater_is_better=True)
weighted_spearman = _Fitness(function=_weighted_spearman,
                             greater_is_better=True)
mean_absolute_error = _Fitness(function=_mean_absolute_error,
                               greater_is_better=False)
mean_square_error = _Fitness(function=_mean_square_error,
                             greater_is_better=False)
root_mean_square_error = _Fitness(function=_root_mean_square_error,
                                  greater_is_better=False)
log_loss = _Fitness(function=_log_loss,
                    greater_is_better=False)


avg_pic = _Fitness(function=_calculate_average_pic,
                            greater_is_better=True)
avg_sic = _Fitness(function=_calculate_average_sic,
                            greater_is_better=True)
max_ic = _Fitness(function=_calculate_max_ic_chunk,
                            greater_is_better=True)
max_ic_train = _Fitness(function= _calculate_max_ic_chunk_train,
                            greater_is_better=True)
given_ic_test = _Fitness(function= _calculate_given_range_ic,
                            greater_is_better=True)

risk_constrained_chunk = _Fitness(function=_risk_constrained_chunk,
                            greater_is_better=True)


#辅助计算指标
pnl = _Fitness(function=_cal_pnl,
                            greater_is_better=True)
rts = _Fitness(function=_cal_rets,
                            greater_is_better=True)


max_dd = _Fitness(function=_calculate_max_drawdown,
                            greater_is_better=False)
avg_mdd = _Fitness(function=_calculate_average_max_drawdown,
                            greater_is_better=False)
calmar = _Fitness(function=_calculate_calmar_ratio,
                            greater_is_better=True)
sharp = _Fitness(function=_calculate_sharpe_ratio,
                            greater_is_better=True)
rolling_sharp = _Fitness(function=_calculate_rolling_sharp,
                            greater_is_better=True)
sharpe_fixed_threshold = _Fitness(function=_sharpe_fixed_threshold,
                            greater_is_better=True)
sharpe_std_threshold = _Fitness(function=_sharpe_std_threshold,
                            greater_is_better=True)
extreme_wrong_sign_penalty = _Fitness(function=_extreme_wrong_sign_penalty,
                            greater_is_better=True)
weighted_margin_score = _Fitness(function=_weighted_margin_score,
                            greater_is_better=True)
asym_extreme_direction_score = _Fitness(function=_asym_extreme_direction_score,
                            greater_is_better=True)
mi_score = _Fitness(function=_calculate_mi_score,
                            greater_is_better=True)
entropy_gain_score = _Fitness(function=_calculate_entropy_gain_score,
                            greater_is_better=True)



# #全集
# _fitness_map =     {'pearson': weighted_pearson,
#                     'spearman': weighted_spearman,
#                     'mae': mean_absolute_error,
#                     'mse': mean_square_error,
#                     'rmse': root_mean_square_error,
#                     'logloss': log_loss,
#                     'avg_pic':avg_pic,
#                     'avg_sic':avg_sic,
#                     'max_ic':max_ic,
#                     'max_ic_train':max_ic_train,
#                     'given_ic_test':given_ic_test,
#                     'pnl':pnl,
#                     'rts':rts,
#                     'max_dd':max_dd,
#                     'avg_mdd':avg_mdd,
#                     'calmar':calmar,
#                     'sharp':sharp,
#                     'rolling_sharp':rolling_sharp,
#                     'sharpe_fixed_threshold':sharpe_fixed_threshold,
#                     'sharpe_std_threshold':sharpe_std_threshold
#                     }

#实际调用
_fitness_map =     {'avg_pic':avg_pic,
                    'avg_sic':avg_sic,
                    'calmar':calmar,
                    'sharp':sharp,
                    'sharpe_fixed_threshold':sharpe_fixed_threshold,
                    'sharpe_std_threshold':sharpe_std_threshold,
                    'extreme_wrong_sign_penalty':extreme_wrong_sign_penalty,
                    'weighted_margin_score':weighted_margin_score,
                    'asym_extreme_direction_score':asym_extreme_direction_score,
                    'mi_score':mi_score,
                    'entropy_gain_score':entropy_gain_score,
                    'max_dd':max_dd,
                    'avg_mdd':avg_mdd,
                    'max_ic':max_ic,
                    'max_ic_train':max_ic_train,
                    'given_ic_test':given_ic_test
                    }

# 只在启用地option3时调用，因为fitness会输出字典或者序列
_backtest_map = {'pnl':pnl,
                 'rts':rts,
                 'rolling_sharp':rolling_sharp
                }


def _plot_margin_metric_demo(y, y_pred_good, y_pred_bad, save_path="fitness_margin_demo.png"):
    """Plot synthetic fat-tail returns and save figure to file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skip plotting.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(y, color="tab:blue", alpha=0.75, linewidth=1.0)
    axes[0, 0].set_title("Synthetic Fat-Tail Return (t-dist)")
    axes[0, 0].set_xlabel("Index")
    axes[0, 0].set_ylabel("Return")

    axes[0, 1].scatter(y, y_pred_good, s=10, alpha=0.45, label="good_pred")
    axes[0, 1].scatter(y, y_pred_bad, s=10, alpha=0.35, label="bad_pred")
    axes[0, 1].set_title("y vs prediction")
    axes[0, 1].set_xlabel("y")
    axes[0, 1].set_ylabel("y_pred")
    axes[0, 1].legend(loc="best")

    axes[1, 0].hist(y, bins=60, alpha=0.8, color="tab:purple")
    axes[1, 0].set_title("Return Distribution (fat-tail)")
    axes[1, 0].set_xlabel("y")
    axes[1, 0].set_ylabel("Count")

    axes[1, 1].hist(y_pred_good, bins=60, alpha=0.55, label="good_pred")
    axes[1, 1].hist(y_pred_bad, bins=60, alpha=0.55, label="bad_pred")
    axes[1, 1].set_title("Prediction Distribution")
    axes[1, 1].set_xlabel("y_pred")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend(loc="best")

    plt.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"plot saved to: {save_path}")


def main():
    """Quick sanity check for new margin-based fitness metrics."""
    np.random.seed(42)
    n = 800
    y = np.random.standard_t(df=3, size=n) * 0.02
    w = np.ones(n)

    y_pred_good = y + np.random.normal(0.0, 0.01, size=n)
    y_pred_bad = -y + np.random.normal(0.0, 0.01, size=n)

    metrics = [
        ("extreme_wrong_sign_penalty", _extreme_wrong_sign_penalty),
        ("weighted_margin_score", _weighted_margin_score),
        ("asym_extreme_direction_score", _asym_extreme_direction_score),
    ]

    print("=== Margin-based fitness demo (fat-tail t-distribution) ===")
    for metric_name, metric_func in metrics:
        good_score = metric_func(y, y_pred_good, w)
        bad_score = metric_func(y, y_pred_bad, w)
        print(
            f"{metric_name}: "
            f"good={good_score:.6f}, bad={bad_score:.6f}, good>bad={good_score > bad_score}"
        )

    _plot_margin_metric_demo(y, y_pred_good, y_pred_bad, save_path="fitness_margin_demo.png")


if __name__ == "__main__":
    main()