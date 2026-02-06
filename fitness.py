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
from scipy.stats import rankdata,pearsonr,spearmanr
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