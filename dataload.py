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
