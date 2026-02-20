"""
加工原始特征，衍生新的特征
"""
import datetime
from dataclasses import dataclass, field
from typing import List
import pandas as pd
import numpy as np
import talib
from scipy.stats import rankdata
from functools import singledispatch
import warnings
import dataload
import yaml
from tqdm import tqdm

warnings.filterwarnings('ignore')


# def define_base_fields():
def define_base_fields(rolling_zscore_window: int = 2000, include_categories: List[str] = None, init_ohlcva_df: pd.DataFrame = None):
    """
    本函数定义了基础的特征计算公式. 将来会持续维护这个函数，增加更多的特征计算公式
    这是唯一的定义特征的地方，其他地方不应该再定义特征
    """
    def fm20(c: pd.Series, o: pd.Series) -> pd.Series:
        '''输入：close，open
        输出：窗口为20的波动的周期特征fm20,from John Elther
        '''
        period = 20
        Deriv = c - o
        # 滤除AM噪音
        HL = np.clip(10 * Deriv, -1, 1)
        a1 = np.exp(-1.414 * np.pi / period)
        b1 = 2 * a1 * np.cos(1.414 * 180 / period)
        # c2 = b1
        c3 = -a1 * a1
        c1 = 1 - b1 - c3
        # 初始化输出序列
        ss = np.zeros_like(HL)
        ss[:2] = HL[:2]  # 使用前两个值作为初始化值
        for i in range(2, len(HL)):
            ss[i] = c1 * (HL[i] + HL[i-1]) / 2 + b1 * ss[i-1] + c3 * ss[i-2]
        return ss

    def fm30(c: pd.Series, o: pd.Series) -> pd.Series:
        '''输入：close，open
        输出：窗口为30的波动的周期特征fm30,from John Elther
        '''
        period = 30
        Deriv = c - o
        # 滤除AM噪音
        HL = np.clip(10 * Deriv, -1, 1)
        a1 = np.exp(-1.414 * np.pi / period)
        b1 = 2 * a1 * np.cos(1.414 * 180 / period)
        # c2 = b1
        c3 = -a1 * a1
        c1 = 1 - b1 - c3
        # 初始化输出序列
        ss = np.zeros_like(HL)
        ss[:2] = HL[:2]  # 使用前两个值作为初始化值
        for i in range(2, len(HL)):
            ss[i] = c1 * (HL[i] + HL[i-1]) / 2 + b1 * ss[i-1] + c3 * ss[i-2]
        return ss

    def fm40(c: pd.Series, o: pd.Series) -> pd.Series:
        '''输入：close，open
        输出：窗口为40的波动的周期特征fm40,from John Elther
        '''
        period = 40
        Deriv = c - o
        # 滤除AM噪音
        HL = np.clip(10 * Deriv, -1, 1)
        a1 = np.exp(-1.414 * np.pi / period)
        b1 = 2 * a1 * np.cos(1.414 * 180 / period)
        # c2 = b1
        c3 = -a1 * a1
        c1 = 1 - b1 - c3
        # 初始化输出序列
        ss = np.zeros_like(HL)
        ss[:2] = HL[:2]  # 使用前两个值作为初始化值
        for i in range(2, len(HL)):
            ss[i] = c1 * (HL[i] + HL[i-1]) / 2 + b1 * ss[i-1] + c3 * ss[i-2]
        return ss

    def fm60(c: pd.Series, o: pd.Series) -> pd.Series:
        '''输入：close，open
        输出：窗口为60的波动的周期特征fm60,from John Elther
        '''
        period = 60
        Deriv = c - o
        # 滤除AM噪音
        HL = np.clip(10 * Deriv, -1, 1)
        a1 = np.exp(-1.414 * np.pi / period)
        b1 = 2 * a1 * np.cos(1.414 * 180 / period)
        # c2 = b1
        c3 = -a1 * a1
        c1 = 1 - b1 - c3
        # 初始化输出序列
        ss = np.zeros_like(HL)
        ss[:2] = HL[:2]  # 使用前两个值作为初始化值
        for i in range(2, len(HL)):
            ss[i] = c1 * (HL[i] + HL[i-1]) / 2 + b1 * ss[i-1] + c3 * ss[i-2]
        return ss

    # def norm(x: np.ndarray) -> np.ndarray:
    #     x = np.asarray(x, dtype=np.float64)
    #     # mean = pd.Series(x).rolling(2000, min_periods=1).mean().values
    #     std = pd.Series(x).rolling(2000, min_periods=1).std().values
    #     # x_value = (x - mean) / np.clip(np.nan_to_num(std),
    #     #                                a_min=1e-6, a_max=None)
    #     x_value = (x ) / np.clip(np.nan_to_num(std),
    #                                    a_min=1e-6, a_max=None)
    #     # x_value = np.clip(x_value, -6, 6)
    #     x_value = np.nan_to_num(x_value, nan=0.0, posinf=0.0, neginf=0.0)
    #     return x_value

    def norm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        # mean = pd.Series(x).rolling(2000, min_periods=1).mean().values
        std = pd.Series(x).rolling(rolling_zscore_window, min_periods=1).std().values
        # x_value = (x - mean) / np.clip(np.nan_to_num(std),
        #                                a_min=1e-6, a_max=None)
        x_value = (x ) / np.clip(np.nan_to_num(std),
                                       a_min=1e-6, a_max=None)
        # x_value = np.clip(x_value, -6, 6)
        x_value = np.nan_to_num(x_value, nan=0.0, posinf=0.0, neginf=0.0)
        return x_value
    

    return {
        'lgp_shortcut_illiq_6': lambda data: norm(np.nan_to_num(pd.Series(2*(data['h'] - data['l']) - np.abs(data['c'] - data['o'])).rolling(6, min_periods=1).apply(lambda x: x.mean()))),
        # 'h_ts_std_10': lambda data: norm(np.nan_to_num(pd.Series(data['h']).rolling(window=10, min_periods=5).std())),
        # 'v_ta_cmo_25': lambda data: norm(talib.CMO(data['vol'], 25)),
        # 'v_ts_argrange_10': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=10, min_periods=5).apply(np.argmax) + 1) - np.nan_to_num(pd.Series(data['vol']).rolling(window=10, min_periods=5).apply(np.argmin) + 1)),
        # 'c_ts_argrange_10': lambda data: norm(np.nan_to_num(pd.Series(data['c']).rolling(window=10, min_periods=5).apply(np.argmax) + 1) - np.nan_to_num(pd.Series(data['c']).rolling(window=10, min_periods=5).apply(np.argmin) + 1)),
        # 'c_ts_day_min_40': lambda data: norm(np.nan_to_num(pd.Series(data['c']).rolling(window=41).apply(lambda x: 40 - x.values.tolist()[:-1].index(np.min(x.values.tolist()[:-1]))))),
        # 'l_ts_prod_8': lambda data: norm(np.nan_to_num(pd.Series(data['l']).rolling(window=8, min_periods=4).apply(np.prod))),
        # 'v_ts_std_20': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=20, min_periods=10).std())),
        # 'h_ta_lr_angle_10': lambda data: norm(np.nan_to_num(talib.LINEARREG_ANGLE(data['h'], timeperiod=10))),
        # 'v_ts_sma_21': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=21, min_periods=11).mean())),
        # 'h_ts_kurt_20': lambda data: norm(np.nan_to_num(pd.Series(data['h']).rolling(window=20, min_periods=10).kurt())),
        # 'v_ts_range_5': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=5, min_periods=3).max()) - np.nan_to_num(pd.Series(data['vol']).rolling(window=5, min_periods=3).min())),
        # 'c_ts_sma_21': lambda data: norm(np.nan_to_num(pd.Series(data['c']).rolling(window=21, min_periods=11).mean())),
        # 'o_ts_argmin_10': lambda data: norm(np.nan_to_num(pd.Series(data['o']).rolling(window=10, min_periods=5).apply(np.argmin) + 1)),
        # 'o_ta_lr_slope_10': lambda data: norm(np.nan_to_num(talib.LINEARREG_SLOPE(data['o'], timeperiod=10))),
        # 'l_ts_argmax_20': lambda data: norm(np.nan_to_num(pd.Series(data['l']).rolling(window=20, min_periods=10).apply(np.argmax) + 1)),
        # 'o_ts_range_10': lambda data: norm(np.nan_to_num(pd.Series(data['o']).rolling(window=10, min_periods=5).max()) - np.nan_to_num(pd.Series(data['o']).rolling(window=10, min_periods=5).min())),
        # 'h_ts_argmax_5': lambda data: norm(np.nan_to_num(pd.Series(data['h']).rolling(window=5, min_periods=3).apply(np.argmax) + 1)),
        # 'l_ts_day_min_40': lambda data: norm(np.nan_to_num(pd.Series(data['l']).rolling(window=41).apply(lambda x: 40 - x.values.tolist()[:-1].index(np.min(x.values.tolist()[:-1]))))),
        # 'v_ts_day_max_20': lambda data: norm(np.nan_to_num(pd.Series(data['vol']).rolling(window=21).apply(lambda x: 20 - x.values.tolist()[:-1].index(np.max(x.values.tolist()[:-1]))))),
        # 'c_ta_tsf_5': lambda data: norm(np.nan_to_num(talib.TSF(data['c'], timeperiod=5))),
        # 'c_power_c': lambda data: norm(np.nan_to_num(np.power(data['c'], 3))),
        # 'l_ts_argrange_5': lambda data: norm(np.nan_to_num(pd.Series(data['l']).rolling(window=5, min_periods=3).apply(np.argmax) + 1) - np.nan_to_num(pd.Series(data['l']).rolling(window=5, min_periods=3).apply(np.argmin) + 1)),
        # 'v_power_a': lambda data: norm(np.nan_to_num(np.power(data['vol'], 3))),
        # 'v_cci_25_obv': lambda data: norm(talib.OBV(norm(np.nan_to_num(talib.CCI(data['h'], data['l'], data['c'], timeperiod=25))), data['vol'])),
        # 'v_cci_25_sum': lambda data: norm(np.cumsum(norm(np.nan_to_num(talib.CCI(data['h'], data['l'], data['c'], timeperiod=25))) * data['vol'])),
        # 'ori_trix_8': lambda data: norm(np.nan_to_num(talib.TRIX(data['c'], timeperiod=8))),
        # 'ori_trix_21': lambda data: norm(np.nan_to_num(talib.TRIX(data['c'], timeperiod=21))),
        # 'ori_trix_55': lambda data: norm(np.nan_to_num(talib.TRIX(data['c'], timeperiod=55))),
        # 'v_trix_8_obv': lambda data: norm(talib.OBV(norm(np.nan_to_num(talib.TRIX(data['c'], timeperiod=8))), data['vol'])),
        # 'v_sar_obv': lambda data: norm(talib.OBV(norm(np.nan_to_num(talib.SAR(data['h'], data['l']))), data['vol'])),
        # 'ori_ta_bop': lambda data: norm(talib.BOP(data['o'], data['h'], data['l'], data['c'])),
        # 'v_bop_sum': lambda data: norm(np.cumsum(norm(talib.BOP(data['o'], data['h'], data['l'], data['c'])) * data['vol'])),
        # 'ori_rsi_6': lambda data: norm(talib.RSI(data['c'], timeperiod=6)),
        # 'ori_rsi_12': lambda data: norm(talib.RSI(data['c'], timeperiod=12)),
        # 'ori_rsi_24': lambda data: norm(talib.RSI(data['c'], timeperiod=24)),
        # 'v_rsi_6_obv': lambda data: norm(talib.OBV(norm(talib.RSI(data['c'], timeperiod=6)), data['vol'])),
        # 'v_rsi_6_sum': lambda data: norm(np.cumsum(norm(talib.RSI(data['c'], timeperiod=6)) * data['vol'])),
        # 'v_rsi_12_obv': lambda data: norm(talib.OBV(norm(talib.RSI(data['c'], timeperiod=12)), data['vol'])),
        # 'v_rsi_12_sum': lambda data: norm(np.cumsum(norm(talib.RSI(data['c'], timeperiod=12)) * data['vol'])),
        # 'v_rsi_24_obv': lambda data: norm(talib.OBV(norm(talib.RSI(data['c'], timeperiod=24)), data['vol'])),
        # 'v_rsi_24_sum': lambda data: norm(np.cumsum(norm(talib.RSI(data['c'], timeperiod=24)) * data['vol'])),
        # 'ori_cmo_14': lambda data: norm(np.nan_to_num(talib.CMO(data['c'], timeperiod=14))),
        # 'ori_cmo_25': lambda data: norm(np.nan_to_num(talib.CMO(data['c'], timeperiod=25))),
        # 'v_cmo_14_obv': lambda data: norm(talib.OBV(norm(np.nan_to_num(talib.CMO(data['c'], timeperiod=14))), data['vol'])),
        # 'fm20': lambda data: norm(np.nan_to_num(fm20(data['c'], data['o']))),
        # 'fm30': lambda data: norm(np.nan_to_num(fm30(data['c'], data['o']))),
        # 'fm40': lambda data: norm(np.nan_to_num(fm40(data['c'], data['o']))),
        # 'fm60': lambda data: norm(np.nan_to_num(fm60(data['c'], data['o']))),
        # 'ori_ta_macd': lambda data: norm(norm(np.nan_to_num(talib.MACD(data['c'], fastperiod=12, slowperiod=26, signalperiod=9)[2]))),
        # 'ori_ta_obv': lambda data: norm(np.nan_to_num(talib.OBV(data['c'], data['vol']))),
        # 'ori_ta_ad': lambda data: norm(np.nan_to_num(talib.AD(data['h'], data['l'], data['c'], data['vol']))),
        # 'ori_ta_bop': lambda data: norm(talib.BOP(data['o'], data['h'], data['l'], data['c'])),
        # 'ma8_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=8, matype=0) / data['c'])),
        # 'ma15_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=15, matype=0) / data['c'])),
        # 'ma25_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=25, matype=0) / data['c'])),
        # 'ma35_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=35, matype=0) / data['c'])),
        # 'ma55_c': lambda data: norm(np.nan_to_num(talib.MA(data['c'], timeperiod=55, matype=0) / data['c'])),
        # 'h_line': lambda data: norm(np.nan_to_num(talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0])),
        # 'm_line': lambda data: norm(np.nan_to_num(talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1])),
        # 'l_line': lambda data: norm(np.nan_to_num(talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2])),
        # 'stdevrate': lambda data: norm((talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0] -
        #                                 talib.BBANDS(data['c'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2]) /
        #                                (data['c'] * 4)),
        # 'sar_index': lambda data: norm(np.nan_to_num(talib.SAR(data['h'], data['l']))),
        # 'sar_close': lambda data: norm((np.nan_to_num(talib.SAR(data['h'], data['l'])) - data['c']) / data['c']),
        # 'mfi_index': lambda data: norm(np.nan_to_num(talib.MFI(data['h'], data['l'], data['c'], data['vol']))),
        # 'mfi_30': lambda data: norm(np.nan_to_num(talib.MFI(data['h'], data['l'], data['c'], data['vol'], timeperiod=30))),
        # 'ppo': lambda data: norm(np.nan_to_num(talib.PPO(data['c'], fastperiod=12, slowperiod=26, matype=0))),
        # 'ad_index': lambda data: norm(np.nan_to_num(talib.AD(data['h'], data['l'], data['c'], data['vol']))),
        # 'ad_real': lambda data: norm(np.nan_to_num(talib.ADOSC(data['h'], data['l'], data['c'], data['vol'], fastperiod=3, slowperiod=10))),
        # 'tr_index': lambda data: norm(np.nan_to_num(talib.TRANGE(data['h'], data['l'], data['c']))),
        # 'sarext': lambda data: norm(np.nan_to_num(talib.SAREXT(data['h'], data['l'], startvalue=0, offsetonreverse=0,
        #                                                        accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0,
        #                                                        accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0))),
        # 'kdj_d': lambda data: talib.STOCH(data['h'], data['l'], data['c'], 9, 3, 3)[1],
        # 'kdj_k': lambda data: talib.STOCH(data['h'], data['l'], data['c'], 9, 3, 3)[0],
        # 'obv_v': lambda data: norm(np.nan_to_num(talib.OBV(data['c'], data['vol']))),
        # 'volume_macd': lambda data: norm(np.nan_to_num(talib.MACD(data['vol'], fastperiod=12, slowperiod=26, signalperiod=9)[2])),
        # 'close_macd': lambda data: norm(np.nan_to_num(talib.MACD(data['c'], fastperiod=12, slowperiod=26, signalperiod=9)[2])),
        # 'cci_55': lambda data: norm(talib.CCI(data['c'], data['l'], data['h'], 55)),
        # 'cci_25': lambda data: norm(talib.CCI(data['c'], data['l'], data['h'], 25)),
        # 'cci_14': lambda data: norm(talib.CCI(data['c'], data['l'], data['h'], 14)),
    }


def calculate_base_fields(data, base_fields, apply_norm=True, rolling_zscore_window=2000):
    for field, formula in tqdm(base_fields.items(), desc="Processing"):
        if apply_norm:
            data[field] = norm1(formula(data), rolling_zscore_window)
        else:
            data[field] = formula(data)
    return data


def _expanding_zscore(x, ddof=1):
    # 这是相当于expanding z-score标准化，但是这里的标准差是用的无偏估计
    x = np.array(x)
    x = np.nan_to_num(x)
    x_cumsum = np.cumsum(x)
    x_squared_cumsum = np.cumsum(x ** 2)
    count = np.arange(1, len(x) + 1)
    x_mean = x_cumsum / count
    x_std = np.sqrt(((x_squared_cumsum - 2 * x_cumsum * x_mean) / count) + x_mean ** 2) * np.sqrt(
        count / (count - ddof))
    x_value = (x - x_mean) / x_std
    # clip的值，需要测算，没有在log_return 基础上乘1000
    # x_value = np.clip(x_value, -6, 6)
    x_value = np.nan_to_num(x_value)

    return x_value


# ----------k_v1 version-----------------
def norm1(x, rolling_zscore_window):
    window = rolling_zscore_window

    arr = np.asarray(x)
    x = np.sign(arr) * np.log1p(np.abs(arr)) / np.log1p(np.abs(np.mean(arr)))

    factors_data = pd.DataFrame(x, columns=['factor'])
    factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
    # factors_mean = factors_data.rolling(window=window, min_periods=1).mean()
    factors_std = factors_data.rolling(window=window, min_periods=1).std()
    # factor_value = (factors_data - factors_mean) / factors_std
    factor_value = (factors_data ) / factors_std
    factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
    # factor_value = factor_value.clip(-6, 6)
    return np.nan_to_num(factor_value).flatten()

def calculate_features_df(input_df, rolling_zscore_window):
    base_fields = define_base_fields()
    data = calculate_base_fields(input_df.copy(
    ), base_fields, apply_norm=True, rolling_zscore_window=rolling_zscore_window)
    data = data.replace([np.nan, -np.inf, np.inf], 0.0)
    return data


def calculate_features_df_tail(input_df, rolling_zscore_window):
    base_fields = define_base_fields()
    data = calculate_base_fields(
        input_df.copy(), base_fields, apply_norm=False)

    last_row = {}
    # 如下这些是最基础的行情数据，不需要进行norm处理
    base_columns = ['o', 'h', 'l', 'c', 'vol_ccy', 'vol','trades',
           'oi', 'oi_ccy', 'toptrader_count_lsr', 'toptrader_oi_lsr', 'count_lsr',
           'taker_vol_lsr']

    for column in data.columns:
        if column in base_columns:
            last_row[column] = data[column].values[-1]
        else:
            column_data = data[column].values
            if len(column_data) >= rolling_zscore_window:
                column_data = column_data[-rolling_zscore_window:]
            else:
                print(
                    f"Warning: {column} has less than {rolling_zscore_window} values.")
            normalized_data = norm1(column_data, rolling_zscore_window)
            last_row[column] = normalized_data[-1]

    last_row_result = pd.Series(last_row, name=input_df.index[-1])
    last_row_result = last_row_result.replace([np.nan, -np.inf, np.inf], 0.0)
    return last_row_result




class BaseFeature:
    
    def __init__(self, init_ohlcva_df, include_categories: List[str] = None, rolling_zscore_window: int = 2000):
        # 将所有列转换为 double 类型
        self.init_ohlcva_df = init_ohlcva_df.astype(np.float64)

        self.rolling_zscore_window = rolling_zscore_window
        print('feature 定义')
        # self.base_fields = define_base_fields()
        self.base_fields = define_base_fields(rolling_zscore_window = rolling_zscore_window, include_categories=include_categories)
        print('init_feature 计算')
        self.init_feature_df = self._call(init_ohlcva_df)
        print('init_feature 完成')

    # def __init__(self, init_ohlcva_df):
    #     # 将所有列转换为 double 类型
    #     init_ohlcva_df = init_ohlcva_df.astype(np.float64)

    #     self.rolling_zscore_window: int = 2000
    #     print('feature 定义')
    #     self.base_fields = define_base_fields()
    #     print('init_feature 计算')
    #     self.init_feature_df = self._call(init_ohlcva_df)
    #     print('init_feature 完成')



    def _call(self, data):
        data = calculate_base_fields(
            data, self.base_fields, apply_norm=False, rolling_zscore_window=self.rolling_zscore_window)
        data = data.replace([np.nan, -np.inf, np.inf], 0.0)
        return data






