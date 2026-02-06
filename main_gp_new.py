import time
import numpy as np
import pandas as pd
from genetic import SymbolicTransformer
from functions import _function_map
import dataload
from datetime import datetime
from pathlib import Path
import warnings
from loguru import logger
import gzip
import yaml
import shutil
import pickle
import joblib
import schedule
import os
import talib as ta
import fitness
from datetime import datetime
from sklearn.linear_model import LinearRegression
import cloudpickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import zscore, kurtosis, skew, yeojohnson, boxcox
from scipy.stats import tukeylambda, mstats
from expressionProgram import FeatureEvaluator
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

norm_y_list = ['avg_pic','avg_sic','max_ic','max_ic_train','given_ic_test']
raw_y_list = ['calmar','sharp','sharpe_fixed_threshold','sharpe_std_threshold','max_dd','avg_mdd']

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



class GPAnalyzer:
    """
    GPAnalyzer 类有以下主要内容：

    initialize_his_data 方法，用于初始化历史数据。这个方法只在第一次调用时加载数据，避免重复加载。
    run 方法，替代之前的 main 函数。这个方法根据配置决定是单次执行还是循环执行任务。

    """
    def __init__(self, yaml_file_path):
        self.yaml_file_path = yaml_file_path
        self.config = self.load_yaml_config(yaml_file_path)
        self.load_config_attributes()
        self.data_initialized = False
        self.base_model_directory = Path.cwd() / 'gp_models'
        self.initialize_his_data()
        self.total_factor_file_name = f"{self.sym}_{self.freq}_{self.y_train_ret_period}_{self.start_date_train}_{self.end_date_train}_{self.start_date_test}_{self.end_date_test}.csv.gz"
        self.total_factor_file_path = self.base_model_directory / self.total_factor_file_name

    def load_yaml_config(self, file_path):
        """
        从YAML文件加载配置。
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def load_config_attributes(self):
        """
        从配置字典中加载属性到类实例。
        """
        self.freq = self.config.get('freq', '')
        self.y_train_ret_period = self.config.get('y_train_ret_period', 1)
        self.sym = self.config.get('sym', '')
        self.start_date_train = self.config.get('start_date_train', '')
        self.end_date_train = self.config.get('end_date_train', '')
        self.start_date_test = self.config.get('start_date_test', '')
        self.end_date_test = self.config.get('end_date_test', '')
        self.gp_settings = self.config.get('gp_settings', {})
        self.metric = self.gp_settings.get('metric', 'pearson')
        self.verbose_logging = self.config.get('verbose_logging', False)
        self.rolling_window = self.config.get('rolling_window', 2000)
        self.annual_bars = calculate_annual_bars(self.freq)
        

        # 自动加载其他配置项
        for key, value in self.config.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def initialize_his_data(self):
        """
        初始化共享数据，包括训练和测试数据集。
        只在第一次调用时执行数据加载。
        """
        if not self.data_initialized:
            self.X_all, self.X_train, self.y_train, self.ret_train, self.X_test, self.y_test, self.ret_test, self.feature_names,self.open_train,self.open_test,self.close_train,self.close_test, self.z_index ,self.ohlc= dataload.data_prepare(
                self.sym, self.freq, self.start_date_train, self.end_date_train,
                self.start_date_test, self.end_date_test, rolling_w=self.rolling_window)
            self.data_initialized = True
            self.test_index = self.z_index[(self.z_index >= pd.to_datetime(self.start_date_test)) & (
                        self.z_index <= pd.to_datetime(self.end_date_test))]
            self.train_index = self.z_index[(self.z_index >= pd.to_datetime(self.start_date_train)) & (
                        self.z_index < pd.to_datetime(self.end_date_train))]           
        else:
            print("Shared data already initialized. Skipping data loading.")



    def gp(self, X, y, feature_names, random_state):
        """
        执行遗传编程过程。

        Args:
            X: 训练数据特征。
            y: 训练数据标签。
            feature_names: 特征名称列表。
            random_state: 随机状态。

        Returns:
            SymbolicTransformer: 训练好的遗传编程模型。
        """
        func = list(_function_map.keys())

        # 转换列表为元组
        self.gp_settings['init_depth'] = tuple(self.gp_settings['init_depth'])
        

        ST_gplearn = SymbolicTransformer(
            population_size=self.gp_settings.get('population_size', 5),
            hall_of_fame=self.gp_settings.get('hall_of_fame', 2),
            n_components=self.gp_settings.get('n_components', 1),
            generations=self.gp_settings.get('generations', 2),
            tournament_size=self.gp_settings.get('tournament_size', 2),
            const_range=self.gp_settings.get('const_range', None),
            init_depth=self.gp_settings.get('init_depth', (2, 5)),
            function_set=func,
            metric=self.metric,
            parsimony_coefficient=self.gp_settings.get('parsimony_coefficient', 0),
            p_crossover=self.gp_settings.get('p_crossover', 0.9),
            p_subtree_mutation=self.gp_settings.get('p_subtree_mutation', 0.01),
            p_hoist_mutation=self.gp_settings.get('p_hoist_mutation', 0.01),
            p_point_mutation=self.gp_settings.get('p_point_mutation', 0.01),
            p_point_replace=self.gp_settings.get('p_point_replace', 0.4),
            feature_names=feature_names,
            n_jobs=self.gp_settings.get('n_jobs', -1),
            corrcoef_threshold=self.gp_settings.get('corrcoef_threshold', 0.9),
            random_state=random_state
        )

        ST_gplearn.fit(X, y)
        return ST_gplearn




    def run_genetic_programming(self, random_state=None):
        """
        fct_generate的一部分
        执行遗传编程过程。
        """
        if self.metric in norm_y_list :
            self.est_gp = self.gp(self.X_train, self.y_train, feature_names=self.feature_names, random_state=random_state)
        else:
            self.est_gp = self.gp(self.X_train, self.ret_train, feature_names=self.feature_names, random_state=random_state)
           
        
        

    def process_best_programs(self):
        """
        fct_generate的一部分
        处理最佳程序并创建数据框。
        """
        best_programs = self.est_gp._best_programs
        best_programs_dict = {}
        fitness_key = f"fitness_{self.metric}"

        for p in best_programs:
            factor_expression = 'alpha_' + str(best_programs.index(p) + 1)
            best_programs_dict[factor_expression] = {fitness_key: p.fitness_, 'expression': str(p), 'depth': p.depth_,
                                               'length': p.length_}

        self.best_programs_df = pd.DataFrame(best_programs_dict).T
        self.best_programs_df['factor_order_in_model'] = self.best_programs_df.index
        self.best_programs_df = self.best_programs_df.sort_values(by=fitness_key, ascending=False)

    def save_initial_results(self):
        """
        fct_generate的一部分
        保存初步结果到文件。
        """
        current_time = datetime.now().strftime('%Y%m%d%H%M%S')
        model_folder_name = f"{self.sym}_{self.freq}_{self.y_train_ret_period}_{self.start_date_train}_{self.end_date_train}_{self.start_date_test}_{self.end_date_test}_{self.metric}_{current_time}"
        base_model_directory = Path.cwd() / 'gp_models'
        self.model_folder = base_model_directory / model_folder_name
        self.model_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f'保存本轮信息和结果到对应文件夹 {self.model_folder}')

        yaml_file = Path(self.yaml_file_path)
        shutil.copy(yaml_file, self.model_folder / yaml_file.name)

        self.best_programs_df['sym'] = self.sym
        self.best_programs_df['freq'] = self.freq
        self.best_programs_df['y_train_ret_period'] = self.y_train_ret_period
        self.best_programs_df['start_date_train'] = self.start_date_train
        self.best_programs_df['end_date_train'] = self.end_date_train
        self.best_programs_df['start_date_test'] = self.start_date_test
        self.best_programs_df['end_date_test'] = self.end_date_test
        self.best_programs_df['metric'] = self.metric
        self.best_programs_df['current_time'] = current_time

        self.best_programs_df.to_csv(self.model_folder / 'best_programs_df.csv.gz', index=False, compression='gzip')



    def load_total_factor_df(self):
        """
        加载总因子数据框。
        """
        if not self.total_factor_file_path.exists():
            print(f"Factor file {self.total_factor_file_path} does not exist.")
            return None

        return pd.read_csv(self.total_factor_file_path, compression='gzip')
 
    def evaluate_single_factor(self, factor_expression, metric):
        """
        评估单个因子的表现

        只生成一个evaluator对象，然后用这个evaluator来评估train和test的表现(截断) -  因为要兼容那些对于start_date敏感的因子
        """
        # self.X_all已经是根据train_start_date和test_end_date截断过的数据
        evaluator = FeatureEvaluator(_function_map, self.feature_names, self.X_all)

        result = evaluator.evaluate(factor_expression)  # 解析因子
        result = np.nan_to_num(result)
        # 训练集和测试集拆分
        result_train, result_test = result[:len(self.y_train)], result[len(self.y_train):]
        if metric in norm_y_list :
            fitness_train = fitness._fitness_map[metric](self.y_train, pd.Series(result_train), np.ones(len(self.y_train)))
            fitness_test = fitness._fitness_map[metric](self.y_test, pd.Series(result_test), np.ones(len(self.y_test)))
        else:
            fitness_train = fitness._fitness_map[metric](self.ret_train, pd.Series(result_train), np.ones(len(self.y_train)))
            fitness_test = fitness._fitness_map[metric](self.ret_test, pd.Series(result_test), np.ones(len(self.y_test)))
       
        return fitness_train, fitness_test    
    
    def evaluate_single_factor_given_ic(self, factor_expression):
        """
        评估单个因子的表现

        只生成一个evaluator对象，然后用这个evaluator来评估train和test的表现(截断) -  因为要兼容那些对于start_date敏感的因子
        """
        # self.X_all已经是根据train_start_date和test_end_date截断过的数据
        evaluator = FeatureEvaluator(_function_map, self.feature_names, self.X_all)

        result = evaluator.evaluate(factor_expression)  # 解析因子
        result = np.nan_to_num(result)
        # 训练集和测试集拆分
        result_train, result_test = result[:len(self.y_train)], result[len(self.y_train):]

        max_ic_train, up_r, dn_r = fitness._fitness_map['max_ic_train'](self.y_train, pd.Series(result_train), np.ones(len(self.y_train)))
        given_ic_test  = fitness._fitness_map['given_ic_test'](self.y_test, pd.Series(result_test), np.ones(len(self.y_test)), up_r, dn_r)

        return max_ic_train, given_ic_test 
 
    def remove_duplicate_columns(self, df):
        """
        移除DataFrame中的重复列。

        Args:
            df: 输入的DataFrame。

        Returns:
            DataFrame: 移除重复列后的DataFrame。
        """
        return df.loc[:, ~df.columns.duplicated()]

    def evaluate_single_factor_for_new_genes(self, factor_expression):
        """
        评估单个新生成因子的表现。
        """
        all_metrics = list(fitness._fitness_map.keys())
        all_metrics = [item for item in all_metrics if item not in ['max_ic_train', 'given_ic_test']]
        for metric in all_metrics:
            fitness_train, fitness_test = self.evaluate_single_factor(str(factor_expression), metric)

            print(f"expression_fitness_train = {fitness_train}")
            print(f"expression_fitness_test = {fitness_test}")

            self.best_programs_df_dedup.loc[
                self.best_programs_df_dedup['expression'] == factor_expression, f'fitness_{metric}_train'] = fitness_train
            self.best_programs_df_dedup.loc[
                self.best_programs_df_dedup['expression'] == factor_expression, f'fitness_{metric}_test'] = fitness_test
            
            
        max_ic_train_check, given_ic_test  = self.evaluate_single_factor_given_ic(factor_expression)  
        self.best_programs_df_dedup.loc[
            self.best_programs_df_dedup['expression'] == factor_expression, 'max_ic_train_check'] = max_ic_train_check
        self.best_programs_df_dedup.loc[
            self.best_programs_df_dedup['expression'] == factor_expression, 'given_ic_test'] = given_ic_test    
 
   
    def evaluate_factors(self):
        """
        评估新生成的因子并计算不同指标的表现。
        """
        self.best_programs_df_dedup = self.best_programs_df.drop_duplicates(subset=['expression'], keep='first')

        factor_expressions = [str(prog) for prog in self.est_gp._best_programs]
        factors_pred_train = self.est_gp.transform(self.X_train)
        factors_pred_test = self.est_gp.transform(self.X_test)

        self.pred_data_df_train = pd.DataFrame(factors_pred_train, columns=factor_expressions)
        self.pred_data_df_test = pd.DataFrame(factors_pred_test, columns=factor_expressions)

        self.pred_data_df_train = self.remove_duplicate_columns(self.pred_data_df_train)
        self.pred_data_df_test = self.remove_duplicate_columns(self.pred_data_df_test)

        logger.info('使用eval解析本轮的因子表达式，并使用所有的metric方法跑一轮fitness')
        for factor_expression in self.pred_data_df_train.columns:
            self.evaluate_single_factor_for_new_genes(factor_expression)

        self.best_programs_df_dedup.to_csv(self.model_folder / 'best_programs_df.csv.gz', index=False,
                                           compression='gzip')



    def save_final_results(self):
        """
        fct_generate的一部分
        保存最终结果到总表中。
        """

        if self.total_factor_file_path.exists():
            total_factor_df = pd.read_csv(self.total_factor_file_path, compression='gzip')
            total_factor_df = pd.concat([total_factor_df, self.best_programs_df_dedup], ignore_index=True).drop_duplicates(
                subset=['expression'], keep='first')
        else:
            total_factor_df = self.best_programs_df_dedup

        total_factor_df.to_csv(self.total_factor_file_path, index=False, compression='gzip')
        print(f"saved {self.total_factor_file_path}")
        print("Factor generation process completed.")

    def fct_generate(self, random_state=None):
        """
        生成因子并进行初步评估。
        """
        logger.info('----开始执行----')

        self.run_genetic_programming(random_state)
        self.process_best_programs()
        self.save_initial_results()
        self.evaluate_factors()
        self.save_final_results()

    def execute_task(self, random_state=None):
        """
        执行单个遗传编程任务。
        """
        try:
            self.fct_generate(random_state)
            print("Task completed successfully.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")


    def run(self):
        """
        根据配置运行任务，支持单次执行和循环执行模式。
        """
        execution_mode = self.config.get('execution_mode', 'once')

        if execution_mode == 'once':
            self.execute_task()
        elif execution_mode == 'loop':
            interval = self.config.get('execution_interval', 30)  # 默认30秒
            print(f"Starting loop mode with {interval} seconds interval.")
            schedule.every(interval).seconds.do(self.execute_task)
            while True:
                schedule.run_pending()
                time.sleep(1)
        else:
            print(f"Unknown execution mode: {execution_mode}")




 
    
    
    def save_total_factor_df(self, total_factor_df):
        """
        保存总因子数据框。
        """
        total_factor_df.to_csv(self.total_factor_file_path, index=False, compression='gzip')
        print(f"Updated factor evaluations saved to {self.total_factor_file_path}")    
    
    def evaluate_existing_factors(self):
        """
        ** 独立于gplearn，单独运行.
        直接读取当前因子库的total_factor_df文件，对所有因子进行评估，并更新文件.
        评估现有因子库中的所有因子
        """

        total_factor_df = self.load_total_factor_df()
        if total_factor_df is not None:

            all_metrics = list(fitness._fitness_map.keys())
            all_metrics = [item for item in all_metrics if item not in ['max_ic_train', 'given_ic_test']]
            for index, row in total_factor_df.iterrows():
                factor_expression = row['expression']
                for metric in all_metrics:
                    train_col = f'fitness_{metric}_train'
                    test_col = f'fitness_{metric}_test'

                    if train_col not in total_factor_df.columns or test_col not in total_factor_df.columns:
                        fitness_train, fitness_test = self.evaluate_single_factor(factor_expression, metric)
                        total_factor_df.loc[index, train_col] = fitness_train
                        total_factor_df.loc[index, test_col] = fitness_test
                max_ic_train_check, given_ic_test  = self.evaluate_single_factor_given_ic(factor_expression)
                total_factor_df.loc[index, 'max_ic_train_check'] = max_ic_train_check
                total_factor_df.loc[index, 'given_ic_test'] = given_ic_test

            self.save_total_factor_df(total_factor_df)

  

    def read_and_cal_metrics(self):
        '''
        读取之前生成的因子值，并计算每一个因子值的metric
        注意这里计算的metric都是返回一个值的'''

        z = pd.read_csv('/home/etern/crypto/gp-crypto/elite_pool/factor_selected.csv')
        z.drop_duplicates(inplace=True)
        
        all_metrics = list(fitness._fitness_map.keys())
        all_metrics = [item for item in all_metrics if item not in ['max_ic_train', 'given_ic_test']]
        
        
        for index, row in z.iterrows():
            factor_expression = row['expression']
            for metric in all_metrics:
                train_col = f'fitness_{metric}_train'
                test_col = f'fitness_{metric}_test'

                try:
                    print(f'此时要计算{self.sym}的metric对应的因子是{factor_expression}')
                    fitness_train, fitness_test = self.evaluate_single_factor(factor_expression, metric)
                    z.loc[index, train_col] = fitness_train
                    z.loc[index, test_col] = fitness_test
                except Exception as e:
                    print(e)
            max_ic_train_check, given_ic_test  = self.evaluate_single_factor_given_ic(factor_expression)
            z.loc[index, 'max_ic_train_check'] = max_ic_train_check
            z.loc[index, 'given_ic_test'] = given_ic_test
                
        # 保存到gp_models文件夹下
        self.save_total_factor_df(z)
        

    def read_and_pick(self):
        # 遍历因子的总表，注意是总表
        elite_pool = []
        csv_path = f'gp_models/{self.sym}_{self.freq}_{self.y_train_ret_period}_{self.start_date_train}_{self.end_date_train}_{self.start_date_test}_{self.end_date_test}.csv.gz'
        # 读取因子值
        z = pd.read_csv(csv_path)

        # 遍历每一个因子，如果符合条件，就加入elite pool中
        for index,row in z.iterrows():
            '''
            此处写筛选因子的逻辑
            
            '''
            cond1 = (row['fitness_sharp_train'] > 2)
            cond2 = (row['fitness_sharp_test'] > 2)  
            cond3 = (row['fitness_avg_pic_train'] > 0.005)
            cond4 = (row['fitness_avg_pic_test'] > 0.005)
            try:
                # 这里写入筛选因子的逻辑，根据不同的metric筛选出因子池，进行拟合
                if cond1 and cond2 and cond3 and cond4:
                    # 记录符合标准的expression
                    elite_pool.append(row['expression'])
            except Exception as e:
                    print(f'an error occurred with {e}')
        elite_pool_size = len(elite_pool)
        print(f'挑选出来{self.sym}的因子池数量{elite_pool_size}')
        return elite_pool


    def plot_and_save_three_series(self,price_train, price_test, pnl_train, pnl_test, rs_train, rs_test ,title, index) -> None:
        fig, axs = plt.subplots(3, 2, figsize=(10, 12))
        fig.suptitle(f'{self.sym}_{index}_{title}', fontsize=16)

        axs[0, 0].plot(self.train_index,price_train, 'r-')
        axs[0, 0].set_title('price_train')
        axs[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 设置日期格式
        axs[0, 0].xaxis.set_major_locator(mdates.MonthLocator())  # 设置日期间隔

        axs[1, 0].plot(self.train_index,pnl_train, 'g-')
        axs[1, 0].set_title('pnl_train')
        axs[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 设置日期格式
        axs[1, 0].xaxis.set_major_locator(mdates.MonthLocator())  # 设置日期间隔

        axs[2, 0].plot(self.train_index,rs_train, 'b-')
        axs[2, 0].set_title('rolling sharp_train')
        axs[2, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 设置日期格式
        axs[2, 0].xaxis.set_major_locator(mdates.MonthLocator())  # 设置日期间隔

        axs[0, 1].plot(self.test_index,price_test, 'r-')
        axs[0, 1].set_title('price_test')
        axs[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 设置日期格式
        axs[0, 1].xaxis.set_major_locator(mdates.MonthLocator())  # 设置日期间隔

        axs[1, 1].plot(self.test_index,pnl_test, 'g-')
        axs[1, 1].set_title('pnl_test')
        axs[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 设置日期格式
        axs[1, 1].xaxis.set_major_locator(mdates.MonthLocator())  # 设置日期间隔

        axs[2, 1].plot(self.test_index,rs_test, 'b-')
        axs[2, 1].set_title('rolling sharp_test')
        axs[2, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 设置日期格式
        axs[2, 1].xaxis.set_major_locator(mdates.MonthLocator())  # 设置日期间隔

        plt.xticks(rotation=45)
        plt.tight_layout()
        

        # 生成文件名并保存
        current_date = datetime.now().strftime("%Y%m%d")
        print('保存收盘价、因子的rolling sharp和因子的pnl')
        filename = f"{self.sym}_{self.freq}_{self.y_train_ret_period}_{self.start_date_train}_{self.end_date_train}_{self.start_date_test}_{self.end_date_test}/factor_drawings/factor_{index}_.png"
        plt.savefig(filename)
        # 关闭图像窗口
        plt.close(fig)

        print(f"因子的pnl和rolling sharp已成功保存")


    def hist_draw(self,factor_series,index):
        '''绘制因子的分布图'''

        plt.figure(figsize=(10, 6))
        pd.Series(factor_series).hist(bins=100, edgecolor='green')
        plt.title('Distribution of single factor')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        file_path = f"{self.sym}_{self.freq}_{self.y_train_ret_period}_{self.start_date_train}_{self.end_date_train}_{self.start_date_test}_{self.end_date_test}/factor_drawings/fcthist_{index}.png"
        plt.savefig(file_path)
        plt.close()
        print(f"-------------------因子{index}的分布图已保存成功！！！-----------------------")

    def backtest_single_factor(self, factor_expression, metric, index, df):
        """
        评估单个因子的简单回测，返回序列
        """
        # self.X_all已经是根据train_start_date和test_end_date截断过的数据
        evaluator = FeatureEvaluator(_function_map, self.feature_names, self.X_all)

        factor_series = evaluator.evaluate(factor_expression)  # 解析因子
        #画因子分布
        self.hist_draw(factor_series,index)
        #画因子的ic_decay
        pd.Series({n:np.corrcoef(factor_series,df[f'return+{n}'].values)[0,1] for n in range(1,50,2)}).plot()
        plt.title(f"{factor_expression}")
        plt.xlabel("Lag (n)")
        plt.ylabel("ic")
        plt.savefig(f"{self.sym}_{self.freq}_{self.y_train_ret_period}_{self.start_date_train}_{self.end_date_train}_{self.start_date_test}_{self.end_date_test}/factor_drawings/ic_deacy_{index}.png")  # 保存为 PNG 文件
        plt.close()
        
        # 训练集和测试集拆分
        result_train, result_test = factor_series[:len(self.y_train)], factor_series[len(self.y_train):]
        fitness_train = fitness._backtest_map[metric](self.ret_train, pd.Series(result_train), np.ones(len(self.y_train)))
        fitness_test = fitness._backtest_map[metric](self.ret_test, pd.Series(result_test), np.ones(len(self.y_test)))

        return fitness_train, fitness_test
    
    
    def elite_factors_further_process(self,cal_new_metric=False):
        # 第一步解析之前生成的的因子，并计算各个因子的metrics，最后生成一个表格，只用计算一次
        # 计算每个因子的metrics，生成一个表格，用于后面的筛选过程
        if cal_new_metric:
            self.read_and_cal_metrics()
        # 第二步读取上一步生成的表格，并筛选合适的因子
        elite_pool = self.read_and_pick()
        z = pd.DataFrame()
        #准备ic_decay的数据
        space = range(1,50,2)
        df = self.ohlc['c'].to_frame()
        for i in space:
            df.loc[:,f'return+{i}'] = np.log(df.loc[:,'c']).shift(-i) - np.log(df.loc[:,'c'])
            # df[f'return+{i}'] = np.where(df[f]>0,df[f'return+{i}']-fee, np.where(df[f]<0,df[f'return+{i}']+fee,0))
            df.replace([np.inf, -np.inf, np.nan], 0.0,inplace = True)      
            
        # 遍历elite pool
        for index,i in enumerate(elite_pool):
            train_rs, test_rs = analyzer.backtest_single_factor(i,'rolling_sharp',index,df)
            train_pnl,test_pnl = analyzer.backtest_single_factor(i,'pnl',index,df)
            # TODO：画图和统计值同时具备
            close_train = pd.Series(self.close_train.reset_index(drop=True))
            close_test = pd.Series(self.close_test.reset_index(drop=True))
            # 下面开始画图
            try:
                self.plot_and_save_three_series(close_train,close_test,pd.Series(train_pnl),pd.Series(test_pnl), pd.Series(train_rs),pd.Series(test_rs),i, index)
                z = pd.DataFrame(elite_pool, columns=['expression'])
            except:
                pass
        # 保存
        print('下面一步保存筛选出来的因子值')
        z.to_csv(f'{self.sym}_{self.freq}_{self.y_train_ret_period}_{self.start_date_train}_{self.end_date_train}_{self.start_date_test}_{self.end_date_test}/factors_elite_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv')
        print('---------------------此轮筛选出来的因子值已经保存成功！！----------------------------')
        return elite_pool


        
    def calculate_factors_values(self,factor_expression):
        evaluator = FeatureEvaluator(_function_map, self.feature_names, self.X_all)
        result = evaluator.evaluate(factor_expression)  # 解析因子
        result_train, result_test = result[:len(self.y_train)], result[len(self.y_train):]
        return result_train,result_test

    def go_model(self,exp_pool):
        '''组合因子生成model'''
        X_train,X_test = [],[]
        for i in exp_pool:
            X_train.append(self.calculate_factors_values(i)[0])
            X_test.append(self.calculate_factors_values(i)[1])
        X_train,X_test = np.array(X_train).T,np.array(X_test).T,
        model = LinearRegression()
        model.fit(X_train,self.ret_train.reshape(-1,1))
        # 根据历史分位,并确定放大缩小倍数
        pos_train = model.predict(X_train).flatten()
        min_val = abs(np.percentile(pos_train, 99))
        max_val = abs(np.percentile(pos_train, 1))
        
        pos_ = model.predict(X_test).flatten()
        scale_n = 2/(min_val+max_val)
        # 大概映射到合理整数区间
        pos_train = pos_train* scale_n
        pos_train = pos_train.clip(-5,5)
        
        pos = pos_ * scale_n
        pos = pos.clip(-5,5)
        model_file = f'{self.sym}_{self.freq}_{self.y_train_ret_period}_{self.start_date_train}_{self.end_date_train}_{self.start_date_test}_{self.end_date_test}/model.pkl'
        with open(model_file, 'wb') as file:
            pickle.dump(model, file)
        print('模型文件保存成功！')
        print(f'模型系数{model.coef_},模型的截距为{model.intercept_}')
        print(f'查看pos{pos.shape}')
        return pos,pos_train

    def real_trading_simulator(self,pos:np.array, data_range = 'test', fee = 0.0005):
        '''模拟真实的交易场景'''
        # 获得下一个bar的open price,和当前bar的close price
        if data_range == 'train':
            next_open = np.concatenate((self.open_train[1:], np.array([0])))
            close = self.close_train
        elif data_range == 'test':
            next_open = np.concatenate((self.open_test[1:], np.array([0])))
            close = self.close_test
        else:
            open_all = pd.concat([self.open_train,self.open_test])
            next_open = np.concatenate((open_all[1:], np.array([0])))
            close =  pd.concat([self.close_train,self.close_test])  
        
        real_pos = pos   # 实际的开仓仓位
        # 获得每次仓位的变化
        pos_change = np.concatenate((np.array([0]), np.diff(real_pos)))
        # 决定以什么价位开仓，当仓位变化大于0时，需要买进，更差的价格是close和next open的最大值
        # 当仓位变化小于0时，需要卖出，更差的价格是close和next open的最小值
        which_price_to_trade = np.where(pos_change, np.maximum(close, next_open), np.minimum(close, next_open))

        next_trade_close = np.concatenate((which_price_to_trade[1:], np.array([which_price_to_trade[-1]])))
        rets = np.log(next_trade_close) - np.log(which_price_to_trade) 
        fee = fee  # 万5手续费
        gain_loss = real_pos * rets - abs(pos_change) * fee
        copy_gain_loss = np.copy(gain_loss)
        # 计算pnl
        pnl = copy_gain_loss.cumsum()

        # 计算胜率
        win_rate_bar = np.sum(gain_loss > 0) / len(gain_loss)
        # 计算盈亏比
        avg_gain_bar = np.mean(gain_loss[gain_loss > 0])
        avg_loss_bar = np.abs(np.mean(gain_loss[gain_loss < 0]))
        profit_loss_ratio_bar = avg_gain_bar / avg_loss_bar if avg_loss_bar != 0 else np.inf
        # 计算总交易次数（仓位变动的次数）
        # 计算年化收益率
        annual_return = np.mean(gain_loss)*self.annual_bars
        sharpe_ratio = annual_return / (np.std(gain_loss)*np.sqrt(self.annual_bars))
        #计算回撤和卡玛
        peak_values = np.maximum.accumulate(pnl)
        drawdowns = (pnl - peak_values) / peak_values
        max_drawdown = np.min(drawdowns)
        Calmar_Ratio = annual_return / -max_drawdown if max_drawdown != 0 else np.inf
        
        
        return pnl,{"Win Rate_bar": win_rate_bar,
                "Profit/Loss Ratio_bar": profit_loss_ratio_bar,
                "Annual Return": annual_return,
                "MAX_Drawdown": max_drawdown,
                "Sharpe Ratio": sharpe_ratio,
                "Calmar Ratio": Calmar_Ratio
                }

    
    def real_trading_simulation_plot(self,pos,pos_train, fee=0.0005):
        
        net_values_train,metrics_train = self.real_trading_simulator(pos_train,'train', fee)
        pos_index_train = self.train_index
        close_train = self.close_train

        net_values,metrics = self.real_trading_simulator(pos,'test', fee)
        pos_index = self.test_index
        close_test = self.close_test
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 12))
        fig.suptitle(f'{self.sym} real trading', fontsize=16)

        axs[0, 0].plot(pos_index_train,close_train, 'b-')
        axs[0, 0].set_title('price_train')
        axs[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 设置日期格式
        axs[0, 0].xaxis.set_major_locator(mdates.MonthLocator())  # 设置日期间隔


        axs[1, 0].plot(pos_index_train,net_values_train, 'r-')
        axs[1, 0].set_title('pnl_train')
        axs[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 设置日期格式
        axs[1, 0].xaxis.set_major_locator(mdates.MonthLocator())  # 设置日期间隔

        axs[0, 1].plot(pos_index,close_test, 'b-')
        axs[0, 1].set_title('price_test')
        axs[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 设置日期格式
        axs[0, 1].xaxis.set_major_locator(mdates.MonthLocator())  # 设置日期间隔


        axs[1, 1].plot(pos_index,net_values, 'r-')
        axs[1, 1].set_title('pnl_test')
        axs[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 设置日期格式
        axs[1, 1].xaxis.set_major_locator(mdates.MonthLocator())  # 设置日期间隔


        # Annotate metrics on the plot
        annotation_text = "\n".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
        plt.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction',
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.grid(True)
        plt.tight_layout()
        dir_path = Path(
            f'{self.sym}_{self.freq}_{self.y_train_ret_period}_{self.start_date_train}_{self.end_date_train}_{self.start_date_test}_{self.end_date_test}/real_trading')
        dir_path.mkdir(parents=True, exist_ok=True)
        # Save the plot
        plt.savefig(f'{self.sym}_{self.freq}_{self.y_train_ret_period}_{self.start_date_train}_{self.end_date_train}_{self.start_date_test}_{self.end_date_test}/real_trading/net_value_performance.png')
    
    



if __name__ == '__main__':
    yaml_file_path = 'parameters.yaml'
    analyzer = GPAnalyzer(yaml_file_path)

    # Option1 - 运行遗传编程任务，一批批的生成新的因子
    analyzer.run()

    # # Option2 - 直接评估现有因子库中的所有因子， 执行metric打分 （可以执行另外的一组metric，重新定义另一个metric_dict即可）。不需要运行gplearn.
    analyzer.evaluate_existing_factors()

    ## Option3 - 寻找出优秀的因子，并绘制出滚动夏普和pnl曲线,再加工模型模型

    # analyzer.read_and_cal_metrics()
    # exp_pool = analyzer.elite_factors_further_process()
    # pos_test,pos_train = analyzer.go_model(exp_pool)
    # analyzer.real_trading_simulation_plot(pos_test,pos_train,0.000)

    # # Option4 - 解析所有的因子表达式
    # # 读取csv.gz文件
    # gz = pd.read_csv('factors_selected.csv')
    # use_historical_data = False
    # if use_historical_data:
    #     evaluator = FeatureEvaluator(_function_map, analyzer.feature_names, analyzer.X_all)
    #     factor_values = pd.DataFrame()
    #     for exp in gz['expression']:
    #         print(f'这一次要解析的因子值是{exp}')
    #         try:
    #             factor_values[exp] = evaluator.evaluate(exp) # 解析因子式为因子值
    #             print('---------------成功！-----------------')
    #         except Exception as e:
    #             print(e)
    #             print('---------------失败！-----------------')
    # else:
    #     import originalFeature
        
    #     mark_data_raw = pd.read_csv('C:/Users/Yidao/Desktop/raw_data_markdata_5m.csv',index_col=0)
    #     mark_data_raw['date'] = pd.to_datetime(mark_data_raw['date'])
    #     mark_data_raw.set_index('date', inplace=True)
        
    #     base_feature = originalFeature.BaseFeature(mark_data_raw)
    #     mark_data = base_feature.init_feature_df
    #     evaluator = FeatureEvaluator(_function_map, mark_data.columns, mark_data.values)
    #     factor_values = pd.DataFrame()
    #     for exp in gz['expression']:
    #         print(f'这一次要解析的因子值是{exp}')
    #         try:
    #             factor_values[exp] = evaluator.evaluate(exp) # 解析因子式为因子值
    #             print('---------------成功！-----------------')
    #         except Exception as e:
    #             print(e)
    #             print('---------------失败！-----------------')


    # Option5 - 多进程并行筛选因子
    # def process_file(yaml_file_path):
    #     analyzer = GPAnalyzer(yaml_file_path)
    #     exp_pool = analyzer.elite_factors_further_process(True)
    #     pos,pos_train = analyzer.go_model(exp_pool)
    #     analyzer.real_trading_simulation_plot(pos,pos_train)
    #
    # # 文件名列表
    # file_names = ['parameters.yaml','parameters1.yaml','parameters2.yaml','parameters3.yaml',]
    #
    # # 使用多进程处理文件
    # with ProcessPoolExecutor() as executor:
    #     executor.map(process_file, file_names)
    
