
import numpy as np
import pandas as pd
import re
from functions import _function_map
import originalFeature
import multiprocessing
from functools import partial


# 定义函数映射

class FeatureEvaluator:
    def __init__(self, program_dict, feature_list, feature_data, window_size=None):
        """
        初始化 FeatureEvaluator 类。

        参数:
        - program_dict: dict，自定义的计算程序字典。
        - feature_list: list，包含特征名称的列表。
        - feature_data: ndarray，包含特征数据的数组，形状为 (m, n)，其中 m 是样本数量，n 是特征数量。
        """
        self.program_dict = program_dict
        self.feature_list = feature_list
        self.window_size = window_size
        self.update_feature_data(feature_data)

    def update_feature_data(self, feature_data):
        if isinstance(feature_data, pd.DataFrame):
            self.feature_dict = {feature: feature_data[feature].values for feature in self.feature_list}
        else:
            self.feature_dict = {self.feature_list[i]: feature_data[:, i] for i in range(len(self.feature_list))}
        if self.window_size:
            self.feature_dict = {k: v[-self.window_size:] for k, v in self.feature_dict.items()}

    def add_new_row(self, new_row):
        """
        添加新的一行数据到特征字典中。

        参数:
        - new_row: 可以是 pandas.Series, numpy.ndarray, 或者 list 类型的新数据行。
        """
        if isinstance(new_row, pd.Series):
            # 如果是 Series，使用索引名称
            for feature in self.feature_list:
                if feature in new_row.index:
                    value = new_row[feature]
                    self.feature_dict[feature] = np.append(self.feature_dict[feature], value)
                    if self.window_size:
                        self.feature_dict[feature] = self.feature_dict[feature][-self.window_size:]
                else:
                    print(f"警告: 特征 '{feature}' 在新行中未找到。")

        elif isinstance(new_row, (np.ndarray, list)):
            # 如果是 ndarray 或 list，假设顺序与 feature_list 相同
            if len(new_row) != len(self.feature_list):
                raise ValueError("新行的长度与特征列表长度不匹配。")

            for feature, value in zip(self.feature_list, new_row):
                self.feature_dict[feature] = np.append(self.feature_dict[feature], value)
                if self.window_size:
                    self.feature_dict[feature] = self.feature_dict[feature][-self.window_size:]

        else:
            raise TypeError("new_row 必须是 pandas.Series, numpy.ndarray, 或 list 类型。")

    def evaluate(self, expression):
        """
        评估给定的因子表达式。
        参数:
        - expression: str，因子表达式的字符串。
        返回:
        - result: ndarray，评估表达式后的结果数组。
        """
        # 将 feature_dict 和 program_dict 合并为上下文
        context = {**self.feature_dict, **self.program_dict}

        try:
            # 使用 eval 评估表达式
            result = eval(expression, {"__builtins__": None}, context)
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {e}")

        return result

    def multiprocess_evaluate(self, expressions, num_processes=None):
        """
        多进程的版本，与evaluate完全隔离独立的写法，方便对比调试
        """
        if num_processes is None:
            num_processes = multiprocessing.cpu_count() - 1

        # Create a pool of worker processes
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use partial to create a function with fixed self parameter
            eval_func = partial(self._evaluate_single_expression)

            # Map the evaluation function to all expressions
            results = pool.map(eval_func, expressions)

        # Combine results into a dictionary
        return dict(zip(expressions, results))


    def _evaluate_single_expression(self, expression):
        """
        与evaluate完全隔离独立的写法，方便对比调试
        Helper method to evaluate a single expression.
        This method will be called by each worker process.
        """
        context = {**self.feature_dict, **self.program_dict}
        print(f'此次要解析的因子值是{expression}')
        try:
            result = eval(expression, {"__builtins__": None}, context)
        except Exception as e:
            print(f"Error evaluating expression '{expression}': {e}")
            result = None
        return result[-1]


if __name__ == "__main__":
    # 自定义的计算程序字典
    program_dict = {
        'add': lambda x, y: x + y,
    }

    # 特征列表
    feature_list = ['feature1', 'feature2', 'feature3']

    # 特征数据
    feature_data = np.array([
        [1, 2, 3, 3, 3],
        [4, 5, 6, 6, 6],
        [7, 8, 9, 9, 9]
    ])

    # 初始化 FeatureEvaluator 类
    evaluator = FeatureEvaluator(program_dict, feature_list, feature_data)

    # 评估因子表达式，包括递归嵌套
    expression = "add(add(feature1, feature2), add(feature3, feature2))"
    result = evaluator.evaluate(expression)

    print("Result:", result)
