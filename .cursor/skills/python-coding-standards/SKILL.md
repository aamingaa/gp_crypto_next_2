---
name: python-coding-standards
description: "Python coding standards for this repository's quantitative research scripts. Use when writing or reviewing Python files that work with numpy, pandas, matplotlib, time-series features, labels, or backtest-style analysis."
origin: ECC
---

# Python 编码规范

适用于当前仓库的 Python 编码规范，重点面向 `numpy`、`pandas`、`matplotlib`、时间序列特征工程、标签生成与回测分析脚本。

## 何时启用

- 编写或评审当前仓库中的 Python 代码
- 修改基于 `numpy` / `pandas` 的数据处理、指标计算或标签逻辑
- 优化脚本性能，但又不希望引入与现有项目不一致的工程化结构

## 核心原则

- 优先可读性，其次才是“聪明”的写法
- 优先最小改动，贴近现有模块组织方式
- 对数值计算保持显式和可验证
- 默认避免隐式状态共享和隐藏副作用
- 出现性能优化时，先保证结果等价，再追求速度

## 修改前检查

开始改 Python 代码前，先做这些事：

- 找出本次需求必须改动的最小文件集合
- 先读附近代码，模仿局部风格，而不是单独发明一套新规范
- 检查是否已有类似函数、数据处理流程或验证方式，优先在现有逻辑上扩展
- 确认本次改动主要属于哪一类：数值计算、数据清洗、标签逻辑、可视化、验证脚本
- 明确哪些行为不能顺手改动，例如列名语义、返回结构、索引约定、绘图输出习惯

## 执行清单

执行时可复制并更新下面这份清单：

```markdown
任务进度：
- [ ] 明确本次需求的精确范围
- [ ] 确认最小改动文件集合
- [ ] 检查周边代码风格和现有实现模式
- [ ] 明确涉及的索引、窗口、列名和返回值约定
- [ ] 如涉及数值优化，先确定基线实现和对比方式
- [ ] 实现最小可行修改
- [ ] 验证修改后的结果、边界行为和项目原有流程一致
- [ ] 检查涉及文件的 lint / 运行问题
- [ ] 勾选已完成项，并总结风险与后续事项
```

## 项目适配原则

当前项目更接近研究脚本与函数模块，而不是大型框架应用，因此默认遵循这些约定：

- 优先写清晰的顶层函数，而不是为了抽象而引入类
- 新逻辑优先并入现有模块，除非职责已经明显失控；并入时尽量不要改动原有逻辑路径
- 保持与周边文件一致的注释密度、命名风格和输出方式
- 不要把简单分析脚本过度改造成复杂工程结构

## 命名规范

```python
# ✅ 模块 / 文件名: snake_case
triple_barrier.py
norm_data_check.py

# ✅ 函数 / 变量: snake_case
def forming_barriers_fast(close, events, pt_sl, molecule):
    ...

rolling_rank = series.rolling(window=window).rank(pct=True)

# ✅ 常量: UPPER_SNAKE_CASE
DEFAULT_WINDOW = 500
MAX_PLOT_POINTS = 10_000

# ✅ 类名: PascalCase，仅在确实需要状态封装时使用
class BacktestConfig:
    ...
```

## 导入规范

- 标准库、第三方库、本地模块分组导入
- 避免未使用导入
- 常见别名保持行业惯例：`numpy as np`、`pandas as pd`
- 除非现有文件已如此组织，否则不要随意使用 `from x import *`

```python
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

## 函数设计

- 函数应聚焦单一职责：数据准备、指标计算、可视化、验证逻辑尽量分开
- 参数名尽量体现业务含义，不要滥用 `data`、`value`、`tmp`
- 返回值结构要稳定；返回多个对象时，顺序和含义要清晰
- 公共函数在收益较高时补充类型标注；已有文件未全面标注时，不要求机械补齐
- 对关键函数写简短 docstring，尤其是窗口、阈值、索引语义和返回值

```python
def cusum_filter(series: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """基于对称 CUSUM 过滤器返回事件时间索引。"""
    ...
```

## DataFrame 与 Series 约定

- 明确索引语义，尤其是时间索引是否已排序、是否允许重复
- 涉及时间序列切片前，优先确认 `index` 已排序
- 修改输入对象前，先判断是否需要复制，避免无意污染调用方数据
- 新列命名应简洁且可追踪，避免难以理解的缩写
- 对齐操作使用 `reindex`、`align`、`join` 等显式方式，不依赖隐含广播

```python
close = pd.Series(close, copy=True).astype(float)
close = close.sort_index()
close = close[~close.index.duplicated(keep="last")]
```

## 数值与性能规则

- 先保证数值逻辑正确，再做向量化或 `numpy` 优化
- 优化前后要保证边界行为一致，例如 `NaN`、`inf`、空窗口、重复值
- 向量化不应显著损害可读性；过于绕的写法可保留循环实现
- 随机过程必须显式控制种子，避免结果不可复现
- 避免静默吞掉数值异常；对 `NaN` / `inf` 的处理要写清楚

```python
np.random.seed(42)
result = np.nan_to_num(result, nan=0.0)
return np.clip(result, -clip, clip)
```

## 可变性与副作用

- 默认不要在函数内部原地修改传入的 `Series` / `DataFrame`
- 如果函数会写图、写文件、打印日志，应在命名或文档里体现
- 可视化和计算尽量分离，避免“顺手画图”污染核心计算函数
- 不要在库函数中保留大量调试 `print`；短期调试代码提交前应清理

## 错误处理

- 输入不满足前提时，优先尽早抛出带上下文的信息
- 对窗口大小、数组长度、索引缺失、列不存在等情况进行显式检查
- 除非确有必要，否则不要写宽泛的 `except Exception`
- `warnings.filterwarnings(...)` 仅在非常明确的局部场景下使用，避免全局屏蔽问题

```python
if n <= window:
    return result
```

## 绘图与输出

- 绘图代码与核心计算分离
- 文件输出路径不要硬编码到个人绝对路径，除非当前脚本就是本地一次性分析工具
- 图表标题、坐标轴、保存文件名应表达清楚分析目的
- 图表标题、坐标轴等应该用英文标注
- 对大数据量绘图，考虑采样或限制点数，避免交互卡顿

## 格式与风格

- 使用 4 空格缩进
- 一行只做一件事，避免把多个计算步骤塞进单条复杂表达式
- 长函数可以存在，但应按逻辑分段，并在必要处加少量注释
- 早返回优于深层嵌套
- 魔法数字在复用或语义重要时提取为具名常量

## 应避免的代码异味

- 同一份时间序列在多个地方重复“预处理”，却没有抽成局部辅助函数
- 变量名过短，导致无法判断是价格、收益、标签还是阈值
- 同时混用位置索引和标签索引，但没有显式说明
- 为了提速引入复杂写法，却没有任何结果一致性验证
- 在研究脚本里堆积大量与当前任务无关的框架代码

## 验证要求

- 新增或修改数值逻辑时，至少做一次结果对比
- 若存在“优化版”函数，应与基线实现核对输出一致性
- 对滚动窗口、首尾样本、空值、重复索引等边界做最小验证
- 如果没有正式测试，也应提供最小可复现的验证方式

## 评审检查清单

- 命名是否清楚表达了金融/时间序列语义
- 输入输出是否稳定，是否避免了隐式副作用
- `pandas` 对齐、切片、索引排序是否明确
- 数值处理是否显式覆盖 `NaN` / `inf` / 边界窗口
- 性能优化是否保留了可读性与结果一致性
- 注释和 docstring 是否说明了关键假设

**Remember**: 保持代码可读、结果可复现、数值假设可解释。这个仓库默认是研究型 Python 脚本环境，不要套用 Java/Spring 风格的结构和术语。
