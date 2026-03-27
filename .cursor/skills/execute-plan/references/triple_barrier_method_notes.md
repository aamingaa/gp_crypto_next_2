# Triple Barrier 方法说明（当前实现）

本文档沉淀 `triple_barrier.py` 当前脚本中的完整流程：数据准备、动态 CUSUM 事件抽样、Triple Barrier 标注、可视化与参数遍历落盘。

## 1. 方法总览

当前实现分为 4 个步骤：

1. 数据读取与预处理  
   - 读取 `ETHUSDT` 指定月份的 1h 数据  
   - `open_time` 转时间戳并设为索引  
   - `close` 转为 `float`，并计算 `log_close = log(close)`

2. 动态 CUSUM 事件检测  
   - 收益：`ret = log_close.diff()`  
   - 波动率：`sigma = ret.ewm(span=240).std()`  
   - 动态阈值：`threshold = (2.0 * sigma).bfill()`  
   - 事件点：`events = cusum_filter_dynamic(log_close, threshold)`

3. Triple Barrier 标注  
   - 调用：`get_barrier(close, events, pt_sl=[1.2, 1.0], max_holding=[0, 10], target=sigma)`  
   - 输出 `barrier_events`，包含：
     - `exit`: 退出时间
     - `price`: 入场价格
     - `ret`: 事件收益（按 side 调整后）
     - `side`: 方向（当前默认 1）

4. 可视化与网格统计  
   - 图 1：CUSUM 触发点（红点）  
   - 图 2：Barrier 结果点，按收益分为止盈/止损/垂直到期  
   - 参数网格遍历后保存统计 CSV

## 2. 关键函数说明

### `cusum_filter(series, threshold)`

- 作用：对称 CUSUM，固定阈值版本
- 输入：`threshold` 为标量 `float`
- 输出：触发事件时间 `pd.DatetimeIndex`

### `cusum_filter_dynamic(series, threshold_series)`

- 作用：动态阈值包装函数（不改动 `cusum_filter` 原行为）
- 输入：与时间索引对齐的阈值序列
- 处理规则：
  - 按 `diff.index` 重对齐阈值
  - 跳过非有限值或 `<=0` 的阈值点
- 输出：事件时间 `pd.DatetimeIndex`

### `get_barrier(close, enter, pt_sl, max_holding, target=None, side=None)`

- 作用：Triple Barrier 主标注函数
- 参数语义：
  - `pt_sl=[pt, sl]`：止盈/止损倍数
  - `max_holding=[days, hours]`：垂直障碍持有时长
  - `target`：波动目标（推荐传 `sigma`，则 `pt/sl` 为 target 的倍数）
- 返回：每个事件的退出时间、收益、方向等字段

## 3. 当前主流程默认参数

- 回看区间：`2025-01` 到 `2025-03`
- CUSUM：
  - `span=240`
  - `k=2.0`（阈值为 `k * sigma`）
- Barrier：
  - `pt_sl=[1.2, 1.0]`
  - `max_holding=[0, 10]`（10 小时）
  - `target=sigma`

## 4. 输出文件

- 图像输出：
  - `picture/cusum_event_marks.png`
  - `picture/barrier_event_marks.png`

- 网格统计输出：
  - `results/barrier_grid_results.csv`

CSV 主要字段包括：

- 参数字段：
  - `cusum_span`, `cusum_k`, `pt`, `sl`, `max_holding_days`, `max_holding_hours`
- 计数字段：
  - `cusum_events_count`, `barrier_events_count`
  - `take_profit_count`, `stop_loss_count`, `vertical_exit_count`
- 比例字段：
  - `take_profit_ratio_non_neutral`, `stop_loss_ratio_non_neutral`
  - `take_profit_ratio_total`, `stop_loss_ratio_total`, `vertical_exit_ratio_total`

## 5. 网格遍历空间（当前代码）

- `cusum_span_grid = [120, 240, 360]`
- `cusum_k_grid = [1.5, 2.0, 2.5]`
- `pt_sl_grid = [(1.0, 1.0), (1.2, 1.0), (1.5, 1.0)]`
- `max_holding_grid = [(0, 6), (0, 10), (0, 24)]`

总组合数：`3 * 3 * 3 * 3 = 81`。

## 6. 结果解读建议

1. 先看 `barrier_events_count`，过滤样本过少的组合  
2. 在样本数可接受前提下，比较 `take_profit_ratio_non_neutral`  
3. 同时看 `vertical_exit_ratio_total`，避免大量事件只靠垂直障碍退出  
4. 若偏向短线，优先观察更短 `max_holding` 的稳定性

## 7. 注意事项

- `target=None` 时，`pt_sl` 变成绝对收益阈值；若设置过大，几乎不会触发止盈止损  
- 建议继续使用 `target=sigma`，让 `pt/sl` 具备波动率自适应语义  
- 动态阈值初段有空值时已通过 `bfill()` 处理，但仍建议关注样本起始区间稳定性

