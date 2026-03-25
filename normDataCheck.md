# normDataCheck 模块说明（供 Skill / Agent 引用）

对应源码：`normDataCheck.py`

## 模块做什么

对收益率等一维序列做**滚动窗口高斯秩（Gauss Rank）归一化**，可选做**逆变换与分位数一致性检验**，并将归一化列当作**多空信号**与 `ret` 列一起做简单手续费回测。脚本末尾示例会按多组 `window` × `step` 网格扫参并把指标写入 CSV。

---

## 快速索引（该选哪个函数）

| 目标 | 调用 |
|------|------|
| 主用的逐点滚动高斯秩变换 | `norm(data, window, clip)` |
| pandas 向量化近似（与 `norm` 平局逻辑不完全相同） | `vectorized_gauss_rank_norm(data, window, clip)` |
| 归一化值 → 原始窗口上分位近似还原（原始尺度上的对比 / 自定义 PnL 等，见下文） | `inverse_norm(normalized_data, original_data, window)` |
| 打印逆变换误差与 Pearson/Spearman，可选保存诊断图 | `verify_inverse_transform(...)` |
| 对比原始/归一化分位数与秩相关，可选保存图 | `check_quantile_consistency(...)` |
| 标签列做多空 + 手续费，返回指标字典 | `evaluate_position_performance(data, label_col, window, fee_rate, is_plot)` |
| 生成 `YYYY-MM` 月份字符串列表（拼数据路径） | `_generate_date_range(start_date, end_date)` |

**数据约定（主脚本）：** `DataFrame` 需含 `close`；需预先有列 `ret`（例如下一期收益）。`evaluate_position_performance` 强依赖 `ret` 与 `label_col`。

---

## `norm(data, window=500, clip=6)`

**功能：** 在长度为 `window` 的历史窗口内，用当前点相对历史的经验分位数，经近似逆正态变换映射到近似标准正态；平局用固定种子的随机数打散。输出截断到 `[-clip, clip]`。

**参数：**

- `data`：一维序列（可转 `ndarray`）。
- `window`：滚动长度；下标 `i` 使用 `[i-window, i)` 的历史，**不包含**当前点在 rank 的“过去窗口”内（当前值参与分位计算）。
- `clip`：输出上下界。

**返回：** 与 `data` 等长的 `ndarray`。若 `len(data) <= window`，返回全零；否则前 `window` 个位置为 0。

**用法示例：**

```python
z = norm(df["ret_5"].values, window=500, clip=6)
df["ret_5_norm"] = z
```

---

## `vectorized_gauss_rank_norm(data, window=500, clip=6)`

**功能：** 用 `pandas` 的 `rolling().rank(pct=True)` 加微小高斯噪声做 dithering，再将分位映射到标准正态分位（设计上对应 `scipy.stats.norm.ppf`）。通常比 `norm` 快，但与 `norm` 的平局处理**不完全等价**。

**参数 / 返回：** 与 `norm` 类似；前 `window` 行在 rolling 后为 NaN，实现里会 `nan_to_num` 为 0 再 clip。

**用法示例：**

```python
z = vectorized_gauss_rank_norm(series.values, window=500, clip=6)
```

**实现注意：** 源码中应对 rolling 分位使用 `scipy.stats.norm.ppf`。若当前文件里误写成与本模块同名函数 `norm` 的 `.ppf`，会报错，应改为 `from scipy.stats import norm as stats_norm` 后调用 `stats_norm.ppf(...)`。

---

## `inverse_norm(normalized_data, original_data, window=2000)`

**功能：** 对某一时刻的归一化值，用标准正态 CDF（通过 `erf`）得到分位 `q`，再在**历史原始值**排序后的窗口中取对应位置，作为近似“逆映射”。

**与 PnL / 回测的关系（重要）：**

- 本模块里**默认的**仓位与 PnL 计算在 `evaluate_position_performance` 中完成：直接用**归一化后的标签列**与 `ret` 做多空与手续费，**不需要**、也**不会**先走 `inverse_norm`。
- **`inverse_norm` 的用途**是得到**原始收益（或原序列）尺度上**的近似还原序列 `recovered`，便于：
  - 与真实 `ret` 对比、做逆变换误差分析（见 `verify_inverse_transform`）；
  - 若你需要在**原始尺度**上另行定义收益、PnL、风险或与裸信号对比（例如用 `recovered` 与 `ret` 构造自定义盈亏逻辑），应**先**用 `inverse_norm` 再在其输出上计算。

**参数：**

- `normalized_data`、`original_data`：等长一维数组。
- `window`：须与正向 `norm` 使用的 `window` 一致。

**返回：** 与输入等长的 `ndarray`；短序列或前段按实现可能为 0。

**用法示例：**

```python
recovered = inverse_norm(z, raw_ret.values, window=2000)
# 若要在原始尺度上做自定义 PnL / 对比，在 recovered 与 ret 上自行构造逻辑（非 evaluate_position_performance 的默认路径）
```

---

## `verify_inverse_transform(original_data, normalized_data, window=2000, is_plot=False, step=1)`

**功能：** 调用 `inverse_norm` 得到 `recovered`，对齐长度后计算相对误差统计及 Pearson/Spearman；**结果主要通过 `print` 输出**。`is_plot=True` 时保存四宫格图到仓库内 `picture/verify_inverse_{window}_{step}.png`（路径在源码中写死，换环境需改路径）。

**参数：** `window` 与正向变换一致；`step` 仅用于输出文件名区分。

**返回：** 无（`None`）；不返回 DataFrame。

**用法示例：**

```python
verify_inverse_transform(
    raw_data["ret_5"].values,
    raw_data["ret_5_norm"].values,
    window=500,
    is_plot=True,
    step=5,
)
```

---

## `check_quantile_consistency(original_data, normalized_data, window=2000, is_plot=False, step=1)`

**功能：** 去掉前 `window` 点后，对原始与归一化序列算多档分位数表，并算排序后序列的 Pearson 相关；分正负子集再打印分位表。**主要输出为 print**。`is_plot=True` 时保存 `picture/quantile_consistency_{window}_{step}.png`。

**返回：** 无。

**用法示例：**

```python
check_quantile_consistency(
    raw_data["ret_5"].values,
    raw_data["ret_5_norm"].values,
    window=500,
    is_plot=True,
    step=5,
)
```

---

## `evaluate_position_performance(data, label_col, window=2000, fee_rate=0.0004, is_plot=False)`

**功能：** 从 `label_col` 读标签，`>0` 做多仓 `1`，否则做空 `-1`；`net_pnl = position * ret - |Δposition| * fee_rate`。跳过前 `window` 行后计算累积收益、回撤及多项统计指标。

**参数：**

- `data`：含 `ret` 与 `label_col` 的 `DataFrame`。
- `label_col`：归一化后的信号列名。
- `window`：评估起始跳过行数（与归一化暖机对齐）。
- `fee_rate`：单边换仓费率（按 `position` 变化计费）。
- `is_plot`：为 True 时弹出 matplotlib 图（累积收益与回撤）。

**返回：** `dict`，键为中文指标名，例如：`总收益率`、`年化收益率`、`年化夏普比率`、`最大回撤`、`胜率`、`换仓成本总计`、`方向准确率`、`年化收益-回撤比`、`总收益-手续费比` 等。

**用法示例：**

```python
perf = evaluate_position_performance(
    raw_data,
    "ret_5_norm",
    window=500,
    fee_rate=0.0004,
    is_plot=False,
)
```

---

## `_generate_date_range(start_date, end_date)`

**功能：** `start_date` / `end_date` 为 `'YYYY-MM'` 字符串，返回包含起止月份在内的每月 `'YYYY-MM'` 列表（按月递增）。

**返回：** `list[str]`。

**用法示例：**

```python
months = _generate_date_range("2025-01", "2025-08")
# ['2025-01', '2025-02', ...]
```

---

## 脚本主流程（文件末尾 `if __name__` 等价块）

1. 用 `_generate_date_range` 拼路径读取多个月份 zip，合并为 `raw_data`。
2. 计算 `ret`（下一根 K 线收益）。
3. 对 `window_list` 与 `step` 循环：算 `ret_{i}`、`ret_{i}_norm = norm(...)`，再 `evaluate_position_performance`，汇总到 `performance_records`。
4. 将结果写入 `picture/perf.csv`（路径以源码为准）。

Skill 若只需库函数，可忽略该块；若复现实验，需准备本地数据路径与列结构。
