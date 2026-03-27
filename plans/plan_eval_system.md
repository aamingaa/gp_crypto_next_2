# Plan: 评估体系改造

## 目标
扩展评估体系，补充非线性关系诊断信号；互信息/熵模块为低优先级，不做硬筛选。

## 范围
1. 现有主评估：保留 `avg_sic/avg_pic/sharp/calmar/max_dd` 等。
2. 新增辅助评估：
   - `mi_score`（互信息）
   - `entropy_gain_score`（信息增益/条件熵下降）

## 实施步骤（Checklist）
- [x] 在评估管线中增加 `mi_score` 与 `entropy_gain_score` 的计算入口（低优先级）。
- [x] 输出两项指标的分位数排名（percentile rank）字段。
- [x] 将排名字段写入结果表（如 `best_programs_df` / 总因子表）。
- [x] 确保 MI/熵不参与任何硬筛选 gate/filter 条件。
- [x] 增加排序一致性与稳定性检查（与 `avg_sic/avg_pic` 对比）。
- [x] 记录实现开关，支持快速启停该模块。

## 输出策略（关键约束）
1. 只输出分位数排名（percentile rank）。
2. 不作为 gate/filter 的硬阈值条件。
3. 作为因子诊断和排序参考字段写入结果表。

## 代码影响范围
- `fitness.py`
  - 增加 MI/熵打分函数（若作为 metric 注册）
- `main_gp_new.py`
  - 在评估环节新增排名列生成与落盘
- （可选）`genetic.py`
  - 若将 MI/熵用于训练 metric，才需要加入白名单

## 验证
1. 与 `avg_sic/avg_pic` 排序一致性（Spearman）。
2. Top 分位因子在 test 上稳定性。
3. 确认没有触发任何基于 MI/熵的硬筛选逻辑。

## 执行结果与验证记录
- 2026-03-27：
  - 在 `fitness.py` 新增 `mi_score` 与 `entropy_gain_score`，并注册至 `_fitness_map`（仅作为辅助评估指标）。
  - 在 `main_gp_new.py` 增加辅助评估开关 `eval_aux_metrics_enabled`（默认 `True`），并在评估输出中新增：
    - `fitness_mi_score_train/test_pct_rank`
    - `fitness_entropy_gain_score_train/test_pct_rank`
  - 新增排序诊断与稳定性检查：
    - 与 `avg_sic/avg_pic` 的 Spearman 相关性（train/test）
    - 新指标 train/test Top10% 重叠率
    - 结果输出到模型目录 `aux_metric_diagnostics.yaml`
  - 明确保留硬筛选逻辑仅使用原有列，MI/熵不进入 gate/filter 条件。
