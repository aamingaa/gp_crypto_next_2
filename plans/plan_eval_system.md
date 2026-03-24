# Plan: 评估体系改造

## 目标
扩展评估体系，补充非线性关系诊断信号；互信息/熵模块为低优先级，不做硬筛选。

## 范围
1. 现有主评估：保留 `avg_sic/avg_pic/sharp/calmar/max_dd` 等。
2. 新增辅助评估：
   - `mi_score`（互信息）
   - `entropy_gain_score`（信息增益/条件熵下降）

## 实施步骤（Checklist）
- [ ] 在评估管线中增加 `mi_score` 与 `entropy_gain_score` 的计算入口（低优先级）。
- [ ] 输出两项指标的分位数排名（percentile rank）字段。
- [ ] 将排名字段写入结果表（如 `best_programs_df` / 总因子表）。
- [ ] 确保 MI/熵不参与任何硬筛选 gate/filter 条件。
- [ ] 增加排序一致性与稳定性检查（与 `avg_sic/avg_pic` 对比）。
- [ ] 记录实现开关，支持快速启停该模块。

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
- 待执行
