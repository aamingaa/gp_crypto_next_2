# 执行状态看板

最后更新：2026-03-27

## 总状态
- [ ] `plan_filter_cleanup.md`
- [ ] `plan_loss_refactor.md`
- [ ] `plan_label_triple_barrier.md`
- [x] `plan_eval_system.md`

## 执行日志

### 2026-03-24
- 初始化状态看板。
- 当前均为待执行。

### 2026-03-27
- 完成 `plan_eval_system.md`：
  - 新增 `mi_score`、`entropy_gain_score` 指标计算入口并接入评估管线。
  - 输出两项指标在 train/test 的 percentile rank 列并写入结果表。
  - 增加 Spearman 一致性检查与 Top10% train/test 稳定性检查。
  - 增加开关 `eval_aux_metrics_enabled`，支持快速启停。
  - 确认 MI/熵不参与硬筛选条件。

## 使用方式
1. 开始某一项时，把对应状态改为 `[-]`。
2. 完成并验证后，改为 `[x]`。
3. 在“执行日志”追加当次改动摘要、验证结果、遗留问题。
