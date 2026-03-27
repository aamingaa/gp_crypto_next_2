# 改造计划索引

本目录将总计划拆分为可独立执行的子计划，先确认再实现。

## 执行清单（打勾面板）
- [ ] `plan_filter_cleanup.md`（筛选逻辑清理）
- [ ] `plan_loss_refactor.md`（Loss/Fitness 改造）
- [ ] `plan_label_triple_barrier.md`（Triple Barrier Label）
- [x] `plan_eval_system.md`（评估体系改造）

说明：
1. 每开始执行一项，先把对应项标记为 `[-]`（进行中）。
2. 执行完成并验证后，改成 `[x]`。
3. 详细进度与时间戳记录在 `status.md`。

## 执行顺序（建议）
1. `plan_filter_cleanup.md`：先注释硬阈值筛选，解除人工门槛。
2. `plan_loss_refactor.md`：完成 margin-based loss/fitness 改造。
3. `plan_label_triple_barrier.md`：接入 Triple Barrier label 与事件采样机制。
4. `plan_eval_system.md`：评估体系扩展（MI/熵仅做分位数排名输出）。

## 子计划清单
- `plan_filter_cleanup.md`
- `plan_loss_refactor.md`
- `plan_label_triple_barrier.md`
- `plan_eval_system.md`
- `status.md`

## 通用规则
1. 先在计划文件确认，再改代码。
2. 每个子计划完成后补“执行结果与验证记录”。
3. 暂不做相关性筛选逻辑改动（冻结）。
