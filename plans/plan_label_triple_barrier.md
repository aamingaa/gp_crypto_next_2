# Plan: Label 改造（Triple Barrier）

## 目标
将主要标签扩展为 Triple Barrier 路径收益标签，并明确事件样本机制与重叠处理。

## 标签设计
1. 连续标签：`tb_ret`（来自 Triple Barrier 事件退出收益）。
2. 离散标签：`tb_label`（`-1/0/1`，阈值由 `eps` 控制）。
3. 不在 label 层做 clip，保持路径信息完整。

## 实施步骤（Checklist）
- [ ] 在 `dataload.py` 增加 `predict_label = triple_barrier_ret | triple_barrier_cls` 分支。
- [ ] 接入 `triple_barrier.get_barrier_fast(...)` 生成事件退出收益并对齐索引。
- [ ] 输出连续标签 `tb_ret` 与离散标签 `tb_label`（不做 label 层 clip）。
- [ ] 明确并实现 `training_mode = event_only | bar_wise`。
- [ ] 明确并实现 `tb_label=0` 语义（`neutral` 或 `no_event`）。
- [ ] 处理重叠事件：实现/接入 uniqueness weighting。
- [ ] 检查并修复 train/test 边界的跨界重叠泄漏风险。
- [ ] 在 `main_gp_new.py` 路由新标签到 `y_train/y_test` 并补充配置说明。

## 事件样本核心问题（必须处理）

### 1) 标签重叠
- 相邻时点持有区间可重叠，样本非独立。
- 需要检查 train/test 边界是否跨界泄漏。
- 引入 uniqueness weighting（并发事件数倒数）作为样本权重因子之一。

### 2) 目标频率
- 明确训练口径：
  - `event_only`：仅事件 bars 训练（优先基线）
  - `bar_wise`：逐 bar 训练（需定义非事件 bars 处理）
- 明确 `tb_label=0` 语义：
  - `neutral` 或 `no_event`

## 代码影响范围
- `dataload.py`
  - 新增 `predict_label = triple_barrier_ret | triple_barrier_cls`
  - 调用 `triple_barrier.get_barrier_fast(...)`
  - 对齐索引并输出标签列
- `main_gp_new.py`
  - 读取并路由新标签到 `y_train/y_test`
  - 补充配置说明与分支注释

## 关键参数
1. `pt_sl`（如 `[1.5, 1.0]`）
2. `max_holding`（如 `[0, 2]`）
3. `eps`（离散标签阈值）
4. `training_mode = event_only | bar_wise`
5. `tb_label_zero_semantics = neutral | no_event`
6. `use_uniqueness_weight` 与 `uniqueness_weight_floor`

## 验证
1. 标签分布与有效样本率（`tb_ret`、`tb_label`）。
2. train/test 是否存在跨界重叠泄漏。
3. 开启/关闭 uniqueness weighting 的指标差异。
4. 与原标签方案的 Top 因子稳定性对比。

## 执行结果与验证记录
- 待执行
