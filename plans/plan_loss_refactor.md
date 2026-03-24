# Plan: Loss/Fitness 改造（Margin-Based）

## 目标
以 margin-based directional penalty 重构方向惩罚，不把 GP 输出当分类概率；对极端行情错向进行加权重罚。

## 设计原则
1. 不做概率化假设（不把 `y_pred` 当 `p`）。
2. 使用 `margin_t = y_pred_t * y_t`。
3. `margin_t < 0` 判定为错方向。
4. 对 `|y_t|` 大样本赋予更高权重。
5. 对同向但 margin 较小样本施加轻度惩罚（可选软边界 `m0`）。

## 实施步骤（Checklist）
- [ ] 在 `fitness.py` 新增 margin 基础工具函数（`margin_t`、`w_t`、极端样本判定）。
- [ ] 实现 `extreme_wrong_sign_penalty`。
- [ ] 实现 `weighted_margin_score`。
- [ ] 实现 `asym_extreme_direction_score`（含 `a >> b` 与软边界 `m0`）。
- [ ] 在 loss 内实现极端行情裁剪（分位数或固定阈值），不改写原始标签。
- [ ] 将新指标接入 `fitness._fitness_map`。
- [ ] 在 `genetic.py` 增加对应 `metric` 白名单。
- [ ] 在 `main_gp_new.py`/配置说明中补充新 metric 用法。

## 计划实现的指标
1. `extreme_wrong_sign_penalty`
   - `penalty_t = w_t * I(margin_t < 0)`
2. `weighted_margin_score`
   - `penalty_t = w_t * max(0, -margin_t)`
3. `asym_extreme_direction_score`
   - `penalty_t = w_t * [a * max(0, -margin_t) + b * max(0, m0 - margin_t)]`，`a >> b`

融合模板：
- `score = reward(y, y_pred) - lambda * mean(penalty_t)`（越大越好）
- `reward` 默认 `mean(y * position(y_pred))` 或 `sharp_like`

## 极端行情抑制（仅在 loss 内）
1. 仅对 loss 内部变量做稳健裁剪：`clip(y_for_loss, q01, q99)` 或固定阈值。
2. 不修改原始标签（`tb_ret/tb_label` 保持原貌）。
3. 记录裁剪阈值与被裁比例，用于对照实验。

## 代码影响范围
- `fitness.py`：新增指标函数并接入 `_fitness_map`
- `genetic.py`：将新指标加入 `metric` 白名单
- `main_gp_new.py`：补充新 metric 说明

## 验证
1. 极端行情错向率下降（`|y|>q95`）。
2. `max_dd/calmar/sharp` 至少一项改善，且不显著恶化其余项。
3. train/test 同步观察，避免过拟合。

## 执行结果与验证记录
- 待执行
