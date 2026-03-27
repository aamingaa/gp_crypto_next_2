---
name: validation-ladder
description: Validate quantitative or time-series code changes with a staged workflow: quick smoke check, directional correctness check, then strict validation without changing core logic. Use when users ask to verify logic, test with small samples first, add assertions, run multi-seed parameter grids, or assess stability metrics like win ratio and lift.
---

# Validation Ladder

用于量化/时序脚本改造后的验证闭环，核心思想：

1. 小样本先跑通（不追求完美）
2. 判断方向是否正确（关键指标是否改善）
3. 在不改主逻辑前提下做强验证（参数网格 + 多 seed + 断言）

## 适用场景

- 用户问“逻辑一定正确吗”“怎么验证是否正确”
- 新增标签/采样/权重/评估逻辑后，需要快速判断可靠性
- 运行时间较长，先希望快速小样本验证
- 需要稳定性证据（而不只是单次跑通）

## 禁忌

- 不要先改主逻辑再验证（先验证再谈重构）
- 不要只给单次结果就下结论“完全正确”
- 不要跳过断言与边界条件检查

## 执行清单

复制并更新：

```markdown
验证进度：
- [ ] 阶段1：小样本跑通（Smoke）
- [ ] 阶段2：方向正确性检查（Direction）
- [ ] 阶段3：强验证（Grid + Multi-seed + Assertions）
- [ ] 汇总结果（通过率、分布、失败样例）
- [ ] 结论分级（可用 / 有风险 / 不通过）
```

## 阶段1：小样本跑通（Smoke）

- 用最小可运行配置，优先验证“流程不断”
- 如脚本较慢，先降规模，例如：
  - `n-bars`: 200~400
  - `sample-len`: 20~50
  - `num-trials`: 5~20
- 只看：能否完整执行、是否有异常、输出字段是否齐全

输出模板：

```markdown
Smoke:
- status: pass/fail
- input_size: ...
- elapsed: ...
- key_outputs_present: [...]
```

## 阶段2：方向正确（Direction）

- 不看“绝对最优”，先看“方向是否符合预期”
- 举例（采样类逻辑）：
  - `uniqueness_lift > 0`
  - `seq_u_mean > std_u_mean`
  - `seq_win_ratio` 高于随机基线
- 结论只说“方向正确/不正确”，不要夸大为“完全正确”

输出模板：

```markdown
Direction:
- metric_a: ...
- metric_b: ...
- expected_direction: up/down
- observed_direction: up/down
- direction_ok: true/false
```

## 阶段3：强验证（不改主逻辑）

必须包含三部分：

1) 参数网格（小/中/大）
- 例如 `small / medium / large`

2) 多随机种子
- 推荐至少 3~5 个 seed

3) 关键断言（4~6 个）
- events 合法性（`t1 >= t0`、无 NaN）
- 权重非负与有限值
- 权重归一化或总量合理
- 抽样长度一致
- 输出列完整
- 索引/时间对齐合法

建议汇总指标：

- `seq_win_ratio` 分布（均值/最小/分位）
- `uniqueness_lift` 分布（均值/最小/正值占比）
- 失败样例列表（配置 + seed + 错误信息）

输出模板：

```markdown
Strict Validation:
- runs: ...
- assertion_failures: ...
- avg_seq_win_ratio: ...
- avg_uniqueness_lift: ...
- lift_positive_ratio: ...
- worst_case:
  - config: ...
  - seed: ...
  - lift: ...
- failures:
  - [config=..., seed=..., error=...]
```

## 结论分级

- **可用**：无断言失败，且方向指标在大多数配置为正
- **有风险**：可运行但分布不稳（例如 lift 正值占比偏低）
- **不通过**：存在断言失败或关键方向指标反向

## 回答用户时的标准措辞

- 不说“100%正确”
- 推荐表达：
  - “当前证据显示流程跑通且方向正确，但仍需更大样本或真实数据进一步确认。”
  - “强验证通过/未通过的依据是：通过率、分布稳定性、断言结果。”

## 快速命令思路（示例）

```bash
# 1) 小样本 smoke
python module.py --n-bars 240 --sample-len 24 --num-trials 10

# 2) 强验证（示意：在 here-doc 中跑网格+seed）
python - <<'PY'
# run grid + multi-seed + assertions, then print summary
PY
```

## 输出最小要求

最终汇报至少包含：

1. 跑通状态（是否可执行）
2. 方向判断（是否符合预期）
3. 强验证摘要（通过率、分布、失败样例）
4. 残余风险与下一步建议

## 配套模板

- 可直接运行的 here-doc 验证脚本模板见 [reference.md](reference.md)
