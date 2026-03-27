# Validation Templates (Here-Doc)

本文件提供可直接运行的验证脚本模板（Python here-doc 版），用于快速复用：

- 小样本跑通（Smoke）
- 方向正确性检查（Direction）
- 强验证（Grid + Multi-seed + Assertions）

---

## 1) 小样本跑通（Smoke）

```bash
python - <<'PY'
import json
import time
import traceback

# TODO: 替换为你的模块与入口
# from your_module import run_pipeline

cfg = {
    "n_bars": 240,
    "sample_len": 24,
    "num_trials": 10,
    "random_seed": 42,
}

t0 = time.time()
status = "pass"
err = None
key_outputs = []

try:
    # TODO: 修改为你的执行函数
    # out = run_pipeline(**cfg)
    out = {"events": [1], "weights": [1], "phi": [1]}  # 占位示例
    key_outputs = [k for k in ["events", "weights", "phi"] if k in out]
except Exception:
    status = "fail"
    err = traceback.format_exc(limit=2)

elapsed = time.time() - t0
print(json.dumps({
    "stage": "smoke",
    "status": status,
    "input_size": cfg,
    "elapsed_sec": round(elapsed, 4),
    "key_outputs_present": key_outputs,
    "error": err
}, ensure_ascii=False))
PY
```

---

## 2) 方向正确性检查（Direction）

适用于“改造是否向预期方向移动”的快速判断。

```bash
python - <<'PY'
import json
import numpy as np

# TODO: 替换为你的真实指标
std_u_mean = 0.60
seq_u_mean = 0.65
seq_win_ratio = 0.62

uniqueness_lift = seq_u_mean - std_u_mean
direction_ok = (uniqueness_lift > 0) and (seq_win_ratio >= 0.5)

print(json.dumps({
    "stage": "direction",
    "std_u_mean": float(std_u_mean),
    "seq_u_mean": float(seq_u_mean),
    "seq_win_ratio": float(seq_win_ratio),
    "uniqueness_lift": float(uniqueness_lift),
    "expected_direction": "up",
    "observed_direction": "up" if uniqueness_lift > 0 else "down",
    "direction_ok": bool(direction_ok)
}, ensure_ascii=False))
PY
```

---

## 3) 强验证模板（Grid + Multi-seed + Assertions）

建议：

- 配置网格：`small / medium / large`
- seeds：至少 `3~5`
- 断言：`4~6` 条（合法性、非负、归一化、长度一致等）

```bash
python - <<'PY'
import json
import numpy as np
import pandas as pd

# TODO: 替换为你的模块
# import your_module as m

configs = [
    {"name": "small", "n_bars": 240, "sample_len": 24, "num_trials": 10},
    {"name": "medium", "n_bars": 480, "sample_len": 48, "num_trials": 15},
    {"name": "large", "n_bars": 960, "sample_len": 96, "num_trials": 20},
]
seeds = [7, 42, 77, 123, 202]

rows = []
failures = []

for cfg in configs:
    for seed in seeds:
        try:
            # TODO: 替换为真实输出
            # out = m.run_pipeline(cfg, seed)
            # quality = m.validate_quality(out, cfg, seed)

            # 占位示例（请替换）
            out = {
                "events_ok": True,
                "weights_non_negative": True,
                "weights_normalized": True,
                "sample_len_ok": True,
            }
            quality = {
                "seq_win_ratio": 0.55 + 0.1 * np.random.RandomState(seed).rand(),
                "uniqueness_lift": 0.01 + 0.05 * np.random.RandomState(seed + 1).rand(),
            }

            # ===== 关键断言（请映射到真实对象） =====
            assert out["events_ok"], "events 非法"
            assert out["weights_non_negative"], "存在负权重"
            assert out["weights_normalized"], "权重归一化异常"
            assert out["sample_len_ok"], "抽样长度不一致"

            rows.append({
                "config": cfg["name"],
                "seed": seed,
                "seq_win_ratio": float(quality["seq_win_ratio"]),
                "uniqueness_lift": float(quality["uniqueness_lift"]),
            })
        except Exception as e:
            failures.append({"config": cfg["name"], "seed": seed, "error": str(e)})

if rows:
    df = pd.DataFrame(rows)
    summary = df.groupby("config").agg(
        runs=("seed", "count"),
        avg_seq_win_ratio=("seq_win_ratio", "mean"),
        min_seq_win_ratio=("seq_win_ratio", "min"),
        avg_uniqueness_lift=("uniqueness_lift", "mean"),
        min_uniqueness_lift=("uniqueness_lift", "min"),
        lift_positive_ratio=("uniqueness_lift", lambda x: float((x > 0).mean())),
    ).reset_index()

    overall = {
        "total_runs": int(len(df)),
        "assertion_failures": int(len(failures)),
        "overall_seq_win_ratio_mean": float(df["seq_win_ratio"].mean()),
        "overall_uniqueness_lift_mean": float(df["uniqueness_lift"].mean()),
        "overall_lift_positive_ratio": float((df["uniqueness_lift"] > 0).mean()),
    }
else:
    summary = pd.DataFrame()
    overall = {
        "total_runs": 0,
        "assertion_failures": int(len(failures)),
        "overall_seq_win_ratio_mean": None,
        "overall_uniqueness_lift_mean": None,
        "overall_lift_positive_ratio": None,
    }

print("=== OVERALL ===")
print(json.dumps(overall, ensure_ascii=False))
print("=== SUMMARY_BY_CONFIG ===")
if not summary.empty:
    print(summary.to_string(index=False))
else:
    print("EMPTY")
print("=== FAILURES ===")
print(json.dumps(failures, ensure_ascii=False))
PY
```

---

## 4) 结果解读模板（发给用户）

```markdown
验证结论：
- 跑通状态：pass/fail
- 方向判断：direction_ok = true/false
- 强验证：通过率 X/Y，断言失败 N 条
- 核心指标：seq_win_ratio 均值、uniqueness_lift 均值/最小值/正值占比
- 结论分级：可用 / 有风险 / 不通过
- 残余风险：...
- 下一步建议：...
```

---

## 5) 常见问题

- **Q: 跑太慢？**  
  先减小 `n_bars / sample_len / num_trials`，优先拿到方向证据，再逐步放大。

- **Q: 单次结果很好，能说“完全正确”吗？**  
  不能。必须看多配置、多 seed 的分布与断言稳定性。

- **Q: 一定要改主逻辑才能验证吗？**  
  不要。先用 here-doc 外挂验证，不动主逻辑，验证通过后再决定是否重构。
