"""
AFML Chapter 4 抽样与加权工具（项目可运行版）。

和 Triple Barrier 配合的核心流程：
1) 计算并发事件数
2) 计算样本唯一性
3) 顺序自助采样（sequential bootstrap）
4) 计算收益/时间衰减权重
"""

from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _as_series(data: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    """标准化为单列 Series，避免函数内原地改动输入。"""
    if isinstance(data, pd.Series):
        out = pd.Series(data, copy=True)
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            out = pd.Series(data.iloc[:, 0], copy=True)
        elif "close" in data.columns:
            out = pd.Series(data["close"], copy=True)
        else:
            raise ValueError(f"{name} 必须是 Series、单列 DataFrame 或包含 close 列")
    else:
        raise ValueError(f"{name} 必须是 pandas Series/DataFrame")

    if out.isna().any():
        raise ValueError(f"{name} 包含 NaN，请先清洗")
    return out


def _validate_events(events: pd.DataFrame) -> pd.DataFrame:
    """校验事件输入：index=t0，且必须包含 t1 列。"""
    if not isinstance(events, pd.DataFrame):
        raise ValueError("events 必须是 DataFrame")
    if "t1" not in events.columns:
        raise ValueError("events 必须包含 t1 列")
    if events.empty:
        raise ValueError("events 不能为空")
    if events.index.hasnans or events["t1"].isna().any():
        raise ValueError("events.index / events['t1'] 不能包含 NaN/NaT")
    return events.sort_index().copy()


def _split_chunks(values: Sequence, num_chunks: int) -> List[Sequence]:
    n = len(values)
    if n == 0:
        return []
    num_chunks = max(1, min(int(num_chunks), n))
    step = int(np.ceil(n / num_chunks))
    return [values[i : i + step] for i in range(0, n, step)]


def process_jobs_(jobs: List[Dict]) -> List:
    """串行执行 jobs。"""
    out = []
    for job in jobs:
        fn = job["func"]
        kwargs = {k: v for k, v in job.items() if k != "func"}
        out.append(fn(**kwargs))
    return out


def process_jobs(jobs: List[Dict], num_threads: int = 1) -> List:
    """线程并行执行 jobs；小数据下开销可能大于收益。"""
    if num_threads <= 1:
        return process_jobs_(jobs)
    out = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for job in jobs:
            fn = job["func"]
            kwargs = {k: v for k, v in job.items() if k != "func"}
            futures.append(executor.submit(fn, **kwargs))
        for future in futures:
            out.append(future.result())
    return out


def mp_pandas_obj(func, pd_obj: Tuple[str, Sequence], num_threads: int = 1, **kwargs):
    """
    轻量版 mp_pandas_obj。
    约定 pd_obj = ("molecule", iterable)。
    """
    arg_name, values = pd_obj
    chunks = _split_chunks(list(values), max(1, int(num_threads)))
    jobs = []
    for chunk in chunks:
        job = {"func": func, arg_name: chunk}
        job.update(kwargs)
        jobs.append(job)
    out = process_jobs(jobs, num_threads=max(1, int(num_threads)))
    if not out:
        return pd.Series(dtype=float)
    if isinstance(out[0], (pd.Series, pd.DataFrame)):
        return pd.concat(out, axis=0)
    return out


# =======================================================
# 4.1 并发事件数
def _count_concurrent_events_in_chunk(data_index: pd.Index, t1: pd.Series, molecule: Sequence) -> pd.Series:
    if len(molecule) == 0:
        return pd.Series(dtype=float)

    t1_ = pd.Series(t1, copy=True).fillna(data_index[-1])
    start = molecule[0]
    end = t1_.loc[molecule].max()
    t1_ = t1_[t1_ >= start].loc[:end]
    if t1_.empty:
        return pd.Series(0.0, index=data_index)

    iloc = data_index.searchsorted(pd.Index([t1_.index[0], t1_.max()]))
    count = pd.Series(0.0, index=data_index[iloc[0] : iloc[1] + 1])
    for t_in, t_out in t1_.items():
        count.loc[t_in:t_out] += 1.0
    return count.loc[start:end]


def count_concurrent_events(data: pd.Series | pd.DataFrame, events: pd.DataFrame, num_threads: int = 1) -> pd.Series:
    """计算每个 bar 的并发事件数量，并对齐到 data.index。"""
    data_series = _as_series(data, "data")
    events_ = _validate_events(events)

    out = mp_pandas_obj(
        func=_count_concurrent_events_in_chunk,
        pd_obj=("molecule", events_.index),
        num_threads=num_threads,
        data_index=data_series.index,
        t1=events_["t1"],
    )
    out = out.loc[~out.index.duplicated(keep="last")]
    return out.reindex(data_series.index).fillna(0.0)


# =======================================================
# 4.2 平均唯一性权重
def _compute_uniqueness_weight_in_chunk(t1: pd.Series, num_conc_events: pd.Series, molecule: Sequence) -> pd.Series:
    if len(molecule) == 0:
        return pd.Series(dtype=float)
    wght = pd.Series(index=pd.Index(molecule), dtype=float)
    for t_in, t_out in t1.loc[wght.index].items():
        denom = num_conc_events.loc[t_in:t_out].replace(0, np.nan)
        wght.loc[t_in] = (1.0 / denom).mean()
    return wght.fillna(0.0)


def weights_by_concurrent_events(
    data: pd.Series | pd.DataFrame,
    events: pd.DataFrame,
    num_threads: int = 1,
) -> pd.DataFrame:
    """按并发度计算事件权重，返回列名 `tW`。"""
    data_series = _as_series(data, "data")
    events_ = _validate_events(events)
    num_co_event = count_concurrent_events(data_series, events_, num_threads=num_threads)

    tw = mp_pandas_obj(
        func=_compute_uniqueness_weight_in_chunk,
        pd_obj=("molecule", events_.index),
        num_threads=num_threads,
        t1=events_["t1"],
        num_conc_events=num_co_event,
    )
    return pd.DataFrame({"tW": tw.reindex(events_.index).fillna(0.0)})


# =======================================================
# 4.3 指示矩阵
def build_indicator_matrix(data: pd.Series | pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """构造事件覆盖矩阵（行=bar，列=事件）。"""
    data_series = _as_series(data, "data").dropna()
    events_ = _validate_events(events)

    data_idx = data_series.loc[events_.index.min() : events_["t1"].max()].index
    ind_m = pd.DataFrame(0.0, index=data_idx, columns=np.arange(events_.shape[0]))
    for i, (t0, t1_) in enumerate(events_["t1"].items()):
        ind_m.loc[t0:t1_, i] = 1.0
    return ind_m[ind_m.sum(axis=1) > 0]


def build_indicator_matrix_parallel(
    data: pd.Series | pd.DataFrame,
    events: pd.DataFrame,
    num_threads: int = 1,
) -> pd.DataFrame:
    """并行接口兼容版本；当前实现优先保证结果稳定。"""
    if num_threads <= 1:
        return build_indicator_matrix(data, events)
    # 对于该类型计算，分块收益有限，这里直接复用稳健实现
    return build_indicator_matrix(data, events)


# =======================================================
# 4.4 平均唯一性
def average_uniqueness(idx_m: pd.DataFrame) -> pd.Series:
    """返回每个事件（每列）的平均唯一性。"""
    if not isinstance(idx_m, pd.DataFrame) or idx_m.empty:
        raise ValueError("idx_m 必须是非空 DataFrame")
    idx_sum = idx_m.sum(axis=1).replace(0, np.nan)
    uniqueness = idx_m.div(idx_sum, axis=0)
    avg_u = uniqueness.where(uniqueness > 0).mean()
    return avg_u.fillna(0.0)


# =======================================================
# 4.5 Sequential Bootstrap
def sequential_bootstrap(
    idx_m: pd.DataFrame,
    sample_len: Optional[int] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> List[int]:
    """顺序自助采样（串行基线实现）。"""
    if not isinstance(idx_m, pd.DataFrame) or idx_m.empty:
        raise ValueError("idx_m 必须是非空 DataFrame")
    if idx_m.isna().any().any():
        raise ValueError("idx_m 不能含 NaN")

    if random_state is None:
        random_state = np.random.RandomState(42)
    if sample_len is None:
        sample_len = idx_m.shape[1]

    phi: List[int] = []
    while len(phi) < sample_len:
        avg_u = pd.Series(0.0, index=idx_m.columns, dtype=float)
        for col in idx_m.columns:
            avg_u.loc[col] = average_uniqueness(idx_m[phi + [col]]).iloc[-1]
        prob = avg_u / avg_u.sum()
        phi.append(int(random_state.choice(idx_m.columns, p=prob.values)))
    return phi


def _bootstrap_candidate_score(idx_m: pd.DataFrame, phi: List[int], col: int) -> float:
    return float(average_uniqueness(idx_m[phi + [col]]).iloc[-1])


def sequential_bootstrap_parallel(
    idx_m: pd.DataFrame,
    sample_len: Optional[int] = None,
    num_threads: int = 1,
    verbose: bool = False,
    random_state: Optional[np.random.RandomState] = None,
) -> List[int]:
    """并行接口兼容版 sequential bootstrap。"""
    if random_state is None:
        random_state = np.random.RandomState(42)
    if sample_len is None:
        sample_len = idx_m.shape[1]

    phi: List[int] = []
    while len(phi) < sample_len:
        jobs = []
        for col in idx_m.columns:
            jobs.append({"func": _bootstrap_candidate_score, "idx_m": idx_m, "phi": phi, "col": col})
        scores = process_jobs(jobs, num_threads=max(1, num_threads))
        avg_u = pd.Series(scores, index=idx_m.columns, dtype=float)
        prob = avg_u / avg_u.sum()
        if verbose:
            print(f"bootstrap step={len(phi)+1}, prob_range=({prob.min():.6f}, {prob.max():.6f})")
        phi.append(int(random_state.choice(idx_m.columns, p=prob.values)))
    return phi


# =======================================================
# 4.9 Monte Carlo 对比
def _run_bootstrap_comparison_once(idx_m: pd.DataFrame, random_state: np.random.RandomState) -> Dict[str, float]:
    random_cols = random_state.choice(idx_m.columns, size=idx_m.shape[1], replace=True)
    std_u = float(average_uniqueness(idx_m[random_cols]).mean())
    phi = sequential_bootstrap_parallel(idx_m=idx_m, sample_len=None, num_threads=1, random_state=random_state)
    seq_u = float(average_uniqueness(idx_m[phi]).mean())
    return {"stdU": std_u, "seqU": seq_u}


def monte_carlo_sequential_bootstrap(
    data: pd.Series | pd.DataFrame,
    events: pd.DataFrame,
    sample_len: Optional[int] = None,
    num_iterations: int = 100,
    num_threads: int = 1,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Monte Carlo 比较标准抽样 vs sequential bootstrap。"""
    idx_m = build_indicator_matrix_parallel(data=data, events=events, num_threads=max(1, num_threads))
    if sample_len is not None and sample_len < idx_m.shape[1]:
        idx_m = idx_m.iloc[:, :sample_len]

    jobs = []
    for i in range(int(num_iterations)):
        jobs.append(
            {
                "func": _run_bootstrap_comparison_once,
                "idx_m": idx_m,
                "random_state": np.random.RandomState(random_seed + i),
            }
        )
    out = process_jobs(jobs, num_threads=max(1, num_threads))
    return pd.DataFrame(out)


# =======================================================
# 4.10 收益权重
def _compute_return_weight_in_chunk(
    log_ret: pd.Series,
    events_t1: pd.Series,
    num_co_event: pd.Series,
    molecule: Sequence,
) -> pd.Series:
    if len(molecule) == 0:
        return pd.Series(dtype=float)
    w = pd.Series(index=pd.Index(molecule), dtype=float)
    for t_in, t_out in events_t1.loc[w.index].items():
        path_ret = log_ret.loc[t_in:t_out]
        path_conc = num_co_event.loc[t_in:t_out].replace(0, np.nan)
        w.loc[t_in] = path_ret.div(path_conc).fillna(0.0).sum()
    return w.abs().fillna(0.0)


def weights_by_return(data: pd.Series | pd.DataFrame, events: pd.DataFrame, num_threads: int = 1) -> pd.DataFrame:
    """按绝对收益和并发度计算权重，并归一化。"""
    data_series = _as_series(data, "data")
    events_ = _validate_events(events)
    num_co_event = count_concurrent_events(data_series, events_, num_threads=num_threads)
    log_ret = np.log(data_series).diff().fillna(0.0)

    w = mp_pandas_obj(
        func=_compute_return_weight_in_chunk,
        pd_obj=("molecule", events_.index),
        num_threads=num_threads,
        log_ret=log_ret,
        events_t1=events_["t1"],
        num_co_event=num_co_event,
    )
    out = pd.DataFrame({"w": w.reindex(events_.index).fillna(0.0)})
    denom = out["w"].sum()
    if denom > 0:
        out["w"] *= out.shape[0] / denom
    return out


# =======================================================
# 4.11 时间衰减权重
def weights_by_time_decay(
    data: pd.Series | pd.DataFrame,
    events: pd.DataFrame,
    num_threads: int = 1,
    td: float = 1.0,
) -> pd.Series:
    """
    时间衰减权重：
    - td=1: 无衰减
    - 0<td<1: 线性衰减但全正
    - td=0: 衰减到 0
    - -1<td<0: 最老一段样本被抹零
    """
    if not isinstance(td, (float, int)):
        raise ValueError("td 必须是数值")
    if td < -1 or td > 1:
        raise ValueError("td 必须在 [-1, 1] 区间内")

    tw = weights_by_concurrent_events(data=data, events=events, num_threads=num_threads)["tW"].sort_index()
    csum = tw.cumsum()
    if csum.empty:
        return csum

    if td >= 0:
        slope = (1.0 - float(td)) / csum.iloc[-1]
    else:
        slope = 1.0 / ((float(td) + 1.0) * csum.iloc[-1])
    const = 1.0 - slope * csum.iloc[-1]
    w = const + slope * csum
    w[w < 0] = 0.0
    return w


# =======================================================
# Triple Barrier 对接工具
def events_from_triple_barrier(
    barrier_events: pd.DataFrame,
    exit_col: str = "exit",
    keep_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    将 triple_barrier 输出转换为 sample_unique 需要的事件格式。

    约定：
    - 输入 index 为事件起点 t0
    - 输入包含退出时间列（默认 `exit`）
    - 输出至少包含 `t1` 列（事件终点）
    """
    if not isinstance(barrier_events, pd.DataFrame):
        raise ValueError("barrier_events 必须是 DataFrame")
    if exit_col not in barrier_events.columns:
        raise ValueError(f"barrier_events 缺少列: {exit_col}")
    if barrier_events.empty:
        raise ValueError("barrier_events 不能为空")
    if barrier_events.index.hasnans:
        raise ValueError("barrier_events.index 不能包含 NaN/NaT")

    out = pd.DataFrame(index=barrier_events.index.copy())
    out["t1"] = pd.to_datetime(barrier_events[exit_col], errors="coerce")
    out = out.dropna(subset=["t1"])
    if out.empty:
        raise ValueError("barrier_events 中没有有效的退出时间")

    # 保证事件区间合法：t1 不早于 t0
    out = out[out["t1"] >= out.index]
    if out.empty:
        raise ValueError("不存在满足 t1 >= t0 的有效事件")

    if keep_cols is not None:
        for col in keep_cols:
            if col in barrier_events.columns:
                out[col] = barrier_events.loc[out.index, col]

    # 同一 t0 若重复，保留最早 t1，减少重叠歧义
    out = out.sort_index()
    out = out.groupby(level=0, sort=True).agg({"t1": "min", **{c: "first" for c in out.columns if c != "t1"}})
    return out


def build_composite_event_weights(
    data: pd.Series | pd.DataFrame,
    events: pd.DataFrame,
    num_threads: int = 1,
    td: float = 1.0,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    基于并发唯一性、收益贡献、时间衰减构建组合权重。
    返回列：
    - tW: 并发唯一性权重
    - w_ret: 收益贡献权重
    - w_td: 时间衰减权重
    - w: 组合权重（可选归一化）
    """
    events_ = _validate_events(events)
    t_weight = weights_by_concurrent_events(data=data, events=events_, num_threads=num_threads)["tW"]
    r_weight = weights_by_return(data=data, events=events_, num_threads=num_threads)["w"]
    d_weight = weights_by_time_decay(data=data, events=events_, num_threads=num_threads, td=td)

    out = pd.DataFrame(index=events_.index)
    out["tW"] = t_weight.reindex(out.index).fillna(0.0)
    out["w_ret"] = r_weight.reindex(out.index).fillna(0.0)
    out["w_td"] = d_weight.reindex(out.index).fillna(0.0)
    out["w"] = out["tW"] * out["w_ret"] * out["w_td"]

    if normalize:
        denom = out["w"].sum()
        if denom > 0:
            out["w"] *= out.shape[0] / denom
    return out


def triple_barrier_sampling_pipeline(
    close: pd.Series | pd.DataFrame,
    enter: Sequence,
    pt_sl: Sequence[float],
    max_holding: Sequence[int],
    target: Optional[pd.Series] = None,
    side: Optional[pd.Series] = None,
    num_threads: int = 1,
    td: float = 1.0,
    sample_len: Optional[int] = None,
    random_seed: int = 42,
    use_fast: bool = True,
) -> Dict[str, Any]:
    """
    一键串联 Triple Barrier + Sample Unique。

    返回：
    - barrier_events: triple barrier 原始输出（含 exit/ret/side）
    - events: sample_unique 事件格式（含 t1）
    - weights: 组合权重表
    - idx_m: 指示矩阵
    - phi: sequential bootstrap 选中的列号（可重复）
    - sampled_events: 按 phi 抽样后的事件（可重复）
    - sampled_weights: sampled_events 对应权重
    """
    data_series = _as_series(close, "close")
    data_series = data_series.sort_index()
    data_series = data_series[~data_series.index.duplicated(keep="last")]

    # 仅在函数内导入，避免模块级循环依赖。
    import triple_barrier as tb

    if use_fast:
        barrier_events = tb.get_barrier_fast(
            close=data_series,
            enter=enter,
            pt_sl=pt_sl,
            max_holding=max_holding,
            target=target,
            side=side,
        )
    else:
        barrier_events = tb.get_barrier(
            close=data_series,
            enter=enter,
            pt_sl=pt_sl,
            max_holding=max_holding,
            target=target,
            side=side,
        )

    events = events_from_triple_barrier(
        barrier_events=barrier_events,
        exit_col="exit",
        keep_cols=["ret", "side"],
    )
    weights = build_composite_event_weights(
        data=data_series,
        events=events[["t1"]],
        num_threads=num_threads,
        td=td,
        normalize=True,
    )
    idx_m = build_indicator_matrix_parallel(data=data_series, events=events[["t1"]], num_threads=max(1, num_threads))

    if idx_m.empty:
        return {
            "barrier_events": barrier_events,
            "events": events,
            "weights": weights,
            "idx_m": idx_m,
            "phi": [],
            "sampled_events": events.iloc[0:0].copy(),
            "sampled_weights": weights.iloc[0:0].copy(),
        }

    if sample_len is None:
        sample_len_ = idx_m.shape[1]
    else:
        sample_len_ = int(sample_len)
        sample_len_ = min(max(sample_len_, 1), idx_m.shape[1])

    rs = np.random.RandomState(random_seed)
    phi = sequential_bootstrap_parallel(
        idx_m=idx_m,
        sample_len=sample_len_,
        num_threads=max(1, num_threads),
        random_state=rs,
    )

    sampled_events = events.iloc[np.asarray(phi, dtype=int)].copy()
    sampled_events["bootstrap_col"] = phi
    sampled_weights = weights.reindex(sampled_events.index).copy()
    sampled_weights["bootstrap_col"] = phi

    return {
        "barrier_events": barrier_events,
        "events": events,
        "weights": weights,
        "idx_m": idx_m,
        "phi": phi,
        "sampled_events": sampled_events,
        "sampled_weights": sampled_weights,
    }


# =======================================================
# 快速随机测试工具（可选）
def random_event_end_times(num_obs: int, num_bars: int, max_h: int, random_seed: int = 42) -> pd.Series:
    """生成整数 bar 版 t0->t1 映射，用于快测。"""
    if num_obs <= 0 or num_bars <= 1 or max_h <= 1:
        raise ValueError("num_obs/num_bars/max_h 必须为正，且 num_bars,max_h > 1")
    rs = np.random.RandomState(random_seed)
    t1 = pd.Series(dtype=int)
    for _ in range(num_obs):
        ix = rs.randint(0, num_bars - 1)
        t1.loc[ix] = min(ix + rs.randint(1, max_h), num_bars - 1)
    return t1.sort_index()


def run_bootstrap_comparison_example(
    num_obs: int,
    num_bars: int,
    max_h: int,
    random_seed: int = 42,
) -> Dict[str, float]:
    """单次对比实验：标准随机抽样 vs sequential bootstrap。"""
    t1 = random_event_end_times(num_obs=num_obs, num_bars=num_bars, max_h=max_h, random_seed=random_seed)
    events = pd.DataFrame({"t1": t1.values}, index=t1.index)
    data = pd.Series(np.arange(num_bars), index=np.arange(num_bars), dtype=float)
    idx_m = build_indicator_matrix(data=data, events=events)
    rs = np.random.RandomState(random_seed)
    random_cols = rs.choice(idx_m.columns, size=idx_m.shape[1], replace=True)
    std_u = float(average_uniqueness(idx_m[random_cols]).mean())
    seq_cols = sequential_bootstrap(idx_m=idx_m, random_state=rs)
    seq_u = float(average_uniqueness(idx_m[seq_cols]).mean())
    return {"stdU": std_u, "seqU": seq_u}


def run_monte_carlo_bootstrap_experiments(
    num_obs: int = 10,
    num_bars: int = 100,
    max_h: int = 5,
    num_iters: int = 1000,
    num_threads: int = 1,
    random_seed: int = 42,
) -> pd.DataFrame:
    """多次 Monte Carlo 实验（教学/验证用途）。"""
    warnings.warn("MT_MC 仅用于验证，不建议生产中直接使用。", UserWarning)
    jobs = []
    for i in range(int(num_iters)):
        jobs.append(
            {
                "func": run_bootstrap_comparison_example,
                "num_obs": num_obs,
                "num_bars": num_bars,
                "max_h": max_h,
                "random_seed": random_seed + i,
            }
        )
    out = process_jobs(jobs, num_threads=max(1, num_threads))
    return pd.DataFrame(out)


def validate_sampling_quality(
    idx_m: pd.DataFrame,
    weights: Optional[pd.DataFrame] = None,
    num_trials: int = 200,
    random_seed: int = 42,
) -> Dict[str, float]:
    """
    对 sample_unique 抽样质量做最小闭环验证。

    输出指标：
    - std_u_mean: 随机抽样平均唯一性均值
    - seq_u_mean: 顺序抽样平均唯一性均值
    - seq_win_ratio: seqU > stdU 的实验占比
    - uniqueness_lift: seq_u_mean - std_u_mean
    - weight_top_1pct_mass / weight_top_5pct_mass: 组合权重头部集中度（可选）
    """
    if not isinstance(idx_m, pd.DataFrame) or idx_m.empty:
        raise ValueError("idx_m 必须是非空 DataFrame")
    if num_trials <= 0:
        raise ValueError("num_trials 必须是正整数")

    rs = np.random.RandomState(random_seed)
    std_values: List[float] = []
    seq_values: List[float] = []

    for i in range(int(num_trials)):
        rs_i = np.random.RandomState(rs.randint(0, 2**31 - 1) + i)
        random_cols = rs_i.choice(idx_m.columns, size=idx_m.shape[1], replace=True)
        std_u = float(average_uniqueness(idx_m[random_cols]).mean())

        seq_cols = sequential_bootstrap_parallel(
            idx_m=idx_m,
            sample_len=idx_m.shape[1],
            num_threads=1,
            random_state=rs_i,
        )
        seq_u = float(average_uniqueness(idx_m[seq_cols]).mean())

        std_values.append(std_u)
        seq_values.append(seq_u)

    std_arr = np.asarray(std_values, dtype=float)
    seq_arr = np.asarray(seq_values, dtype=float)
    metrics: Dict[str, float] = {
        "std_u_mean": float(np.nanmean(std_arr)),
        "seq_u_mean": float(np.nanmean(seq_arr)),
        "seq_win_ratio": float(np.mean(seq_arr > std_arr)),
    }
    metrics["uniqueness_lift"] = metrics["seq_u_mean"] - metrics["std_u_mean"]

    if isinstance(weights, pd.DataFrame) and "w" in weights.columns and not weights.empty:
        w = pd.to_numeric(weights["w"], errors="coerce").fillna(0.0)
        w = w[w > 0]
        if not w.empty:
            w_sorted = w.sort_values(ascending=False)
            n = len(w_sorted)
            top1 = max(1, int(np.ceil(n * 0.01)))
            top5 = max(1, int(np.ceil(n * 0.05)))
            total = float(w_sorted.sum())
            metrics["weight_top_1pct_mass"] = float(w_sorted.iloc[:top1].sum() / total) if total > 0 else np.nan
            metrics["weight_top_5pct_mass"] = float(w_sorted.iloc[:top5].sum() / total) if total > 0 else np.nan
        else:
            metrics["weight_top_1pct_mass"] = np.nan
            metrics["weight_top_5pct_mass"] = np.nan
    return metrics


def main() -> None:
    """
    最小可运行示例：
    1) 合成价格序列
    2) Triple Barrier 生成事件
    3) Sample Unique 抽样/加权
    4) 输出抽样质量指标
    """
    parser = argparse.ArgumentParser(description="Run triple-barrier + sample-unique quick validation.")
    parser.add_argument("--n-bars", type=int, default=360, help="Number of synthetic bars.")
    parser.add_argument("--sample-len", type=int, default=40, help="Bootstrap sample length.")
    parser.add_argument("--num-trials", type=int, default=30, help="Trials for sampling quality validation.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    rs = np.random.RandomState(args.random_seed)
    n = int(args.n_bars)
    if n < 120:
        raise ValueError("n-bars 至少为 120，避免事件过少导致评估不稳定")
    index = pd.date_range("2025-01-01", periods=n, freq="15min")

    # 构造可复现的 fat-tail 收益，再还原为价格序列
    log_ret = rs.standard_t(df=5, size=n) * 0.002
    close = pd.Series(100.0 * np.exp(np.cumsum(log_ret)), index=index, name="close")

    # 用简单分位规则构造 enter，避免依赖额外复杂预处理
    trigger = pd.Series(log_ret, index=index).abs()
    enter = trigger[trigger > trigger.quantile(0.90)].index

    out = triple_barrier_sampling_pipeline(
        close=close,
        enter=enter,
        pt_sl=[1.2, 1.0],
        max_holding=[0, 16],  # 约 4 小时
        target=trigger.ewm(span=96).std().bfill(),
        side=None,
        num_threads=1,
        td=0.8,
        sample_len=args.sample_len,
        random_seed=args.random_seed,
        use_fast=True,
    )

    quality = validate_sampling_quality(
        idx_m=out["idx_m"],
        weights=out["weights"],
        num_trials=args.num_trials,
        random_seed=args.random_seed,
    )

    print("=== sample_unique pipeline summary ===")
    print(f"events_raw={len(out['barrier_events'])}, events_valid={len(out['events'])}, sampled={len(out['sampled_events'])}")
    print("=== sampling quality ===")
    for key, value in quality.items():
        print(f"{key}: {value:.6f}")


# =======================================================
# 向后兼容别名（旧名 -> 新名）
num_co_events = count_concurrent_events
wght_by_coevents = weights_by_concurrent_events
idx_matrix = build_indicator_matrix
mp_idx_matrix = build_indicator_matrix_parallel
av_unique = average_uniqueness
seq_bts = sequential_bootstrap
mp_seq_bts = sequential_bootstrap_parallel
MC_seq_bts = monte_carlo_sequential_bootstrap
wght_by_rtn = weights_by_return
wght_by_td = weights_by_time_decay
rnd_t1 = random_event_end_times
auxMC = run_bootstrap_comparison_example
MT_MC = run_monte_carlo_bootstrap_experiments


if __name__ == "__main__":
    main()
