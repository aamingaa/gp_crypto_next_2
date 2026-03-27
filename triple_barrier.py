from datetime import datetime
from itertools import product
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cusum_filter(series: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """
    对应 Lopez de Prado《AFML》里的对称 CUSUM 过滤器（Snippet 2.4/2.5 风格）。

    Parameters
    ----------
    series : pd.Series
        一般传入价格的对数或收益序列，index 为时间。
    threshold : float
        CUSUM 阈值（绝对值越大，触发的事件越少）。

    Returns
    -------
    pd.DatetimeIndex
        触发事件的时间索引（用于后续 Triple Barrier 的 enter）。
    """
    series = series.astype(float)
    diff = series.diff().dropna()

    t_events = []
    s_pos, s_neg = 0.0, 0.0

    for t in diff.index:
        d = diff.loc[t]
        s_pos = max(0.0, s_pos + d)
        s_neg = min(0.0, s_neg + d)

        if s_pos > threshold:
            s_pos = 0.0
            t_events.append(t)
        elif s_neg < -threshold:
            s_neg = 0.0
            t_events.append(t)

    return pd.DatetimeIndex(t_events)


def prepare_close_for_cusum(close: pd.Series, use_log: bool = True):
    """
    整理 close，并给出传入 cusum_filter 的序列（默认 log(close)）。
    返回 (close, base)。
    """
    close = pd.Series(close, copy=True).astype(float)
    close = close.sort_index()
    close = close[~close.index.duplicated(keep="last")]
    base = np.log(close) if use_log else close
    return close, base


def cusum_filter_dynamic(series: pd.Series, threshold: pd.Series) -> pd.DatetimeIndex:
    """
    动态阈值版本的 CUSUM 包装函数，不改动原始 cusum_filter 行为。

    Parameters
    ----------
    series : pd.Series
        一般传入价格的对数或收益序列，index 为时间。
    threshold : pd.Series
        与 series 时间索引对齐的动态阈值序列。

    Returns
    -------
    pd.DatetimeIndex
        触发事件的时间索引。
    """
    series = pd.Series(series, copy=False).astype(float)
    diff = series.diff().dropna()
    threshold_series = pd.Series(threshold, copy=False).astype(float).reindex(diff.index)

    t_events = []
    s_pos, s_neg = 0.0, 0.0

    for t in diff.index:
        cur_threshold = threshold_series.loc[t]
        if not np.isfinite(cur_threshold) or cur_threshold <= 0:
            continue

        d = diff.loc[t]
        s_pos = max(0.0, s_pos + d)
        s_neg = min(0.0, s_neg + d)

        if s_pos > cur_threshold:
            s_pos = 0.0
            t_events.append(t)
        elif s_neg < -cur_threshold:
            s_neg = 0.0
            t_events.append(t)

    return pd.DatetimeIndex(t_events)


def cusum_filter_side(series: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """
    对应 Lopez de Prado《AFML》里的对称 CUSUM 过滤器（Snippet 2.4/2.5 风格）。

    Parameters
    ----------
    series : pd.Series
        一般传入价格的对数或收益序列，index 为时间。
    threshold : float
        CUSUM 阈值（绝对值越大，触发的事件越少）。

    Returns
    -------
    pd.DatetimeIndex
        触发事件的时间索引（用于后续 Triple Barrier 的 enter）。
    """
    series = series.astype(float)
    diff = series.diff().dropna()

    t_events = []
    s_pos, s_neg = 0.0, 0.0
    sides = []

    for t in diff.index:
        d = diff.loc[t]
        s_pos = max(0.0, s_pos + d)
        s_neg = min(0.0, s_neg + d)

        if s_pos > threshold:
            s_pos = 0.0
            t_events.append(t)
            sides.append(1.0)
        elif s_neg < -threshold:
            s_neg = 0.0
            t_events.append(t)
            sides.append(-1.0)

    return pd.Series(sides, index=pd.DatetimeIndex(t_events))


def add_vertical_barrier(t_events, close, days=0, hours=0, minutes=0, seconds=0):
    """
    Advances in Financial Machine Learning, Snippet 3.4 page 49.
    Adding a Vertical Barrier
    For each index in t_events, it finds the timestamp of the next price bar at or immediately after
    a number of days num_days. This vertical barrier can be passed as an optional argument t1 in get_events.
    This function creates a series that has all the timestamps of when the vertical barrier would be reached.
    :param t_events: (pd.Series) Series of events (symmetric CUSUM filter)
    :param close: (pd.Series) Close prices
    :param num_days: (int) Number of days to add for vertical barrier
    :param num_hours: (int) Number of hours to add for vertical barrier
    :param num_minutes: (int) Number of minutes to add for vertical barrier
    :param num_seconds: (int) Number of seconds to add for vertical barrier
    :return: (pd.Series) Timestamps of vertical barriers
    """
    timedelta = pd.Timedelta(
        '{} days, {} hours, {} minutes, {} seconds'.format(days, hours, minutes, seconds))
    # Find index to closest to vertical barrier
    nearest_index = close.index.searchsorted(t_events + timedelta)

    # Exclude indexes which are outside the range of close price index
    nearest_index = nearest_index[nearest_index < close.shape[0]]

    # Find price index closest to vertical barrier time stamp
    nearest_timestamp = close.index[nearest_index]
    filtered_events = t_events[:nearest_index.shape[0]]

    vertical_barriers = pd.Series(data=nearest_timestamp, index=filtered_events)
    return vertical_barriers


def forming_barriers(close, events, pt_sl, molecule): 
    """
    Advances in Financial Machine Learning, Snippet 3.2, page 45.
    Triple Barrier Labeling Method
    This function applies the triple-barrier labeling method. It works on a set of
    datetime index values (molecule). This allows the program to parallelize the processing.
    Mainly it returns a DataFrame of timestamps regarding the time when the first barriers were reached.
    :param close: (pd.Series) Close prices
    :param events: (pd.Series) Indices that signify "events" (see cusum_filter function
    for more details)
    :param pt_sl: (np.array) Element 0, indicates the profit taking level; Element 1 is stop loss level
    :param molecule: (an array) A set of datetime index values for processing
    :return: (pd.DataFrame) Timestamps of when first barrier was touched
    """
    # Apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['exit']].copy(deep=True)

    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]

    # Profit taking active
    if profit_taking_multiple > 0:
        profit_taking = profit_taking_multiple * events_['trgt']
    else:
        profit_taking = pd.Series(index=events.index)  # NaNs

    # Stop loss active
    if stop_loss_multiple > 0:
        stop_loss = -stop_loss_multiple * events_['trgt']
    else:
        stop_loss = pd.Series(index=events.index)  # NaNs

    out['pt'] = pd.Series(dtype=events.index.dtype)
    out['sl'] = pd.Series(dtype=events.index.dtype)

    # Get events
    for loc, vertical_barrier in events_['exit'].fillna(close.index[-1]).items():
        closing_prices = close[loc: vertical_barrier]  # Path prices for a given trade
        cum_returns = (closing_prices / close[loc] - 1) * events_.at[loc, 'side']  # Path returns
        out.at[loc, 'sl'] = cum_returns[cum_returns < stop_loss[loc]].index.min()  # Earliest stop loss date
        out.at[loc, 'pt'] = cum_returns[cum_returns > profit_taking[loc]].index.min()  # Earliest profit taking date

    return out


def forming_barriers_fast(close, events, pt_sl, molecule): 
    """
    Optimized version of forming_barriers using numpy and integer indexing.
    """
    events_ = events.loc[molecule]
    out = events_[['exit']].copy()

    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]

    if profit_taking_multiple > 0:
        profit_taking = profit_taking_multiple * events_['trgt']
    else:
        profit_taking = pd.Series(index=events_.index, data=np.inf)

    if stop_loss_multiple > 0:
        stop_loss = -stop_loss_multiple * events_['trgt']
    else:
        stop_loss = pd.Series(index=events_.index, data=-np.inf)

    close_vals = close.values
    close_index = close.index
    
    start_indices = close_index.get_indexer(events_.index)
    end_indices = close_index.get_indexer(events_['exit'].fillna(close_index[-1]))
    
    sides = events_['side'].values
    pt_vals = profit_taking.values
    sl_vals = stop_loss.values
    
    sl_dates = [pd.NaT] * len(events_)
    pt_dates = [pd.NaT] * len(events_)

    for i in range(len(events_)):
        s_idx = start_indices[i]
        e_idx = end_indices[i]
        if s_idx < 0 or e_idx < 0:
            continue
            
        path_prices = close_vals[s_idx : e_idx + 1]
        path_returns = (path_prices / close_vals[s_idx] - 1.0) * sides[i]
        
        sl_hits = np.where(path_returns < sl_vals[i])[0]
        if len(sl_hits) > 0:
            sl_dates[i] = close_index[s_idx + sl_hits[0]]
            
        pt_hits = np.where(path_returns > pt_vals[i])[0]
        if len(pt_hits) > 0:
            pt_dates[i] = close_index[s_idx + pt_hits[0]]

    out['sl'] = sl_dates
    out['pt'] = pt_dates
    return out


def get_barrier(close, enter, pt_sl, max_holding, target=None, side=None):
    """

    :param close: (pd.Series) Close prices
    :param enter: (pd.Series) of entry points. These are timestamps that will seed every triple barrier.
        These are the timestamps
    :param pt_sl: (2 element array) Element 0, indicates the profit taking level; Element 1 is stop loss level.
        A non-negative float that sets the width of the two barriers. (if target is not None, pt_sl is ratio)
    :param target: (pd.Series) target rate
    :param max_holding: (2 element list) [days, hours]
    :param side: (pd.Series) Side of the bet (long/short) as decided by the primary model.
        1 if long, -1 if short
    :return: (pd.DataFrame) Events
            -events.index is event's starttime
            -events['exit'] is event's endtime
            -events['side'] implies the algo's position side
            -events['ret'] is return of each bet
    """

    # 1) Get target
    if target is None:
        target_ = pd.Series(1,index=enter)
    
    else:
        target_ = target.reindex(enter)

    # 2) Get vertical barrier (max holding period)
    vertical_barrier = add_vertical_barrier(enter, close, days=max_holding[0], hours=max_holding[1])

    # 3) Form events object, apply stop loss on vertical barrier
    if side is None:
        side_ = pd.Series(1.0, index=target_.index)
        pt_sl_ = [pt_sl[0], pt_sl[1]]
    else:
        side_ = side.reindex(target_.index)  # Subset side_prediction on target index.
        pt_sl_ = pt_sl[:2]

    # Create a new df
    events = pd.concat({'exit': vertical_barrier, 'trgt': target_,'side': side_}, axis=1)

    # Apply Triple Barrier
    first_touch_dates = forming_barriers(close, events, pt_sl_, events.index)
    

    for ind in events.index:
        events.at[ind, 'exit'] = first_touch_dates.loc[ind, :].dropna().min()

    events_x = events.dropna(subset=['exit'])

    out_df = pd.DataFrame(index=events.index)
    out_df['exit'] = events['exit']
    out_df['price'] = close
    out_df['ret'] = 0
    out_df.loc[events_x.index,'ret'] = (np.log(close.loc[events_x['exit'].array].array) - np.log(close.loc[events_x.index])) * events['side']
    out_df['side'] = events['side']
    return out_df


def get_barrier_fast(close, enter, pt_sl, max_holding, target=None, side=None):
    """
    Optimized version of get_barrier.
    """
    if target is None:
        target_ = pd.Series(1.0, index=enter)
    else:
        target_ = target.reindex(enter)

    vertical_barrier = add_vertical_barrier(enter, close, days=max_holding[0], hours=max_holding[1])

    if side is None:
        side_ = pd.Series(1.0, index=target_.index)
        pt_sl_ = [pt_sl[0], pt_sl[1]]
    else:
        side_ = side.reindex(target_.index)
        pt_sl_ = pt_sl[:2]

    events = pd.concat({'exit': vertical_barrier, 'trgt': target_,'side': side_}, axis=1)
    first_touch_dates = forming_barriers_fast(close, events, pt_sl_, events.index)
    
    events['exit'] = first_touch_dates.min(axis=1)
    events_x = events.dropna(subset=['exit'])

    out_df = pd.DataFrame(index=events.index)
    out_df['exit'] = events['exit']
    out_df['price'] = close.reindex(events.index)
    out_df['side'] = events['side']
    out_df['ret'] = 0.0
    
    if not events_x.empty:
        exit_prices = close.loc[events_x['exit']].values
        entry_prices = close.loc[events_x.index].values
        out_df.loc[events_x.index, 'ret'] = (np.log(exit_prices) - np.log(entry_prices)) * events_x['side'].values
        
    return out_df


def _generate_date_range(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m')
    end = datetime.strptime(end_date, '%Y-%m')
    date_list = []
    current_dt = start
    while current_dt <= end:
        date_list.append(current_dt.strftime('%Y-%m'))
        # current += timedelta(days=1)
        if current_dt.month == 12:
                current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
        else:
                current_dt = current_dt.replace(month=current_dt.month + 1)
    return date_list



if __name__ == "__main__":

    start_date = '2025-01'
    end_date = '2025-03'
    
    df_list = []
    date_list = _generate_date_range(start_date, end_date)
    for date in date_list:
        file_path = f'/Users/aming/data/ETHUSDT/1h/ETHUSDT-1h-{date}.zip'
        df = pd.read_csv(file_path)
        df_list.append(df)
        print(f'df{len(df)}, read df_list {len(df_list)} done')

    raw_data = pd.concat(df_list, ignore_index=True)
    raw_data['open_time'] = pd.to_datetime(raw_data['open_time'], unit='ms')
    raw_data.set_index('open_time', inplace=True)
    
    close = raw_data['close'].astype(float)
    log_close = np.log(close)
    ret = log_close.diff()
    sigma = ret.ewm(span=240).std()
    threshold = (2.0 * sigma).bfill()

    events = cusum_filter_dynamic(log_close, threshold=threshold)

    # 价格曲线
    plt.figure(figsize=(12, 5))
    plt.plot(close.index, close.values, label="close", linewidth=1.2)
    # 触发点（红色散点）
    plt.scatter(events, close.loc[events], color="red", s=20, label="cusum events (adaptive thr)", zorder=3)
    
    plt.title("Close with CUSUM Event Marks")
    plt.xlabel("time")
    plt.ylabel("price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("picture/cusum_event_marks.png", dpi=150, bbox_inches="tight")
    # plt.show()

    barrier_events = get_barrier(close, events, pt_sl=[1.2, 1.0], max_holding=[0, 10], target=sigma)
    profit_events = barrier_events[barrier_events["ret"] > 0]
    loss_events = barrier_events[barrier_events["ret"] < 0]
    neutral_events = barrier_events[barrier_events["ret"] == 0]

    profit_count = len(profit_events)
    loss_count = len(loss_events)
    total_count = len(barrier_events)
    non_neutral_count = profit_count + loss_count

    if non_neutral_count > 0:
        profit_ratio_non_neutral = profit_count / non_neutral_count
        loss_ratio_non_neutral = loss_count / non_neutral_count
    else:
        profit_ratio_non_neutral = 0.0
        loss_ratio_non_neutral = 0.0

    if total_count > 0:
        profit_ratio_total = profit_count / total_count
        loss_ratio_total = loss_count / total_count
    else:
        profit_ratio_total = 0.0
        loss_ratio_total = 0.0

    # 价格曲线
    plt.figure(figsize=(12, 5))
    plt.plot(close.index, close.values, label="close", linewidth=1.2)
    # 区分止盈、止损和未触发水平障碍（由垂直障碍退出）
    plt.scatter(
        profit_events.index,
        profit_events["price"],
        color="green",
        s=20,
        label="take profit",
        zorder=3,
    )
    plt.scatter(
        loss_events.index,
        loss_events["price"],
        color="orange",
        s=20,
        label="stop loss",
        zorder=3,
    )
    if not neutral_events.empty:
        plt.scatter(
            neutral_events.index,
            neutral_events["price"],
            color="gray",
            s=18,
            label="vertical exit",
            zorder=2,
        )
    
    plt.title("Close with Barrier Event Marks")
    plt.xlabel("time")
    plt.ylabel("price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("picture/barrier_event_marks.png", dpi=150, bbox_inches="tight")
    # plt.show()

    # 参数网格遍历并保存统计结果
    cusum_span_grid = [120, 240, 360]
    cusum_k_grid = [1.5, 2.0, 2.5]
    pt_sl_grid = [(1.0, 1.0), (1.2, 1.0), (1.5, 1.0)]
    max_holding_grid = [(0, 6), (0, 10), (0, 24)]

    grid_rows = []
    for span, k, (pt, sl), (hold_days, hold_hours) in product(
        cusum_span_grid, cusum_k_grid, pt_sl_grid, max_holding_grid
    ):
        sigma_i = ret.ewm(span=span).std()
        threshold_i = (k * sigma_i).bfill()
        events_i = cusum_filter_dynamic(log_close, threshold=threshold_i)

        barrier_i = get_barrier(
            close,
            events_i,
            pt_sl=[pt, sl],
            max_holding=[hold_days, hold_hours],
            target=sigma_i,
        )

        profit_i = barrier_i[barrier_i["ret"] > 0]
        loss_i = barrier_i[barrier_i["ret"] < 0]
        neutral_i = barrier_i[barrier_i["ret"] == 0]

        profit_count_i = len(profit_i)
        loss_count_i = len(loss_i)
        neutral_count_i = len(neutral_i)
        total_count_i = len(barrier_i)
        non_neutral_count_i = profit_count_i + loss_count_i

        if non_neutral_count_i > 0:
            profit_ratio_non_neutral_i = profit_count_i / non_neutral_count_i
            loss_ratio_non_neutral_i = loss_count_i / non_neutral_count_i
        else:
            profit_ratio_non_neutral_i = 0.0
            loss_ratio_non_neutral_i = 0.0

        if total_count_i > 0:
            profit_ratio_total_i = profit_count_i / total_count_i
            loss_ratio_total_i = loss_count_i / total_count_i
            neutral_ratio_total_i = neutral_count_i / total_count_i
        else:
            profit_ratio_total_i = 0.0
            loss_ratio_total_i = 0.0
            neutral_ratio_total_i = 0.0

        grid_rows.append(
            {
                "cusum_span": span,
                "cusum_k": k,
                "pt": pt,
                "sl": sl,
                "max_holding_days": hold_days,
                "max_holding_hours": hold_hours,
                "cusum_events_count": len(events_i),
                "barrier_events_count": total_count_i,
                "take_profit_count": profit_count_i,
                "stop_loss_count": loss_count_i,
                "vertical_exit_count": neutral_count_i,
                "take_profit_ratio_non_neutral": profit_ratio_non_neutral_i,
                "stop_loss_ratio_non_neutral": loss_ratio_non_neutral_i,
                "take_profit_ratio_total": profit_ratio_total_i,
                "stop_loss_ratio_total": loss_ratio_total_i,
                "vertical_exit_ratio_total": neutral_ratio_total_i,
            }
        )

    grid_df = pd.DataFrame(grid_rows)
    grid_df = grid_df.sort_values(
        by=["take_profit_ratio_non_neutral", "take_profit_count"], ascending=False
    ).reset_index(drop=True)
    os.makedirs("results", exist_ok=True)
    output_csv = "results/barrier_grid_results.csv"
    grid_df.to_csv(output_csv, index=False)
    print(f"grid search results saved: {output_csv}, rows={len(grid_df)}")
    
