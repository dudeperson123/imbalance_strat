import csv
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
import os

PT = ZoneInfo("America/Los_Angeles")

# Set to windows you want to test immediately; if optimizing, set to None
fast_test = None # 172021
slow_test = None # 769924

# instance password: 
# oYtK?LmS9$XDOCYSjwuTB*vNQ(hmH=u3

# LOCKED max cagr params at observed time: 5 days, 16:29:18 (21 trading days) on friday feb 27, Calendar days: 34, trading weeks: 4.9
# 231104, 602715

# Config

CSV_FILENAME = "data/imbalance_prices.csv"  # /Users/SaiSanjayD/Documents/PythonPrograms/imbalance_prices.csv

FAST_MIN = 1  # 1
FAST_MAX = 1000000  # 1000000
FAST_STEP = 10000  # 10000
SLOW_MIN = 1  # 1
SLOW_MAX = 1000000  # 1000000
SLOW_STEP = 10000  # 10000

TRADE_FEE = 0.0003  # 0.0003 is a safe, conservative estimate.
# IMPORTANT: Have a 0.06% minimum avg return when going live to ensure no loss of money

# Weights - decide how much the optimizer values each. tweak to need
w_avg_ret = 0
w_cagr = 1

OPT_COLUMN = "imbalance"

# ---------------------------------------------------------------------
# Compute total trading seconds within market hours (maturity-aware)
# ---------------------------------------------------------------------
def compute_liquid_trading_seconds_matured(
    ts: np.ndarray,
    liq: np.ndarray,
    mature: np.ndarray,
) -> tuple[float, int]:
    """
    Computes total trading seconds and number of days,
    counting only timestamps where:
        liquidhours == 1 AND slow window is matured
    """
    tradable_mask = (liq == 1) & (mature == 1)
    if not np.any(tradable_mask):
        return 0.0, 0

    tradable_ts = ts[tradable_mask].astype("datetime64[s]")
    dates = tradable_ts.astype("datetime64[D]")

    # Find contiguous segments per day and sum (last-first) per day,
    # without allocating per-day arrays in a Python loop.
    # dates is sorted since ts is in-order and we only mask.
    day_change = dates[1:] != dates[:-1]
    # start indices of each day segment
    starts = np.concatenate(([0], np.nonzero(day_change)[0] + 1))
    # end indices (inclusive)
    ends = np.concatenate((np.nonzero(day_change)[0], [dates.size - 1]))

    # Compute seconds difference per day segment
    # (datetime64[s] subtraction yields timedelta64[s])
    total_seconds = float(np.sum((tradable_ts[ends] - tradable_ts[starts]).astype("timedelta64[s]").astype(np.int64)))
    trading_days = int(starts.size)
    return total_seconds, trading_days


# ---------------------------------------------------------------------
# Vectorized backtest using fast/slow average crossovers
# ---------------------------------------------------------------------
def backtest(
    fast_window: int,
    slow_window: int,
    timestamps: np.ndarray,
    imbalances: np.ndarray,
    prices: np.ndarray,
    liquidhours: np.ndarray,
    length,
    w_avg_ret: float,
    w_cagr: float,
):
    if fast_window >= slow_window:
        return (
            -np.inf,
            [],
            (np.zeros_like(imbalances, dtype=float), np.zeros_like(imbalances, dtype=float)),
            [],
            0.0,
        )

    n = len(timestamps)
    # --- Rolling averages ---
    cumsum = np.cumsum(imbalances, dtype=np.float64)

    # fast rolling average
    roll_fast = np.empty(n, dtype=np.float64)
    fw = int(fast_window)
    if fw > 0:
        roll_fast[:fw] = cumsum[:fw] / np.arange(1, fw + 1, dtype=np.float64)
        roll_fast[fw:] = (cumsum[fw:] - cumsum[:-fw]) / float(fw)
    else:
        roll_fast.fill(0.0)

    # slow rolling average
    roll_slow = np.empty(n, dtype=np.float64)
    sw = int(slow_window)
    roll_slow[:sw] = cumsum[:sw] / np.arange(1, sw + 1, dtype=np.float64)
    roll_slow[sw:] = (cumsum[sw:] - cumsum[:-sw]) / float(sw)

    # --- Crossover signals ---
    diff = roll_fast - roll_slow
    sign = np.zeros_like(diff, dtype=np.int8)
    sign[diff > 0.0] = 1
    sign[diff < 0.0] = -1
    prev_sign = np.roll(sign, 1)
    prev_sign[0] = 0
    valid = np.arange(n) >= sw
    sign_change = (sign != prev_sign) & (sign != 0) & valid
    trade_indices = np.nonzero(sign_change)[0]

    # Only allow trades during liquid hours
    if trade_indices.size:
        trade_indices = trade_indices[liquidhours[trade_indices] == 1]

    trade_sides = np.where(sign[trade_indices] == 1, "LONG", "SHORT")

    # --- Force close at end of liquid hours each day ---
    dates = timestamps.astype("datetime64[D]")
    mature = np.zeros(length, dtype=np.int8)
    mature[sw:] = 1

    # Identify liquid & mature indices
    liq_idx = np.nonzero((liquidhours == 1) & (mature == 1))[0]
    if liq_idx.size:
        liq_dates = dates[liq_idx]
        liq_day_change = liq_dates[1:] != liq_dates[:-1]
        liq_ends = np.concatenate((np.nonzero(liq_day_change)[0], [liq_idx.size - 1]))
        last_liq_indices = liq_idx[liq_ends]
        if last_liq_indices.size:
            trade_indices = np.concatenate((trade_indices, last_liq_indices))
            trade_sides = np.concatenate((trade_sides, np.full(last_liq_indices.size, "CLOSE", dtype=object)))

    # --- New: First-of-day trades ---
    # Identify first liquid & mature index per day; THIS BLOCK CAN BE COMMENTED TO DISABLE
    if liq_idx.size:
        first_day_mask = np.concatenate(([True], liq_dates[1:] != liq_dates[:-1]))
        first_liq_indices = liq_idx[first_day_mask]

        # Determine trade side based on fast vs slow MA
        first_trade_sides = np.where(
            roll_fast[first_liq_indices] > roll_slow[first_liq_indices],
            "LONG",
            "SHORT"
        )

        # Append first-of-day trades
        trade_indices = np.concatenate((trade_indices, first_liq_indices))
        trade_sides = np.concatenate((trade_sides, first_trade_sides))

    # --- Sort trades chronologically ---
    trade_times = timestamps[trade_indices]
    sort_idx = np.argsort(trade_times, kind="mergesort")
    trade_indices = trade_indices[sort_idx]
    trade_sides = trade_sides[sort_idx]

    trade_prices = prices[trade_indices]

    # --- Compute returns ---
    m = trade_indices.size
    pct_change = np.zeros(m, dtype=np.float64)
    if m > 1:
        prev_prices = trade_prices[:-1]
        curr_prices = trade_prices[1:]
        prev_sides = trade_sides[:-1]

        long_mask = prev_sides == "LONG"
        short_mask = prev_sides == "SHORT"

        r = np.zeros(m - 1, dtype=np.float64)
        if np.any(long_mask):
            r[long_mask] = (curr_prices[long_mask] - prev_prices[long_mask]) / prev_prices[long_mask] - TRADE_FEE
        if np.any(short_mask):
            r[short_mask] = (prev_prices[short_mask] - curr_prices[short_mask]) / prev_prices[short_mask] - TRADE_FEE

        pct_change[1:] = r * 100.0
        include_mask = long_mask | short_mask
        returns_array = r[include_mask]
    else:
        returns_array = np.array([], dtype=np.float64)

    # Compute trading time
    total_seconds, trading_days = compute_liquid_trading_seconds_matured(timestamps, liquidhours, mature)
    trading_hours = total_seconds / 3600.0

    if returns_array.size:
        total_compounded_return = float(np.prod(1.0 + returns_array) - 1.0)
        cagr = (1.0 + total_compounded_return) ** (1629.0 / trading_hours) - 1.0 if trading_hours > 0 else 0.0
        avg_return_per_trade = float(np.mean(returns_array))
        ### Below is test
        # equity_curve = np.cumprod(np.insert(1.0 + returns_array, 0, 1.0))
        # roll_max = np.maximum.accumulate(equity_curve)
        # drawdowns = (equity_curve - roll_max) / roll_max
        # max_drawdown = float(drawdowns.min())
        ### End test
        if avg_return_per_trade < TRADE_FEE * 2:
            objective = -np.inf
        else:
            objective = cagr # TEST TOO
    else:
        objective = -np.inf 

    # Trade log
    time_hms = np.datetime_as_string(timestamps[trade_indices], unit="s")
    trade_log = [
        f"{time_hms[i]} {trade_sides[i]} @ {trade_prices[i]} ({pct_change[i]:.4f}%)"
        for i in range(m)
    ]

    return objective, trade_log, (roll_fast, roll_slow), returns_array.tolist(), trading_hours


# ---------------------------------------------------------------------
# Optimized grid search for window pairs (fast/slow)
# ---------------------------------------------------------------------
def optimize_windows(
    ts_data,
    imb_data,
    price_data,
    liq_data,
    length,
    w_avg_ret,
    w_cagr,
    it,
    it_len,
    SLOW_MIN,
    SLOW_MAX,
    SLOW_STEP,
    FAST_MIN,
    FAST_MAX,
    FAST_STEP,
):
    best_fast = FAST_MIN
    best_slow = SLOW_MIN
    best_objective = -np.inf 
    best_trade_log = None
    best_roll_avgs = None
    best_returns = None

    fast_windows = np.arange(FAST_MIN, FAST_MAX + 1, FAST_STEP, dtype=np.int32)
    slow_windows = np.arange(SLOW_MIN, SLOW_MAX + 1, SLOW_STEP, dtype=np.int32)

    # Number of valid (f, s) with f < s — needed to keep tqdm behavior
    # Keep semantics but speed by precomputing counts via searchsorted vectorization
    # For each fast, count slow values > fast.
    # idx = first index where slow_windows > fast
    idxs = np.searchsorted(slow_windows, fast_windows, side="right")
    total = int(np.sum(slow_windows.size - idxs[idxs < slow_windows.size]))

    with tqdm(total=total, desc=f"Optimizing {it}/{it_len}") as pbar:
        for fast, s_start in zip(fast_windows, idxs):
            if s_start >= slow_windows.size:
                continue

            for slow in slow_windows[s_start:]:
                obj, tlog, ravgs, rets, trading_hours = backtest(
                    int(fast),
                    int(slow),
                    ts_data,
                    imb_data,
                    price_data,
                    liq_data,
                    length,
                    w_avg_ret,
                    w_cagr,
                )
                if obj > best_objective:
                    best_fast = int(fast)
                    best_slow = int(slow)
                    best_objective = obj
                    best_trade_log = tlog
                    best_roll_avgs = ravgs
                    best_returns = rets

                    tqdm.write(
                        f"New Best | best_fast: {best_fast}, best_slow: {best_slow} | ann_mult: x{round(obj+1, 1)}"
                    )

                pbar.update(1)

    return (best_fast, best_slow), best_objective, best_trade_log, best_roll_avgs, best_returns


def compute_warmup_seconds(ts: np.ndarray, min_window: int) -> float:
    if ts.size <= min_window:
        return 0.0
    t0 = ts[0].astype("datetime64[s]").item()
    t1 = ts[min_window].astype("datetime64[s]").item()
    return (t1 - t0).total_seconds()


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------
def main():
    # Load CSV once
    # Faster CSV parsing: avoid DictReader overhead. Keep exact semantics of parsing.
    with open(CSV_FILENAME, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Find indices once
        try:
            now_i = header.index("now")
            imb_i = header.index(OPT_COLUMN) # change to norm_imbalance if want to test with normalized imbalance.
            liq_i = header.index("liquidhours")
            prc_i = header.index("price")
        except ValueError as e:
            raise ValueError(f"CSV missing required column: {e}")

        rows = list(reader)

    length = len(rows)
    min_required = max(FAST_MAX, SLOW_MAX) if not fast_test and not slow_test else max(fast_test, slow_test)
    if length < min_required:
        print(f"Not enough data in {CSV_FILENAME}.")
        print(f"Current number of rows:   {length}")
        print(f"Required number of rows:  {min_required}")
        raise ValueError(f"Not enough data for window={min_required}")

    ts = np.empty(length, dtype="datetime64[ns]")
    imb = np.empty(length, dtype=np.float64)
    prc = np.empty(length, dtype=np.float64)
    liq = np.empty(length, dtype=np.int8)

    # Slightly faster parsing loop, but preserves semantics
    fromiso = datetime.fromisoformat
    PT_local = PT
    for i, row in enumerate(rows):
        dt = fromiso(row[now_i])
        if dt.tzinfo is not None:
            dt = dt.astimezone(PT_local).replace(tzinfo=None)
        ts[i] = np.datetime64(dt)
        imb[i] = float(row[imb_i])
        prc[i] = float(row[prc_i])
        liq[i] = int(row[liq_i])

    if fast_test is None and slow_test is None:
        # Run grid search for optimal (fast, slow)
        it_len = 5  # 5 total iterations; decoration purposes only
        (best_fast, best_slow), best_objective, best_trade_log, best_roll_avgs, best_returns = optimize_windows(
            ts, imb, prc, liq, length, w_avg_ret, w_cagr, 1, it_len,
            SLOW_MIN, SLOW_MAX, SLOW_STEP, FAST_MIN, FAST_MAX, FAST_STEP,
        )  # Comment/uncomment below for testing purposes
        (best_fast, best_slow), best_objective, best_trade_log, best_roll_avgs, best_returns = optimize_windows(
            ts, imb, prc, liq, length, w_avg_ret, w_cagr, 2, it_len,
            max(best_slow - 50000, 1), min(best_slow + 50000, SLOW_MAX), 1000, max(best_fast - 50000, 1), min(best_fast + 50000, FAST_MAX), 1000
        )
        (best_fast, best_slow), best_objective, best_trade_log, best_roll_avgs, best_returns = optimize_windows(
            ts, imb, prc, liq, length, w_avg_ret, w_cagr, 3, it_len,
            max(best_slow - 5000, 1), min(best_slow + 5000, SLOW_MAX), 100, max(best_fast - 5000, 1), min(best_fast + 5000, FAST_MAX), 100
        )
        (best_fast, best_slow), best_objective, best_trade_log, best_roll_avgs, best_returns = optimize_windows(
            ts, imb, prc, liq, length, w_avg_ret, w_cagr, 4, it_len,
            max(best_slow - 500, 1), min(best_slow + 500, SLOW_MAX), 10, max(best_fast - 500, 1), min(best_fast + 500, FAST_MAX), 10
        )
        (best_fast, best_slow), best_objective, best_trade_log, best_roll_avgs, best_returns = optimize_windows(
            ts, imb, prc, liq, length, w_avg_ret, w_cagr, 5, it_len,
            max(best_slow - 50, 1), min(best_slow + 50, SLOW_MAX), 1, max(best_fast - 50, 1), min(best_fast + 50, FAST_MAX), 1
        )

    else:
        it_len = 1  # only 1 iteration, only for decoration
        (best_fast, best_slow), best_objective, best_trade_log, best_roll_avgs, best_returns = optimize_windows(
            ts,
            imb,
            prc,
            liq,
            length,
            w_avg_ret,
            w_cagr,
            1,
            it_len,
            slow_test,
            slow_test,
            1,
            fast_test,
            fast_test,
            1,
        )

    # Slow window maturity mask
    mature = np.zeros(length, dtype=np.int8)
    mature[best_slow:] = 1

    # Compute trading time using maturity-aware logic
    total_seconds, trading_days = compute_liquid_trading_seconds_matured(ts, liq, mature)
    calendar_days = (ts[-1].astype("datetime64[D]") - ts[0].astype("datetime64[D]")).astype(int) + 1
    trading_weeks = round(calendar_days / 7, 1)
    trading_hours = total_seconds / 3600.0

    # --- Compute summary metrics ---
    trade_times, trade_prices_clean, trade_sides = [], [], []
    for entry in best_trade_log:
        time_str, side_str, _, price_str, _ = entry.split()
        trade_times.append(datetime.fromisoformat(time_str))
        trade_prices_clean.append(float(price_str))
        trade_sides.append(side_str)

    returns = np.array(best_returns, dtype=np.float64)
    total_trades = len(returns)

    # --- Time metrics ---
    trading_minutes = total_seconds / 60.0
    hh_mm_ss = str(timedelta(seconds=int(total_seconds)))
    trading_frequency = trading_minutes / total_trades if total_trades > 0 else np.nan
    compounded_return = float(np.prod(1.0 + returns) - 1.0) if total_trades > 0 else 0.0
    avg_return = float(np.mean(returns)) if total_trades > 0 else 0.0
    return_per_day = round(avg_return * total_trades * 100 / trading_days, 2)
    cagr = (1.0 + compounded_return) ** (1629.0 / trading_hours) - 1.0 if trading_hours > 0 else 0.0
    win_rate = float(np.sum(returns > 0.0) / total_trades) if total_trades > 0 else 0.0

    equity_curve = np.cumprod(np.insert(1.0 + returns, 0, 1.0)) if total_trades > 0 else np.array([1.0], dtype=np.float64)

    if equity_curve.size > 0:
        roll_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - roll_max) / roll_max
        max_drawdown = float(drawdowns.min())
    else:
        max_drawdown = 0.0

    if total_trades > 1 and np.std(returns) != 0.0:
        sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(total_trades))
    else:
        sharpe_ratio = 0.0

    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0.0 else np.nan
    downside_returns = returns[returns < 0.0]
    if downside_returns.size > 0 and np.std(downside_returns) != 0.0:
        sortino_ratio = float(np.mean(returns) / np.std(downside_returns) * np.sqrt(total_trades))
    else:
        sortino_ratio = np.nan

    if total_trades > 0:
        gross_profit = float(np.sum(returns[returns > 0.0]))
        gross_loss = float(-np.sum(returns[returns < 0.0]))
    else:
        gross_profit = 0.0
        gross_loss = 0.0
    profit_factor = gross_profit / gross_loss if gross_loss != 0.0 else np.nan

    # Print trade log & summary
    print("Trade Log:\n")
    for entry in best_trade_log:
        print(entry)

    print("\n=== Backtest Summary ===")
    print(f"Optimal windows: fast={best_fast}, slow={best_slow}")
    print(f"Total trades: {total_trades}")
    print(f"Total time trading (observed): {hh_mm_ss} ({trading_days} trading days)")
    print(f"Calendar days: {calendar_days}, trading weeks: {trading_weeks}")
    print(f"Trading frequency: {trading_frequency:.2f} min/trade")
    print(f"Total compounded return: {compounded_return*100:.2f}%")
    print(f"Average return per trade (for optimal window pair): {avg_return*100:.6f}%")
    print(f"Non-compounded average return per day: {return_per_day}%")
    print(f"CAGR: {cagr*100:.2f}% (x{round(cagr+1, 1)})")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Max drawdown: {max_drawdown*100:.2f}%")
    print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"Calmar ratio: {calmar_ratio:.2f}")
    print(f"Sortino ratio: {sortino_ratio:.2f}")

    # Imbalance line plot(Plotly)
    best_roll_fast, best_roll_slow = best_roll_avgs

    # Mature & liquid mask
    mask = (liq == 1) & (np.arange(len(ts)) >= best_slow)

    fig = go.Figure()

    # Continuous rolling averages
    fig.add_trace(go.Scatter(
        x=ts,
        y=best_roll_fast,
        mode='lines',
        name=f'Fast MA ({best_fast})',
        line=dict(width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=ts,
        y=best_roll_slow,
        mode='lines',
        name=f'Slow MA ({best_slow})',
        line=dict(width=2)
    ))

    # Highlight trading time
    # Use semi-transparent rectangle shapes for efficiency
    # We collapse consecutive True values to ranges
    mask_diff = np.diff(mask.astype(int))
    starts = np.where(mask_diff == 1)[0] + 1
    ends = np.where(mask_diff == -1)[0] + 1

    # Handle edge cases
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask)-1)

    for s, e in zip(starts, ends):
        fig.add_vrect(
            x0=ts[s],
            x1=ts[e],
            fillcolor="green",
            opacity=0.15,
            layer="below",
            line_width=0,
            annotation_text="",
        )

    fig.update_layout(
        title="Rolling Imbalance Averages (Continuous 27/4 with Trading Highlight)",
        xaxis_title="Date & Time",
        yaxis_title="Imbalance",
        legend_title="Lines",
        hovermode=False
    )

    fig.show()

    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, lw=1.5, label="Equity Curve")
    plt.title("Equity Curve")
    plt.xlabel("Trade Index")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
