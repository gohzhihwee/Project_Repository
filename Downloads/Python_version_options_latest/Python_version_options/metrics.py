import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Hedge-fund portfolio metrics
# ---------------------------------------------------------------------------

def compute_portfolio_metrics(
    equity_curve: pd.DataFrame,
    weekly_returns: list,
    trade_log: list,
    spy_returns: Optional[pd.Series] = None,
    initial_capital: float = 100_000.0,
    ann_factor: int = 52,
) -> dict:
    """
    Compute hedge-fund-grade portfolio metrics from backtest data.

    Parameters
    ----------
    equity_curve : DataFrame with DatetimeIndex and 'value' column (weekly)
    weekly_returns : list of weekly decimal returns (from _all_weekly_returns)
    trade_log : list of {date, contract, qty, entry_price, exit_price, pnl}
    spy_returns : optional weekly SPY decimal returns aligned to equity dates
    initial_capital : starting portfolio value (default $100k)
    ann_factor : annualisation multiplier (52 for weekly data)

    Returns
    -------
    dict with keys:
        cagr, ann_vol, sharpe, sortino, calmar,
        max_drawdown, max_dd_weeks,
        var_95_pct, var_99_pct, cvar_95_pct,
        win_rate_pct, profit_factor, avg_win, avg_loss, n_trades,
        total_return,
        alpha_pct, beta, info_ratio  (only if spy_returns provided)
    """
    if equity_curve.empty or len(weekly_returns) < 2:
        return {}

    rets = np.array(weekly_returns, dtype=float)
    values = equity_curve['value'].values

    # ── Annualised return (CAGR) ──────────────────────────────────────────
    n_years = len(rets) / ann_factor
    final_ratio = float(values[-1] / initial_capital)
    cagr = float(final_ratio ** (1.0 / max(n_years, 1e-9)) - 1.0) if final_ratio > 0 else -1.0

    # ── Annualised volatility ─────────────────────────────────────────────
    ann_vol = float(rets.std(ddof=1) * np.sqrt(ann_factor))

    # ── Sharpe Ratio ──────────────────────────────────────────────────────
    std = float(rets.std(ddof=1))
    sharpe = float(rets.mean() / std * np.sqrt(ann_factor)) if std > 0 else 0.0

    # ── Sortino Ratio ─────────────────────────────────────────────────────
    # Uses downside deviation (returns < 0) only — more appropriate than
    # Sharpe for options strategies whose return distributions are asymmetric
    # (long gamma = right-skewed; short theta = left-skewed).  Penalising
    # upside variance equally (as Sharpe does) distorts the risk picture.
    downside = rets[rets < 0]
    ds_std = float(downside.std(ddof=1) * np.sqrt(ann_factor)) if len(downside) > 1 else 1e-9
    sortino = float(rets.mean() * ann_factor / ds_std) if ds_std > 1e-9 else 0.0

    # ── Max Drawdown + Duration ───────────────────────────────────────────
    cummax = np.maximum.accumulate(values)
    dd_series = (values - cummax) / cummax
    mdd = float(dd_series.min())
    max_dd_weeks = _max_consecutive(dd_series < -1e-6)

    # ── Calmar Ratio ──────────────────────────────────────────────────────
    # CAGR / |MDD| — captures leverage impact that Sharpe misses.
    # A Sharpe of 1.5 with a 50% drawdown is uninvestable.
    calmar = float(cagr / abs(mdd)) if mdd < -1e-9 else 0.0

    # ── Value at Risk and Conditional VaR (historical simulation) ────────
    # Historical VaR avoids normal-distribution assumption — critical for
    # options strategies with fat tails.  CVaR (Expected Shortfall) is
    # the average loss beyond VaR; it is a coherent risk measure and is
    # preferred by regulators (FRTB) and most LP DDQs over plain VaR.
    var_95 = float(np.percentile(rets, 5))
    var_99 = float(np.percentile(rets, 1))
    tail_95 = rets[rets <= var_95]
    cvar_95 = float(tail_95.mean()) if len(tail_95) > 0 else var_95

    # ── Trade statistics ──────────────────────────────────────────────────
    if trade_log:
        pnls = np.array([t['pnl'] for t in trade_log], dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        n_trades = int(len(pnls))
        win_rate = float(len(wins) / n_trades)
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss   = float(abs(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = (float(gross_profit / gross_loss) if gross_loss > 0
                         else (float('inf') if gross_profit > 0 else 0.0))
        avg_win  = float(wins.mean())   if len(wins)   > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    else:
        n_trades = 0
        win_rate = profit_factor = avg_win = avg_loss = 0.0

    out: dict = {
        'cagr':          round(cagr * 100, 2),
        'ann_vol':       round(ann_vol * 100, 2),
        'sharpe':        round(sharpe, 3),
        'sortino':       round(sortino, 3),
        'calmar':        round(calmar, 3),
        'max_drawdown':  round(mdd * 100, 2),
        'max_dd_weeks':  max_dd_weeks,
        'var_95_pct':    round(var_95 * 100, 2),
        'var_99_pct':    round(var_99 * 100, 2),
        'cvar_95_pct':   round(cvar_95 * 100, 2),
        'win_rate_pct':  round(win_rate * 100, 2),
        'profit_factor': round(min(profit_factor, 9999.0), 3),
        'avg_win':       round(avg_win, 2),
        'avg_loss':      round(avg_loss, 2),
        'n_trades':      n_trades,
        'total_return':  round((final_ratio - 1.0) * 100, 2),
    }

    # ── Alpha, Beta, Information Ratio vs SPY ────────────────────────────
    # Alpha = annualised CAPM excess return.  Beta shows correlation to SPY.
    # Information Ratio = active return / tracking error — how efficiently
    # the strategy generates alpha relative to deviation from the benchmark.
    if spy_returns is not None and len(spy_returns) >= 4:
        pair = _align_returns(rets, spy_returns)
        if pair is not None:
            port_r, spy_r = pair
            cov_mat = np.cov(port_r, spy_r, ddof=1)
            beta  = float(cov_mat[0, 1] / cov_mat[1, 1]) if cov_mat[1, 1] > 0 else 0.0
            alpha_w = float(port_r.mean() - beta * spy_r.mean())
            alpha = float(alpha_w * ann_factor)
            active = port_r - spy_r
            te = float(active.std(ddof=1) * np.sqrt(ann_factor))
            info_ratio = float(active.mean() * ann_factor / te) if te > 1e-9 else 0.0
            out['alpha_pct']  = round(alpha * 100, 2)
            out['beta']       = round(beta, 3)
            out['info_ratio'] = round(info_ratio, 3)

    return out


# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------

def compute_rolling_sharpe(
    weekly_returns: list,
    window: int = 26,
    ann_factor: int = 52,
) -> pd.Series:
    """Rolling annualised Sharpe over `window` weeks."""
    rets = pd.Series(weekly_returns, dtype=float)

    def _sharpe(r: pd.Series) -> float:
        s = float(r.std(ddof=1))
        return float(r.mean() / s * np.sqrt(ann_factor)) if s > 0 else 0.0

    return rets.rolling(window).apply(_sharpe, raw=False)


def compute_drawdown_series(values: np.ndarray) -> np.ndarray:
    """Percentage drawdown at each point (negative, e.g. -5.2 means 5.2% below peak)."""
    cummax = np.maximum.accumulate(values)
    return (values - cummax) / cummax * 100.0


# ---------------------------------------------------------------------------
# ML pricing-model metrics
# ---------------------------------------------------------------------------

def compute_mape(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (%).

    Uses `actuals` as the denominator, skipping entries where |actual| < 1e-6
    to avoid division by zero on deep-OTM near-zero options.

    Justification for use alongside RMSE/MAE:
      RMSE/MAE are in dollar terms and are dominated by ATM options (which have
      higher absolute prices).  MAPE is scale-invariant, so a $0.03 error on a
      $0.30 deep-OTM option (10%) is not swamped by a $0.10 error on a $5 ATM
      option (2%).  Using all three together gives a full error picture.
    """
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    mask = np.abs(actuals) > 1e-6
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs(predictions[mask] - actuals[mask]) / np.abs(actuals[mask])) * 100.0)


def compute_directional_accuracy(
    predictions: np.ndarray,
    actuals_entry: np.ndarray,
    actuals_exit: np.ndarray,
) -> float:
    """
    Percentage of predictions where the model correctly identified the direction
    of option mispricing:  sign(model_price − market_entry) == sign(exit − entry).

    Justification: signal generation only needs the model to be right about
    *direction*, not magnitude.  A model with low RMSE but directional accuracy
    near 50% is equivalent to a coin flip for trading — it will lose money after
    bid-ask spreads.  This metric bridges pricing accuracy and trading utility.
    """
    predictions   = np.asarray(predictions,   dtype=float)
    actuals_entry = np.asarray(actuals_entry, dtype=float)
    actuals_exit  = np.asarray(actuals_exit,  dtype=float)
    model_signal    = np.sign(predictions   - actuals_entry)
    realized_signal = np.sign(actuals_exit  - actuals_entry)
    valid = (model_signal != 0) & (realized_signal != 0)
    if valid.sum() == 0:
        return float('nan')
    return float((model_signal[valid] == realized_signal[valid]).mean() * 100.0)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _max_consecutive(bool_arr: np.ndarray) -> int:
    """Length of the longest run of True values."""
    max_run = cur = 0
    for v in bool_arr:
        cur = cur + 1 if v else 0
        max_run = max(max_run, cur)
    return int(max_run)


def _align_returns(
    port_rets: np.ndarray,
    spy_rets: pd.Series,
) -> Optional[tuple]:
    """Align two return arrays to the same (shorter) length."""
    n = min(len(port_rets), len(spy_rets))
    if n < 4:
        return None
    return port_rets[-n:], np.asarray(spy_rets.values[-n:], dtype=float).ravel()
