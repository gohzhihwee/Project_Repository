import json
import pathlib
from typing import Optional

import matplotlib
matplotlib.use('Agg')               # must come before pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm as sp_norm

from metrics import (
    compute_drawdown_series,
    compute_mape,
    compute_rolling_sharpe,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = pathlib.Path('/app/output')

_STRATEGY_DISPLAY: dict = {
    'BuyAndHoldStrategy':    'Buy & Hold',
    'MomentumStrategy':      'Momentum',
    'MeanReversionStrategy': 'Mean Reversion',
    'MMARStrategy':          'MMAR',
    'BlackScholesStrategy':  'Black-Scholes',
    'HestonStrategy':        'Heston',
    'MertonStrategy':        'Merton',
    'BatesStrategy':         'Bates',
    'MixedStrategy':         'Mixed Ensemble',
}

_STRATEGY_KEYS = list(_STRATEGY_DISPLAY.keys())

# Consistent figure style — configured manually to avoid seaborn version skew
_RCPARAMS = {
    'axes.facecolor':     '#f8f9fa',
    'figure.facecolor':   'white',
    'axes.grid':          True,
    'grid.color':         '#dddddd',
    'grid.linewidth':     0.7,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'font.size':          10,
}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def generate_visualizations(
    equity_curve: list,
    weekly_returns: list,
    spy_prices: pd.DataFrame,
    model_accuracy_metrics: Optional[dict],
    dm_results: Optional[dict],
    trade_log: list,
    gbr_meta_importances: Optional[dict],
    gbr_meta_train_rmse: Optional[float],
    strategy_selector_importances: Optional[dict],
    portfolio_metrics: dict,
    output_dir: pathlib.Path = OUTPUT_DIR,
) -> None:
    """
    Generate the full end-of-run chart suite and summary files.

    Charts written
    ──────────────
    01_equity_curve.png          portfolio vs SPY, metrics annotation
    02_drawdown.png              underwater (drawdown %) chart
    03_rolling_sharpe.png        26-week rolling annualised Sharpe
    04_return_distribution.png   histogram + normal overlay + VaR/CVaR
    05_monthly_heatmap.png       calendar heatmap of monthly P&L %
    06_model_performance.png     RMSE, MAE, MAPE, directional accuracy
    07_strategy_allocation.png   colour-coded strategy timeline
    08_notional_exposure.png     notional exposure ($) over time
    09_gbr_importance.png        GBR feature importances (pricing + selector)

    Summary files
    ─────────────
    backtest_summary.json        all portfolio and model metrics
    equity_curve.csv             weekly equity curve with strategy labels
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(_RCPARAMS)

    if not equity_curve:
        print('[VIZ] No equity curve data — skipping chart generation.')
        return

    curve_df = _build_curve_df(equity_curve)
    rets = np.array(weekly_returns, dtype=float)
    spy_weekly = _compute_spy_weekly(spy_prices, curve_df.index)

    _safe_chart(_chart_equity_curve, curve_df, spy_weekly, portfolio_metrics, output_dir)
    _safe_chart(_chart_drawdown, curve_df, spy_weekly, output_dir)
    _safe_chart(_chart_rolling_sharpe, rets, curve_df.index, output_dir)
    _safe_chart(_chart_return_distribution, rets, portfolio_metrics, output_dir)
    _safe_chart(_chart_monthly_heatmap, curve_df, output_dir)

    if model_accuracy_metrics:
        _safe_chart(_chart_model_performance, model_accuracy_metrics, dm_results, output_dir)

    _safe_chart(_chart_strategy_allocation, curve_df, output_dir)

    if 'notional_exposure' in curve_df.columns:
        _safe_chart(_chart_notional_exposure, curve_df, output_dir)

    if gbr_meta_importances or strategy_selector_importances:
        _safe_chart(_chart_gbr_importance,
            gbr_meta_importances, gbr_meta_train_rmse,
            strategy_selector_importances, output_dir,
        )

    _save_summary_json(portfolio_metrics, model_accuracy_metrics, dm_results, output_dir)
    _save_equity_csv(curve_df, output_dir)

    _print_manifest(output_dir, model_accuracy_metrics,
                    'notional_exposure' in curve_df.columns,
                    bool(gbr_meta_importances or strategy_selector_importances))


def save_interim_chart(
    equity_curve: list,
    output_dir: pathlib.Path = OUTPUT_DIR,
) -> None:
    """
    Write a lightweight equity-curve PNG during the backtest run so the user
    can watch progress in real time by opening ./output/00_live_equity_curve.png
    in any image viewer that auto-refreshes.  Always overwrites the same file.
    """
    if not equity_curve or len(equity_curve) < 2:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(_RCPARAMS)

    dates  = [pd.Timestamp(r['date']) for r in equity_curve]
    values = [r['value'] / 1_000 for r in equity_curve]
    n_rebalances = len(equity_curve)
    last_date = dates[-1].strftime('%Y-%m-%d')
    last_val  = values[-1]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, values, color='#1a5276', linewidth=2)
    ax.set_ylabel('Portfolio Value ($k)', fontsize=10)
    ax.set_title(
        f'Live Equity Curve  —  {last_date}  |  ${last_val:.1f}k  |  '
        f'{n_rebalances} rebalances',
        fontsize=11, fontweight='bold',
    )
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}k'))
    fig.tight_layout()
    fig.savefig(output_dir / '00_live_equity_curve.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart functions (private)
# ---------------------------------------------------------------------------

def _chart_equity_curve(
    curve_df: pd.DataFrame,
    spy_weekly: Optional[pd.Series],
    pm: dict,
    output_dir: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    # Portfolio line
    ax.plot(curve_df.index, curve_df['value'] / 1_000,
            color='#1a5276', linewidth=2.2, label='Options Portfolio', zorder=3)

    # SPY buy-and-hold normalised to same starting capital
    if spy_weekly is not None and len(spy_weekly) > 1:
        first_val = curve_df['value'].iloc[0] / 1_000
        spy_norm  = spy_weekly / spy_weekly.iloc[0] * first_val
        ax.plot(spy_norm.index, spy_norm.values,
                color='#e67e22', linewidth=1.8, linestyle='--',
                label='SPY Buy & Hold', zorder=2)

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Portfolio Value ($k)', fontsize=11)
    ax.set_title('Portfolio Equity Curve vs SPY Benchmark', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}k'))

    # Metrics annotation box — key numbers at a glance (institutional style)
    lines = [
        f"Total Return : {pm.get('total_return', 'N/A'):+}%",
        f"CAGR         : {pm.get('cagr', 'N/A')}%",
        f"Sharpe       : {pm.get('sharpe', 'N/A')}",
        f"Sortino      : {pm.get('sortino', 'N/A')}",
        f"Calmar       : {pm.get('calmar', 'N/A')}",
        f"Max DD       : {pm.get('max_drawdown', 'N/A')}%",
    ]
    if 'alpha_pct' in pm:
        lines += [
            f"Alpha (ann.) : {pm['alpha_pct']}%",
            f"Beta         : {pm['beta']}",
            f"Info Ratio   : {pm.get('info_ratio', 'N/A')}",
        ]
    ax.text(
        0.02, 0.97, '\n'.join(lines),
        transform=ax.transAxes, fontsize=8.5, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#aaaaaa', alpha=0.92),
    )

    fig.tight_layout()
    fig.savefig(output_dir / '01_equity_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _chart_drawdown(
    curve_df: pd.DataFrame,
    spy_weekly: Optional[pd.Series],
    output_dir: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    dd = compute_drawdown_series(curve_df['value'].values)
    ax.fill_between(curve_df.index, dd, 0, color='#c0392b', alpha=0.35, label='Portfolio DD')
    ax.plot(curve_df.index, dd, color='#922b21', linewidth=1.4)

    if spy_weekly is not None and len(spy_weekly) > 1:
        spy_aligned = spy_weekly.reindex(curve_df.index, method='ffill').dropna()
        spy_dd = compute_drawdown_series(spy_aligned.values)
        ax.plot(spy_aligned.index[: len(spy_dd)], spy_dd,
                color='#e67e22', linewidth=1.3, linestyle='--', label='SPY DD')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Drawdown (%)', fontsize=11)
    ax.set_title('Drawdown (Underwater) Chart', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / '02_drawdown.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _chart_rolling_sharpe(
    rets: np.ndarray,
    dates: pd.DatetimeIndex,
    output_dir: pathlib.Path,
    window: int = 26,
) -> None:
    if len(rets) < window + 1:
        return

    rs = compute_rolling_sharpe(list(rets), window=window)
    plot_dates = dates[: len(rs)]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(plot_dates, rs.values, color='#2980b9', linewidth=1.8)
    ax.axhline(0, color='black',   linewidth=0.9)
    ax.axhline(1, color='#27ae60', linewidth=0.9, linestyle='--', alpha=0.7, label='Sharpe = 1')
    ax.axhline(-1, color='#c0392b', linewidth=0.9, linestyle='--', alpha=0.7, label='Sharpe = −1')
    ax.fill_between(plot_dates, rs.values, 0,
                    where=(np.array(rs.values) >= 0), alpha=0.25, color='#27ae60')
    ax.fill_between(plot_dates, rs.values, 0,
                    where=(np.array(rs.values) < 0),  alpha=0.25, color='#c0392b')

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Annualised Sharpe', fontsize=11)
    ax.set_title(f'Rolling {window}-Week Sharpe Ratio', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / '03_rolling_sharpe.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _chart_return_distribution(
    rets: np.ndarray,
    pm: dict,
    output_dir: pathlib.Path,
) -> None:
    rets_pct = rets * 100
    mu  = float(rets_pct.mean())
    sig = float(rets_pct.std(ddof=1))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(rets_pct, bins=40, color='#2980b9', alpha=0.55, density=True,
            label='Weekly Returns', edgecolor='white', linewidth=0.4)

    # Normal distribution overlay — reveals fat tails and skewness
    x = np.linspace(rets_pct.min() - 1, rets_pct.max() + 1, 400)
    ax.plot(x, sp_norm.pdf(x, mu, sig), color='#2c3e50', linewidth=1.8,
            linestyle='--', label=f'Normal  μ={mu:.2f}%  σ={sig:.2f}%')

    # VaR and CVaR vertical lines
    var95  = pm.get('var_95_pct',  float(np.percentile(rets_pct, 5)))
    cvar95 = pm.get('cvar_95_pct', float(rets_pct[rets_pct <= var95].mean()))
    ax.axvline(var95,  color='#e74c3c', linewidth=1.8, linestyle=':',
               label=f'Historical VaR 95% = {var95:.2f}%')
    ax.axvline(cvar95, color='#8e44ad', linewidth=1.8, linestyle=':',
               label=f'CVaR / ES 95% = {cvar95:.2f}%')

    # Win-rate annotation
    win_rate = pm.get('win_rate_pct', None)
    pf       = pm.get('profit_factor', None)
    if win_rate is not None:
        ax.text(0.98, 0.97,
                f"Win rate:      {win_rate:.1f}%\n"
                f"Profit factor: {pf:.2f}\n"
                f"Avg win:      ${pm.get('avg_win', 0):,.0f}\n"
                f"Avg loss:     ${pm.get('avg_loss', 0):,.0f}\n"
                f"N trades:      {pm.get('n_trades', 0)}",
                transform=ax.transAxes, fontsize=8.5, va='top', ha='right',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='#aaa', alpha=0.9))

    ax.set_xlabel('Weekly Return (%)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Weekly Return Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    fig.tight_layout()
    fig.savefig(output_dir / '04_return_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _chart_monthly_heatmap(
    curve_df: pd.DataFrame,
    output_dir: pathlib.Path,
) -> None:
    # Resample weekly equity curve to month-end, compute monthly return
    monthly = curve_df['value'].resample('ME').last()
    monthly_ret = monthly.pct_change().dropna() * 100
    if len(monthly_ret) < 2:
        return

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot = (
        pd.DataFrame({
            'year':  monthly_ret.index.year,
            'month': monthly_ret.index.month,
            'ret':   monthly_ret.values,
        })
        .pivot(index='year', columns='month', values='ret')
    )
    pivot.columns = [month_labels[m - 1] for m in pivot.columns]
    # Full-year columns (fill missing months with NaN — displayed as blank)
    all_months_df = pd.DataFrame(index=pivot.index, columns=month_labels, dtype=float)
    all_months_df.update(pivot)

    vabs = float(np.nanmax(np.abs(all_months_df.values)))
    vabs = max(vabs, 1.0)

    fig, ax = plt.subplots(figsize=(14, max(3, len(all_months_df) * 0.85 + 1.5)))
    sns.heatmap(
        all_months_df.astype(float), ax=ax,
        cmap='RdYlGn', center=0, vmin=-vabs, vmax=vabs,
        linewidths=0.5, linecolor='white',
        annot=True, fmt='.1f', annot_kws={'size': 9},
        cbar_kws={'label': 'Monthly Return (%)', 'shrink': 0.75},
        mask=all_months_df.isna(),
    )
    ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Year', fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / '05_monthly_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _chart_model_performance(
    model_metrics: dict,
    dm_results: Optional[dict],
    output_dir: pathlib.Path,
) -> None:
    """
    Four-panel chart:
      TL — RMSE & MAE ($)           measures dollar prediction accuracy
      TR — MAPE (%)                 scale-invariant, fair across strikes
      BL — Directional Accuracy     trading utility (>50% = beats random)
      BR — Pearson Correlation      linear alignment of prediction & realization
    """
    models    = list(model_metrics.keys())
    if not models:
        return

    rmse_vals = [model_metrics[m].get('RMSE', 0.0) for m in models]
    mae_vals  = [model_metrics[m].get('MAE',  0.0) for m in models]
    mape_vals = [float(model_metrics[m].get('MAPE', float('nan'))) for m in models]
    dir_vals  = [float(model_metrics[m].get('Directional_Accuracy', float('nan'))) for m in models]
    corr_vals = [model_metrics[m].get('Correlation', 0.0) for m in models]

    x = np.arange(len(models))
    w = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Pricing Model Performance Metrics', fontsize=15, fontweight='bold', y=1.01)

    # ── RMSE & MAE ────────────────────────────────────────────────────────
    ax = axes[0, 0]
    b1 = ax.bar(x - w / 2, rmse_vals, w, label='RMSE ($)', color='#2980b9', alpha=0.85)
    b2 = ax.bar(x + w / 2, mae_vals,  w, label='MAE ($)',  color='#e67e22', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right')
    ax.set_ylabel('Pricing Error ($)')
    ax.set_title('RMSE & MAE  (lower = better)\n'
                 'Dollar-weighted; large errors penalised quadratically by RMSE')
    ax.legend(fontsize=9)
    _annotate_bars(ax, b1, '{:.3f}')
    _annotate_bars(ax, b2, '{:.3f}')

    # ── MAPE ─────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    valid_mape = [v for v in mape_vals if not np.isnan(v)]
    best_mape  = min(valid_mape) if valid_mape else -1
    colors_m = ['#27ae60' if (not np.isnan(v) and v == best_mape) else '#2980b9'
                for v in mape_vals]
    b = ax.bar(x, mape_vals, color=colors_m, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Mean Absolute % Error  (lower = better)\n'
                 'Scale-invariant: prevents ATM options dominating error stats')
    _annotate_bars(ax, b, '{:.1f}%', skip_nan=True)

    # ── Directional Accuracy ──────────────────────────────────────────────
    ax = axes[1, 0]
    colors_d = ['#27ae60' if (not np.isnan(v) and v >= 50) else '#c0392b'
                for v in dir_vals]
    b = ax.bar(x, dir_vals, color=colors_d, alpha=0.85)
    ax.axhline(50, color='black', linewidth=0.9, linestyle='--', alpha=0.55,
               label='Random baseline (50%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right')
    ax.set_ylabel('Directional Accuracy (%)')
    ax.set_title('Directional Accuracy  (>50% = beats random)\n'
                 'sign(model − entry) == sign(exit − entry): trading-signal quality')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    _annotate_bars(ax, b, '{:.1f}%', skip_nan=True)

    # ── Pearson Correlation ───────────────────────────────────────────────
    ax = axes[1, 1]
    b = ax.bar(x, corr_vals, color='#8e44ad', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right')
    ax.set_ylabel('Pearson Correlation')
    ax.set_title('Prediction–Realisation Correlation  (higher = better)\n'
                 'Linear alignment between model price and eventual exit price')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylim(-1, 1.15)
    _annotate_bars(ax, b, '{:.3f}')

    # ── Diebold–Mariano note ──────────────────────────────────────────────
    if dm_results:
        dm_lines = ['Diebold–Mariano tests (squared-error, vs Black-Scholes):']
        for k, v in dm_results.items():
            m = k.replace('_vs_BS', '')
            p = v.get('p_value', 1.0)
            sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else 'ns'))
            dm_lines.append(
                f"  {m:10s}  p={p:.3f} {sig}  →  favors {v.get('favors', '?')}"
            )
        fig.text(
            0.5, -0.05, '\n'.join(dm_lines),
            ha='center', fontsize=8.5, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#aaa', alpha=0.9),
        )

    fig.tight_layout()
    fig.savefig(output_dir / '06_model_performance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _chart_strategy_allocation(
    curve_df: pd.DataFrame,
    output_dir: pathlib.Path,
) -> None:
    if 'strategy' not in curve_df.columns:
        return

    cmap = plt.get_cmap('tab10')
    color_map = {s: cmap(i / max(len(_STRATEGY_KEYS) - 1, 1))
                 for i, s in enumerate(_STRATEGY_KEYS)}

    dates  = curve_df.index
    strats = curve_df['strategy'].fillna('Unknown')
    unique = strats.unique()

    fig, (ax_strip, ax_curve) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [0.7, 3]},
    )
    fig.suptitle('Strategy Allocation Timeline', fontsize=14, fontweight='bold')

    # ── Top strip: colour per week ────────────────────────────────────────
    for i in range(len(dates) - 1):
        s = strats.iloc[i]
        c = color_map.get(s, '#888888')
        width_days = (dates[i + 1] - dates[i]).days
        ax_strip.barh(0, width_days, left=dates[i], height=1, color=c, align='center')
    ax_strip.set_xlim(dates[0], dates[-1])
    ax_strip.set_yticks([])
    ax_strip.set_title('Active strategy each week (colour-coded)', fontsize=9)
    ax_strip.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # ── Bottom: equity curve coloured by strategy ─────────────────────────
    values_k = curve_df['value'].values / 1_000
    for i in range(len(dates) - 1):
        s = strats.iloc[i]
        c = color_map.get(s, '#888888')
        ax_curve.plot(dates[i: i + 2], values_k[i: i + 2], color=c, linewidth=2.2)
    ax_curve.set_xlabel('Date', fontsize=11)
    ax_curve.set_ylabel('Portfolio Value ($k)', fontsize=11)
    ax_curve.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}k'))

    patches = [
        mpatches.Patch(color=color_map.get(s, '#888'),
                       label=_STRATEGY_DISPLAY.get(s, s))
        for s in unique if s in color_map
    ]
    ax_curve.legend(handles=patches, fontsize=8, loc='upper left', ncol=3)

    fig.tight_layout()
    fig.savefig(output_dir / '07_strategy_allocation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _chart_notional_exposure(
    curve_df: pd.DataFrame,
    output_dir: pathlib.Path,
) -> None:
    notional_k = curve_df['notional_exposure'] / 1_000

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(curve_df.index, notional_k, 0, color='#2980b9', alpha=0.30)
    ax.plot(curve_df.index, notional_k, color='#1a5276', linewidth=1.8)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Notional Exposure ($k)', fontsize=11)
    ax.set_title('Options Notional Exposure Over Time', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}k'))
    fig.tight_layout()
    fig.savefig(output_dir / '08_notional_exposure.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _chart_gbr_importance(
    meta_importances: Optional[dict],
    meta_train_rmse: Optional[float],
    selector_importances: Optional[dict],
    output_dir: pathlib.Path,
) -> None:
    n_panels = (1 if meta_importances else 0) + (1 if selector_importances else 0)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle('GBR Meta-Learner Feature Importances', fontsize=14, fontweight='bold')

    idx = 0
    if meta_importances:
        ax = axes[idx]; idx += 1
        labels = list(meta_importances.keys())
        vals   = list(meta_importances.values())
        # Colour pricing-model features differently from auxiliary features
        colors = ['#2980b9' if '_pred' in l else '#e67e22' for l in labels]
        display_labels = [
            l.replace('_pred', '').upper().replace('_', ' ') for l in labels
        ]
        ax.barh(display_labels, vals, color=colors, alpha=0.85)
        ax.set_xlabel('Feature Importance (Gini impurity reduction)')
        rmse_str = (f'  —  train RMSE: ${meta_train_rmse:.4f}'
                    if meta_train_rmse is not None else '')
        ax.set_title(f'Pricing Ensemble GBR{rmse_str}', fontsize=11)
        # Blue = model price features, orange = market features
        blue_patch  = mpatches.Patch(color='#2980b9', label='Model price inputs')
        orng_patch  = mpatches.Patch(color='#e67e22', label='Market features (vol, moneyness, TTM)')
        ax.legend(handles=[blue_patch, orng_patch], fontsize=8)

    if selector_importances:
        ax = axes[idx]
        labels = list(selector_importances.keys())
        vals   = list(selector_importances.values())
        display_labels = [l.replace('_', ' ').title() for l in labels]
        ax.barh(display_labels, vals, color='#8e44ad', alpha=0.85)
        ax.set_xlabel('Feature Importance (mean across all strategy outputs)')
        ax.set_title('Strategy Selector GBR\n(which regime features drive selection?)', fontsize=11)

    fig.tight_layout()
    fig.savefig(output_dir / '09_gbr_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary file writers
# ---------------------------------------------------------------------------

def _save_summary_json(
    pm: dict,
    model_metrics: Optional[dict],
    dm_results: Optional[dict],
    output_dir: pathlib.Path,
) -> None:
    summary = {
        'portfolio':      pm,
        'model_accuracy': model_metrics,
        'diebold_mariano': dm_results,
    }

    def _clean(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return str(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return _clean(float(obj))
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(output_dir / 'backtest_summary.json', 'w') as fh:
        json.dump(_clean(summary), fh, indent=2, default=str)


def _save_equity_csv(curve_df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    curve_df.reset_index().to_csv(output_dir / 'equity_curve.csv', index=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_curve_df(equity_curve: list) -> pd.DataFrame:
    df = pd.DataFrame(equity_curve)
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date').sort_index()


def _compute_spy_weekly(
    spy_prices: pd.DataFrame,
    equity_dates: pd.DatetimeIndex,
) -> Optional[pd.Series]:
    """Return SPY prices indexed to equity curve dates (forward-filled for holidays)."""
    if spy_prices is None or spy_prices.empty:
        return None
    spy = spy_prices['SPY'].copy()
    spy.index = pd.to_datetime(spy.index)
    aligned = spy.reindex(equity_dates, method='ffill')
    if aligned.isna().values.all():
        return None
    return aligned.dropna()


def _annotate_bars(
    ax: plt.Axes,
    bars,
    fmt: str = '{:.3f}',
    skip_nan: bool = False,
) -> None:
    for bar in bars:
        h = bar.get_height()
        if skip_nan and (np.isnan(h) or np.isinf(h)):
            continue
        ax.annotate(
            fmt.format(h),
            xy=(bar.get_x() + bar.get_width() / 2, max(h, 0)),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center', va='bottom', fontsize=7.5,
        )


def _safe_chart(fn, *args, **kwargs) -> None:
    """Call a chart function; print a warning on failure rather than crashing."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        print(f'[VIZ] WARNING: {fn.__name__} failed — {exc}')


def _print_manifest(
    output_dir: pathlib.Path,
    model_metrics: Optional[dict],
    has_notional: bool,
    has_gbr: bool,
) -> None:
    print(f'\n[VIZ] Charts written to {output_dir}/')
    candidates = [
        '01_equity_curve.png',
        '02_drawdown.png',
        '03_rolling_sharpe.png',
        '04_return_distribution.png',
        '05_monthly_heatmap.png',
        '06_model_performance.png',
        '07_strategy_allocation.png',
        '08_notional_exposure.png',
        '09_gbr_importance.png',
        'backtest_summary.json',
        'equity_curve.csv',
    ]
    for f in candidates:
        if (output_dir / f).exists():
            print(f'[VIZ]   {f}')
        else:
            print(f'[VIZ]   {f}  ← NOT WRITTEN (skipped or failed)')
