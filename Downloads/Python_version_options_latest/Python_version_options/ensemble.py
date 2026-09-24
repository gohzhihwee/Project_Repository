from AlgorithmImports import *          # resolved via local stub
from models import *
import math
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
def block_bootstrap_sharpe(returns, block_size=5, n_bootstrap=100, ann_factor=52):
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n < block_size * 2:
        return None

    std = returns.std()
    observed_sharpe = float(returns.mean() / std * np.sqrt(ann_factor)) if std > 0 else 0.0

    rng = np.random.default_rng(42)
    n_blocks = int(np.ceil(n / block_size))
    bootstrap_sharpes = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([returns[s:s + block_size] for s in starts])[:n]
        s_std = sample.std()
        bootstrap_sharpes[i] = float(sample.mean() / s_std * np.sqrt(ann_factor)) if s_std > 0 else 0.0

    return {
        'observed_sharpe': round(observed_sharpe, 3),
        'ci_95_lower': round(float(np.percentile(bootstrap_sharpes, 2.5)), 3),
        'ci_95_upper': round(float(np.percentile(bootstrap_sharpes, 97.5)), 3),
        'p_value_vs_zero': round(float(np.mean(bootstrap_sharpes <= 0)), 4),
    }


class _MockBacktester:
    def __init__(self, real_algo):
        self.portfolio = {}              # empty → bool(portfolio) is False
        self._underlying_symbol = real_algo._underlying_symbol
        self._real = real_algo

    def history(self, *args, **kwargs):
        return self._real.history(*args, **kwargs)


def seed_strategy_history() -> list:
    """
    Generate synthetic bootstrap rows encoding economic priors about which strategy
    suits which market regime.  These act as a weak prior that real data quickly
    overwrites via the rolling window and exponential decay weights in train().
    """
    regimes = [
        # (vol,  mom,    ma_dev,  hurst)  → dominant regime
        (0.32,  0.04,   0.03,   0.57),   # high-vol, trending up
        (0.32, -0.04,  -0.03,   0.55),   # high-vol, trending down
        (0.28,  0.01,   0.00,   0.50),   # high-vol, directionless
        (0.14,  0.02,   0.015,  0.53),   # low-vol, gentle uptrend
        (0.14, -0.01,  -0.015,  0.53),   # low-vol, gentle downtrend
        (0.20,  0.00,   0.025,  0.42),   # mean-reverting, price above MA
        (0.20,  0.00,  -0.025,  0.42),   # mean-reverting, price below MA
        (0.22,  0.03,   0.01,   0.58),   # trending (high Hurst), moderate vol
    ]

    seeds = []
    for vol, mom, ma_dev, hurst in regimes:
        high_vol  = vol > 0.25
        trending  = abs(mom) > 0.015 and hurst > 0.52
        reverting = abs(ma_dev) > 0.018 and hurst < 0.48
        up        = mom > 0

        row: dict = {
            'volatility': vol, 'momentum_5d': mom,
            'ma_deviation_20': ma_dev, 'hurst': hurst,
            'BuyAndHold':   0.003 if up else -0.002,
            'Momentum':     0.006 if (trending and up) else (-0.004 if trending else 0.000),
            'MeanReversion':0.005 if reverting else -0.002,
            'MMAR':         0.003 if hurst > 0.54 or hurst < 0.46 else 0.001,
            'BS':           0.002 if not high_vol else -0.001,
            'Heston':       0.003 if vol > 0.20 else 0.001,
            'Merton':       0.004 if high_vol else 0.001,
            'Bates':        0.005 if high_vol else 0.001,
            'Mixed':        0.002,  # always mediocre but never worst
        }
        seeds.append(row)
    return seeds


class StrategySelector:
    STRATEGY_NAMES = [
        'BuyAndHold', 'Momentum', 'MeanReversion',
        'MMAR', 'BS', 'Heston', 'Merton', 'Bates', 'Mixed',
    ]
    FEATURE_COLS = ['volatility', 'momentum_5d', 'ma_deviation_20', 'hurst']
    MIN_SAMPLES  = 8   # first real training after ~8 weeks of live data

    # Rolling window cap: keep at most this many rows so old regimes don't dominate
    MAX_HISTORY  = 52

    def __init__(self):
        self._multi_gbr          = None
        self._scaler             = None
        self.is_trained          = False
        self.feature_importances: dict = {}

    def train(self, history_rows, algorithm):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.preprocessing import StandardScaler

        if len(history_rows) < self.MIN_SAMPLES:
            return

        # Rolling window: cap at MAX_HISTORY most-recent observations
        recent = history_rows[-self.MAX_HISTORY:]
        df     = pd.DataFrame(recent)
        all_cols = self.FEATURE_COLS + self.STRATEGY_NAMES
        for c in all_cols:
            if c not in df.columns:
                df[c] = 0.0

        valid = df[all_cols].dropna()
        if len(valid) < self.MIN_SAMPLES:
            return

        n = len(valid)
        X = valid[self.FEATURE_COLS].values
        Y = valid[self.STRATEGY_NAMES].values

        # Exponential decay weights: e^-2 for oldest row, e^0=1 for newest
        raw_w        = np.exp(np.linspace(-2.0, 0.0, n))
        sample_w     = (raw_w / raw_w.sum()) * n   # scaled so sum ≈ n

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Regularise more aggressively when the live-data window is small
        depth      = 2 if n < 20 else 3
        min_leaf   = max(3, n // 8)
        lr         = 0.05 if n < 20 else 0.08

        multi_gbr = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=100,
                max_depth=depth,
                learning_rate=lr,
                subsample=0.8,
                min_samples_leaf=min_leaf,
                random_state=42,
            ),
            n_jobs=1,
        )
        multi_gbr.fit(X_scaled, Y, sample_weight=sample_w)

        self._multi_gbr = multi_gbr
        self._scaler    = scaler
        self.is_trained = True

        all_fi   = np.array([est.feature_importances_ for est in multi_gbr.estimators_])
        mean_fi  = all_fi.mean(axis=0)
        self.feature_importances = {c: round(float(v), 4)
                                    for c, v in zip(self.FEATURE_COLS, mean_fi)}

        pred_means = multi_gbr.predict(X_scaled).mean(axis=0)
        summary    = {name: round(float(v), 5)
                      for name, v in zip(self.STRATEGY_NAMES, pred_means)}
        algorithm.debug(
            f"StrategySelector GBR trained on {n} rows (window={self.MAX_HISTORY}). "
            f"Importances: {self.feature_importances}  Predicted returns: {summary}"
        )

    def select(self, features_dict) -> str:
        if not self.is_trained:
            # Regime-based heuristic during the cold-start window
            return self._regime_based_fallback(features_dict)
        try:
            X        = np.array([[features_dict.get(c, 0.0) for c in self.FEATURE_COLS]])
            X_scaled = self._scaler.transform(X)
            pred     = self._multi_gbr.predict(X_scaled)[0]
            return self.STRATEGY_NAMES[int(np.argmax(pred))]
        except Exception:
            return self._regime_based_fallback(features_dict)

    def _regime_based_fallback(self, features_dict) -> str:
        """Rule-based strategy selector used before the GBR has enough training data."""
        vol    = features_dict.get('volatility', 0.20)
        mom    = features_dict.get('momentum_5d', 0.0)
        ma_dev = features_dict.get('ma_deviation_20', 0.0)
        hurst  = features_dict.get('hurst', 0.5)

        if vol > 0.28:
            # Volatile, jump-prone environment: Bates (stochastic vol + jumps)
            return 'Bates'
        if hurst > 0.54 and abs(mom) > 0.012:
            # Strongly trending + persistent: ride momentum via options
            return 'Momentum'
        if hurst < 0.46 or abs(ma_dev) > 0.020:
            # Mean-reverting market: exploit deviations from MA
            return 'MeanReversion'
        if vol < 0.15:
            # Low vol, smooth dynamics: BS model is adequate
            return 'BS'
        return 'Mixed'



# ---------------------------------------------------------------------------
# Model registry: display name → tracker column and prepared-options label
# ---------------------------------------------------------------------------

MODEL_COLUMNS = {
    'BS':          'bs_pred',
    'MMAR':        'mmar_pred',
    'MMAR_STATIC': 'mmar_static_pred',
    'HESTON':      'heston_pred',
    'MERTON':      'merton_pred',
    'MERTON_Q':    'merton_q_pred',
    'BATES':       'bates_pred',
    'MIXED':       'mixed_pred',
    # Closed-form prices from the same fitted parameters (robustness: scores
    # the models free of Monte Carlo noise). Compared against BS_CF.
    'BS_CF':       'bs_cf_pred',
    'HESTON_CF':   'heston_cf_pred',
    'MERTON_CF':   'merton_cf_pred',
    'MERTON_Q_CF': 'merton_q_cf_pred',
    'BATES_CF':    'bates_cf_pred',
}
PRICE_LABELS = {
    'BS': 'BS', 'MMAR': 'MMAR', 'MMAR_STATIC': 'MMAR Static', 'HESTON': 'Heston',
    'MERTON': 'Merton', 'MERTON_Q': 'Merton Q', 'BATES': 'Bates', 'MIXED': 'Mixed',
    'BS_CF': 'BS CF', 'HESTON_CF': 'Heston CF', 'MERTON_CF': 'Merton CF',
    'MERTON_Q_CF': 'Merton Q CF', 'BATES_CF': 'Bates CF',
}
CLOSED_FORM_MODELS = [m for m in MODEL_COLUMNS if m.endswith('_CF')]


def _reference_for(model: str) -> str:
    """DM reference: MC models vs MC Black-Scholes, closed forms vs closed-form BS."""
    return 'BS_CF' if model.endswith('_CF') else 'BS'
# Base models fed to the GBR pricing ensemble (paper §4.12).
ENSEMBLE_BASE_MODELS = ['MMAR', 'BS', 'HESTON', 'MERTON', 'BATES']
GBR_FEATURE_COLS = ['mmar_pred', 'bs_pred', 'heston_pred', 'merton_pred', 'bates_pred',
                    'volatility', 'moneyness', 'ttm']
# The GBR trains on at most this many most-recent prior-date predictions.
GBR_TRAINING_WINDOW = 1000

# Calibration sample: up to this many contracts per (K/S band × maturity) cell
# of the t−1 chain (10 cells → ≤ 400 contracts), shared by every quote-
# calibrated model.
CALIB_CONTRACTS_PER_CELL = 40
# Cascade draws used inside the MMAR NLS objective (pricing uses N_MC_PATHS).
MMAR_CALIB_DRAWS = 2_000
# Timeout (seconds) for one date's batch of model pricing jobs.
PRICING_TIMEOUT_S = 600


def _dm_test(d: np.ndarray, dates: np.ndarray) -> Optional[dict]:
    """
    Diebold–Mariano test of equal squared-error loss, d = e_cand² − e_ref².

    Primary (date-clustered): d is averaged within each rebalance date, and the
    test runs on that series of T date means — Newey–West HAC variance
    (Bartlett kernel, bandwidth max(1, ⌊T^{1/3}⌋)), Harvey–Leybourne–Newbold
    small-sample correction for h = 1, referred to t(T−1). This respects the
    common-date dependence of contracts priced from one calibration.

    Reference (contract-level): the same statistic on the pooled contract
    series, as reported in earlier drafts.
    """
    from scipy.stats import t as t_dist

    def _nw_stat(x: np.ndarray) -> tuple:
        n = len(x)
        x_bar = float(x.mean())
        lag = max(1, int(n ** (1 / 3)))
        nw_var = float(np.var(x, ddof=0))
        for k in range(1, min(lag, n - 1) + 1):
            gamma_k = float(np.dot(x[:-k] - x_bar, x[k:] - x_bar) / n)
            nw_var += 2.0 * (1.0 - k / (lag + 1)) * gamma_k
        if nw_var <= 0:
            return float('nan'), float('nan')
        return x_bar / math.sqrt(nw_var / n), nw_var

    d = np.asarray(d, dtype=float)
    if len(d) < 10:
        return None
    dm_c, _ = _nw_stat(d)
    p_c = float(2 * (1 - t_dist.cdf(abs(dm_c), df=len(d) - 1))) if np.isfinite(dm_c) else float('nan')

    by_date = pd.Series(d).groupby(pd.Series(np.asarray(dates))).mean().sort_index().values
    T = len(by_date)
    if T >= 5:
        dm_raw, _ = _nw_stat(by_date)
        dm = dm_raw * math.sqrt((T - 1) / T)
        p = float(2 * (1 - t_dist.cdf(abs(dm), df=T - 1))) if np.isfinite(dm) else float('nan')
    else:
        dm, p = float('nan'), float('nan')
    return {
        'DM_stat':          round(float(dm), 4),
        'p_value':          round(p, 4),
        'n_dates':          T,
        'n':                int(len(d)),
        'DM_stat_contract': round(float(dm_c), 4),
        'p_value_contract': round(p_c, 4),
        'favors':           None if not np.isfinite(dm) else ('candidate' if dm < 0 else 'reference'),
    }


class ModelPerformanceTracker:
    """
    Stores every OOS prediction (no cap) and scores it two ways:
      same-day  (primary):   model price vs the day-t market mid it was priced
                             against (`actual_price`) — the OOS pricing error;
      next-week (secondary): model price vs the mid of the same contract on the
                             next rebalance date (`realized_price`) — a one-week
                             forecast error. Filled only from that date's quoted
                             chain; contracts not quoted then (e.g. expired) stay
                             unrealised and are excluded rather than back-filled.
    """

    def __init__(self, max_history: Optional[int] = None):
        self._prediction_history = []
        self._max_history = max_history

    def record_prediction(self, timestamp, contract_symbol, strike, expiry, option_type,
                          model_prices_dict, actual_price, volatility=None, moneyness=None,
                          ttm=None, spot=None, dividend_yield=None):

        _call_put = 'Call' if option_type == 'call' else 'Put'
        record = {
            'timestamp': timestamp,
            'contract_symbol': contract_symbol,
            'strike': strike,
            'expiry': expiry,
            'option_type': option_type,
            'volatility': volatility,
            'moneyness': moneyness,
            'ttm': ttm,
            'spot': spot,
            'dividend_yield': dividend_yield,
            'actual_price': actual_price,
            'realized_price': None,
        }
        for model, col in MODEL_COLUMNS.items():
            record[col] = model_prices_dict.get(f'{PRICE_LABELS[model]} {_call_put}')
        self._prediction_history.append(record)

        if self._max_history is not None and len(self._prediction_history) > self._max_history:
            self._prediction_history = self._prediction_history[-self._max_history:]

    def record_realization(self, contract_symbol: str, realized_price: float,
                           expiry=None) -> None:
        """
        Mark the most-recent unrealized prediction for this contract as realized.
        The `expiry` parameter is accepted for backwards-compatibility but is NOT
        used for matching — contract_symbol alone is unique (it encodes strike,
        expiry date, and right).
        """
        sym = str(contract_symbol)
        for record in reversed(self._prediction_history):
            if record['contract_symbol'] == sym and record['realized_price'] is None:
                record['realized_price'] = float(realized_price)
                break

    def mark_open_predictions(self, day_mids: dict, from_date=None) -> int:
        """
        Fill realized_price for unrealized predictions from `day_mids`, a
        {contract_symbol: mid} dict of ONE day's quoted chain. With `from_date`
        given, only predictions made on that rebalance date are filled, so every
        realization is exactly one rebalance interval ahead.
        Returns the number of records newly realized.
        """
        n_filled = 0
        for record in self._prediction_history:
            if record['realized_price'] is not None:
                continue
            if from_date is not None and _as_date(record['timestamp']) != from_date:
                continue
            price = day_mids.get(record['contract_symbol'])
            if price and price > 0:
                record['realized_price'] = float(price)
                n_filled += 1
        return n_filled

    # ---- data access ------------------------------------------------------

    def get_prediction_dataframe(self) -> Optional[pd.DataFrame]:
        if not self._prediction_history:
            return None
        df = pd.DataFrame(self._prediction_history)
        df['date'] = df['timestamp'].map(_as_date)
        return df

    def get_calibration_dataframe(self):
        """Realised (next-week) records only, in the legacy column layout."""
        realized_records = [r for r in self._prediction_history if r['realized_price'] is not None]
        if not realized_records:
            return None
        df = pd.DataFrame(realized_records)
        return df[list(MODEL_COLUMNS.values())
                  + ['volatility', 'moneyness', 'ttm', 'actual_price', 'realized_price']]

    def _available_models(self, df: pd.DataFrame) -> list:
        return [m for m, c in MODEL_COLUMNS.items() if c in df.columns and df[c].notna().any()]

    # ---- metrics ----------------------------------------------------------

    def get_model_accuracy_metrics(self):
        """
        Same-day metrics under the legacy keys (RMSE, MAE, MAPE, Correlation,
        Sample_Size) plus next-week metrics (RMSE_next, MAE_next,
        Correlation_next, Directional_Accuracy, Sample_Size_next).
        """
        from metrics import compute_mape, compute_directional_accuracy

        df = self.get_prediction_dataframe()
        if df is None or len(df) < 10:
            return None

        metrics = {}
        for model in self._available_models(df):
            col = MODEL_COLUMNS[model]
            sd = df[df[col].notna() & df['actual_price'].notna()]
            if len(sd) <= 5:
                continue
            pred = sd[col].values.astype(float)
            mid = sd['actual_price'].values.astype(float)
            err = pred - mid
            ape = np.abs(err) / np.abs(mid)
            out = {
                'RMSE':        round(float(np.sqrt(np.mean(err ** 2))), 6),
                'MAE':         round(float(np.mean(np.abs(err))), 6),
                'MAPE':        round(compute_mape(pred, mid), 2),
                'MedAPE':      round(float(np.median(ape) * 100.0), 2),
                'Correlation': round(float(np.corrcoef(pred, mid)[0, 1]), 4),
                'Sample_Size': int(len(sd)),
                'n_dates':     int(sd['date'].nunique()),
            }
            nw = sd[sd['realized_price'].notna()]
            if len(nw) > 5:
                p_n = nw[col].values.astype(float)
                e_n = nw['actual_price'].values.astype(float)
                r_n = nw['realized_price'].values.astype(float)
                out.update({
                    'RMSE_next':            round(float(np.sqrt(np.mean((p_n - r_n) ** 2))), 6),
                    'MAE_next':             round(float(np.mean(np.abs(p_n - r_n))), 6),
                    'Correlation_next':     round(float(np.corrcoef(p_n, r_n)[0, 1]), 4),
                    'Directional_Accuracy': round(compute_directional_accuracy(p_n, e_n, r_n), 2),
                    'Sample_Size_next':     int(len(nw)),
                })
            metrics[model] = out
        return metrics

    def _target(self, df: pd.DataFrame, target: str) -> str:
        return 'actual_price' if target == 'same_day' else 'realized_price'

    def run_diebold_mariano_tests(self, target: str = 'same_day'):
        df = self.get_prediction_dataframe()
        if df is None or len(df) < 20:
            return None
        tcol = self._target(df, target)
        results = {}
        for model in self._available_models(df):
            ref = _reference_for(model)
            if model == ref:
                continue
            col, ref_col = MODEL_COLUMNS[model], MODEL_COLUMNS[ref]
            v = df[df[col].notna() & df[ref_col].notna() & df[tcol].notna()]
            d = (v[col] - v[tcol]).values ** 2 - (v[ref_col] - v[tcol]).values ** 2
            res = _dm_test(d, v['date'].values)
            if res is None:
                continue
            res['favors'] = None if res['favors'] is None else (model if res['favors'] == 'candidate' else ref)
            res['target'] = target
            results[f'{model}_vs_{ref}'] = res
        return results or None

    def run_pairwise_dm_tests(self, target: str = 'same_day') -> Optional[pd.DataFrame]:
        """
        Diebold–Mariano test for every pair of models (A, B), on the contracts
        both priced: d = e_A² − e_B², date-clustered primary statistic plus the
        pooled contract-level reference. DM < 0 favours model A.
        """
        df = self.get_prediction_dataframe()
        if df is None or len(df) < 20:
            return None
        tcol = self._target(df, target)
        models = self._available_models(df)
        rows = []
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                ca, cb = MODEL_COLUMNS[a], MODEL_COLUMNS[b]
                v = df[df[ca].notna() & df[cb].notna() & df[tcol].notna()]
                ea, eb = (v[ca] - v[tcol]).values, (v[cb] - v[tcol]).values
                res = _dm_test(ea ** 2 - eb ** 2, v['date'].values)
                if res is None:
                    continue
                rows.append({
                    'model_a': a, 'model_b': b, 'target': target,
                    'rmse_a': round(float(np.sqrt(np.mean(ea ** 2))), 6),
                    'rmse_b': round(float(np.sqrt(np.mean(eb ** 2))), 6),
                    'DM_stat': res['DM_stat'], 'p_value': res['p_value'], 'n_dates': res['n_dates'],
                    'DM_stat_contract': res['DM_stat_contract'],
                    'p_value_contract': res['p_value_contract'], 'n': res['n'],
                    'favors': None if res['favors'] is None else (a if res['favors'] == 'candidate' else b),
                })
        return pd.DataFrame(rows) if rows else None

    def run_mmar_calibration_protocol_dm_test(self, target: str = 'same_day') -> Optional[dict]:
        """
        DM test of the quote-calibrated MMAR (σ, H, s fitted weekly) against the
        static-H MMAR (H fixed at the returns-based partition-function estimate,
        σ and s still fitted). A rejection in favour of the calibrated model
        means the H measured in returns is not the H that prices options.
        """
        df = self.get_prediction_dataframe()
        if df is None or len(df) < 20:
            return None
        tcol = self._target(df, target)
        c_cal, c_st = MODEL_COLUMNS['MMAR'], MODEL_COLUMNS['MMAR_STATIC']
        v = df[df[c_cal].notna() & df[c_st].notna() & df[tcol].notna()]
        d = (v[c_cal] - v[tcol]).values ** 2 - (v[c_st] - v[tcol]).values ** 2
        res = _dm_test(d, v['date'].values)
        if res is None:
            return None
        res['favors'] = None if res['favors'] is None else (
            'MMAR_calibrated' if res['favors'] == 'candidate' else 'MMAR_static')
        res['target'] = target
        return res

    def compute_bucketed_metrics(self) -> Optional[pd.DataFrame]:
        """
        Per (K/S band × maturity bucket × model): same-day and next-week RMSE,
        MAE, median APE, n, dates, date-clustered DM vs BS, and the sign shares
        behind the directional-accuracy composition benchmark.

        K/S bands follow the evaluation stratification (calibration.EVAL_MONEYNESS_EDGES);
        they are strike/spot bands, so e.g. K/S < 1 is ITM for calls, OTM for puts.
        Maturity buckets: short 7–30 and medium 31–90 calendar days.
        """
        from calibration import EVAL_MONEYNESS_EDGES, MATURITY_BUCKETS_DAYS

        df = self.get_prediction_dataframe()
        if df is None or len(df) < 10:
            return None
        edges = list(EVAL_MONEYNESS_EDGES)
        band_labels = [f'{lo:.2f}-{hi:.2f}' for lo, hi in zip(edges[:-1], edges[1:])]
        df['_band'] = pd.cut(df['moneyness'], bins=edges, labels=band_labels, include_lowest=True)
        days = np.rint(df['ttm'].astype(float) * 365.25)
        mat_names = ['short', 'medium']
        df['_mat'] = np.select([(days >= lo) & (days <= hi) for lo, hi in MATURITY_BUCKETS_DAYS],
                               mat_names, default='')
        df = df[df['_band'].notna() & (df['_mat'] != '')]

        rows = []
        for band in ['all'] + band_labels:
            for mat in ['all'] + mat_names:
                cell = df
                if band != 'all':
                    cell = cell[cell['_band'] == band]
                if mat != 'all':
                    cell = cell[cell['_mat'] == mat]
                if len(cell) < 5:
                    continue
                for model in self._available_models(cell):
                    col = MODEL_COLUMNS[model]
                    ref = _reference_for(model)
                    ref_col = MODEL_COLUMNS[ref]
                    row = {'model': model, 'ks_band': band, 'maturity_bucket': mat, 'dm_reference': ref}
                    for target, tcol, suffix in (('same_day', 'actual_price', ''),
                                                 ('next_week', 'realized_price', '_next')):
                        v = cell[cell[col].notna() & cell[ref_col].notna() & cell[tcol].notna()]
                        row[f'n{suffix}'] = int(len(v))
                        if len(v) < 5:
                            continue
                        e = v[col].values - v[tcol].values
                        row[f'n_dates{suffix}'] = int(v['date'].nunique())
                        row[f'rmse{suffix}'] = round(float(np.sqrt(np.mean(e ** 2))), 6)
                        row[f'mae{suffix}'] = round(float(np.mean(np.abs(e))), 6)
                        row[f'median_ape_pct{suffix}'] = round(
                            float(np.median(np.abs(e) / v['actual_price'].values) * 100.0), 2)
                        if model != ref:
                            d = e ** 2 - (v[ref_col].values - v[tcol].values) ** 2
                            dm = _dm_test(d, v['date'].values)
                            if dm is not None:
                                row[f'dm_stat_vs_bs{suffix}'] = dm['DM_stat']
                                row[f'dm_p_vs_bs{suffix}'] = dm['p_value']
                                row[f'dm_stat_contract_vs_bs{suffix}'] = dm['DM_stat_contract']
                        if target == 'next_week':
                            row['pos_signal_share'] = round(float(np.mean(v[col].values > v['actual_price'].values)), 4)
                            row['neg_move_share'] = round(float(np.mean(v['realized_price'].values < v['actual_price'].values)), 4)
                    rows.append(row)
        return pd.DataFrame(rows) if rows else None


def _as_date(ts):
    return ts.date() if hasattr(ts, 'date') and callable(ts.date) else ts


class OptionPricingCalculator:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self._price_history = None
        self._performance_tracker = ModelPerformanceTracker()

        self._ensemble_weights = {m: 1.0 / len(ENSEMBLE_BASE_MODELS) for m in ENSEMBLE_BASE_MODELS}

        self._meta_learner = None
        self._feature_scaler = None
        self._meta_learner_trained = False
        self._last_volatility = 0.2
        self._meta_importances: dict  = {}
        self._meta_train_rmse: float  = float('nan')
        self._meta_oos_rmse:   float  = float('nan')
        self._meta_oos_mae:    float  = float('nan')
        self._meta_oos_n:      int    = 0
        self.heston_params = None
        self.merton_params = None
        self._initialize_params()

    def get_performance_tracker(self):
        return self._performance_tracker

    def get_mmar_calibration_history_df(self) -> Optional[pd.DataFrame]:
        if not self._mmar_calib_history:
            return None
        return pd.DataFrame(self._mmar_calib_history)

    def get_calibration_log_df(self) -> Optional[pd.DataFrame]:
        return pd.DataFrame(self._calib_log) if self._calib_log else None

    def get_param_log_df(self) -> Optional[pd.DataFrame]:
        """Fitted parameters of every model on every calibration date (one row per date × model)."""
        return pd.DataFrame(self._param_log) if self._param_log else None

    def get_mc_check_log_df(self) -> Optional[pd.DataFrame]:
        return pd.DataFrame(self._mc_check_log) if self._mc_check_log else None

    def get_mmar_calibration_summary(self) -> Optional[dict]:
        """Aggregate convergence diagnostics across all weekly MMAR NLS calibrations."""
        hist = self._mmar_calib_history
        if not hist:
            return None
        n_dates = len(hist)
        n_converged = sum(1 for row in hist if row.get('converged'))
        hursts = [row['hurst'] for row in hist if row.get('hurst') is not None]
        h_ret = [row['hurst_returns'] for row in hist if row.get('hurst_returns') is not None]
        return {
            'n_dates':             n_dates,
            'n_converged':         n_converged,
            'cascade_s_at_bound':        int(sum(1 for row in hist if row.get('sigma_at_bound'))),
            'static_cascade_s_at_bound': int(sum(1 for row in hist if row.get('static_s_at_bound'))),
            'converged_fraction':  round(n_converged / n_dates, 4) if n_dates else None,
            'hurst_median':        round(float(np.median(hursts)), 4) if hursts else None,
            'hurst_std':           round(float(np.std(hursts, ddof=1)), 4) if len(hursts) > 1 else None,
            'hurst_returns_median': round(float(np.nanmedian(h_ret)), 4) if h_ret else None,
        }

    def _initialize_params(self):
        self.heston_params = {
            'v0': 0.04,
            'kappa': 1.5,
            'theta': 0.04,
            'sigma': 0.2,
            'rho': -0.5
        }
        self.merton_params = {
            'lambda_jump': 0.1,
            'mu_jump': -0.1,
            'sigma_jump': 0.1
        }
        # Calibrated values (None until the first calibration runs)
        self._bs_calibrated_sigma:    float | None = None
        self._merton_diffusion_sigma: float | None = None
        self._merton_q_params:        dict  | None = None
        self._bates_heston_params:    dict  | None = None
        self._mmar_params:            dict  | None = None
        self._mmar_static_params:     dict  | None = None
        self._hurst_returns:          float | None = None
        self._mmar_calib_history:     list = []
        self._calib_log:              list = []
        self._param_log:              list = []
        self._mc_check_log:           list = []

    def update_price_history(self, prices_df):
        """`prices_df['SPY']` must be the dividend-adjusted (total-return) close."""
        self._price_history = prices_df
        if not prices_df.empty:
            log_rets = np.log(prices_df['SPY'].pct_change() + 1).dropna()
            if not log_rets.empty:
                realized_vol = float(log_rets.std() * np.sqrt(252))
                self._last_volatility = realized_vol
                # Only seed from realised vol before NLS calibration has run
                if self._bs_calibrated_sigma is None:
                    self.heston_params['v0']    = realized_vol ** 2
                    self.heston_params['theta'] = realized_vol ** 2

    def update_ensemble_weights(self, weights_dict):

        if abs(sum(weights_dict.values()) - 1.0) < 1e-6:
            self._ensemble_weights = weights_dict
        else:
            self.algorithm.debug(f"Warning: Ensemble weights do not sum to 1.0: {weights_dict}")

    # ---- current parameter sets ------------------------------------------

    def _bs_sigma(self) -> float:
        return self._bs_calibrated_sigma if self._bs_calibrated_sigma is not None else self._last_volatility

    def _merton_mc_params(self) -> dict:
        sigma = self._merton_diffusion_sigma if self._merton_diffusion_sigma is not None else self._last_volatility
        return {'sigma': sigma, **self.merton_params}

    def _mmar_param_sets(self) -> dict:
        bs = self._bs_sigma()
        h_ret = self._hurst_returns if self._hurst_returns is not None else 0.5
        cal = self._mmar_params or {'sigma': bs, 'hurst': 0.5, 'cascade_sigma': 0.0}
        static = self._mmar_static_params or {'sigma': bs, 'hurst': h_ret, 'cascade_sigma': 0.0}
        return {'MMAR': cal, 'MMAR_STATIC': static}

    # ---- pricing ---------------------------------------------------------

    def calculate_model_prices_batch(self, universe_df: pd.DataFrame, spot: float,
                                     r: float, q: float, current_date) -> pd.DataFrame:
        """
        OOS prices for every contract in `universe_df` (columns strike, ttm,
        option_type) under every model, from the parameters fitted on t−1.

        BS / Merton / Merton-Q / Heston / Bates: N_MC_PATHS-path Monte Carlo,
        one vectorised simulation per model for the whole date (common random
        numbers across contracts), seeded per (model, date). MMAR: conditional
        Monte Carlo over N_MC_PATHS cascade draws. Each MC model is also priced
        in closed form and the |MC − closed form| gap is logged as a
        convergence check. The per-model jobs run in a ThreadPoolExecutor with
        an explicit timeout.

        Returns a DataFrame aligned with universe_df, one column per model name.
        """
        from calibration import (bs_price_vec, merton_price_vec, heston_price_vec,
                                 bates_price_vec)

        K = universe_df['strike'].values.astype(float)
        T = universe_df['ttm'].values.astype(float)
        is_call = (universe_df['option_type'].values == 'call')
        date_key = int(current_date.strftime('%Y%m%d'))

        def _seed(i: int) -> np.random.SeedSequence:
            return np.random.SeedSequence([RNG_BASE_SEED, date_key, i])

        bs_sigma = self._bs_sigma()
        merton = self._merton_mc_params()
        bates = {**(self._bates_heston_params or self.heston_params), **self.merton_params}
        mc_jobs = {
            'BS':     ('bs', {'sigma': bs_sigma}),
            'MERTON': ('merton', merton),
            'HESTON': ('heston', self.heston_params),
            'BATES':  ('bates', bates),
        }
        if self._merton_q_params is not None:
            mq = self._merton_q_params
            mc_jobs['MERTON_Q'] = ('merton', {'sigma': mq['diffusion_sigma'], 'lambda_jump': mq['lambda_jump'],
                                              'mu_jump': mq['mu_jump'], 'sigma_jump': mq['sigma_jump']})

        closed_form = {
            'BS':     lambda p: bs_price_vec(spot, K, T, r, q, p['sigma'] * np.sqrt(T), is_call),
            'MERTON': lambda p: merton_price_vec(spot, K, T, r, q, p['sigma'], p['lambda_jump'],
                                                 p['mu_jump'], p['sigma_jump'], is_call),
            'HESTON': lambda p: heston_price_vec(spot, K, T, r, q, p['v0'], p['kappa'], p['theta'],
                                                 p['sigma'], p['rho'], is_call),
            'BATES':  lambda p: bates_price_vec(spot, K, T, r, q, p['v0'], p['kappa'], p['theta'],
                                                p['sigma'], p['rho'], p['lambda_jump'], p['mu_jump'],
                                                p['sigma_jump'], is_call),
        }
        closed_form['MERTON_Q'] = closed_form['MERTON']

        def _mc_job(name: str, i: int):
            kind, params = mc_jobs[name]
            t0 = time.perf_counter()
            prices, se = mc_price_contracts(kind, params, spot, r, q, K, T, is_call, N_MC_PATHS, _seed(i))
            elapsed = time.perf_counter() - t0
            cf = closed_form[name](params)
            return name, prices, cf, {'date': str(current_date.date()), 'model': name,
                                  'seconds': round(elapsed, 3),
                                  'median_abs_mc_minus_cf': round(float(np.median(np.abs(prices - cf))), 5),
                                  'max_abs_mc_minus_cf': round(float(np.max(np.abs(prices - cf))), 5),
                                  'median_mc_se': round(float(np.median(se)), 5),
                                  'max_abs_z': round(float(np.max(np.abs(prices - cf) / np.maximum(se, 1e-4))), 2)}

        def _mmar_job(name: str, i: int):
            p = self._mmar_param_sets()[name]
            t0 = time.perf_counter()
            z = draw_cascade_normals(N_MC_PATHS, np.random.default_rng(_seed(i)))
            prices = mmar_price_vec(spot, K, T, r, q, p['sigma'], p['hurst'], p['cascade_sigma'], is_call, z)
            return name, prices, None, {'date': str(current_date.date()), 'model': name,
                                        'seconds': round(time.perf_counter() - t0, 3)}

        out = pd.DataFrame(index=universe_df.index)
        with ThreadPoolExecutor(max_workers=len(mc_jobs) + 2) as executor:
            futures = [executor.submit(_mc_job, name, i) for i, name in enumerate(mc_jobs)]
            futures += [executor.submit(_mmar_job, name, 10 + i)
                        for i, name in enumerate(['MMAR', 'MMAR_STATIC'])]
            for future in as_completed(futures, timeout=PRICING_TIMEOUT_S):
                name, prices, cf, log_row = future.result()
                out[name] = prices
                if cf is not None:
                    out[f'{name}_CF'] = cf
                self._mc_check_log.append(log_row)
        for name in MODEL_COLUMNS:
            if name not in out.columns:
                out[name] = np.nan

        check = [row for row in self._mc_check_log[-len(futures):] if 'max_abs_mc_minus_cf' in row]
        self.algorithm.debug("MC pricing (N=%d): " % N_MC_PATHS + "  ".join(
            f"{row['model']} {row['seconds']:.2f}s |MC-CF| med {row['median_abs_mc_minus_cf']:.4f} "
            f"max {row['max_abs_mc_minus_cf']:.4f}" for row in check))

        out['MIXED'] = self._mixed_prices(out, universe_df, T, spot)
        return out

    def _mixed_prices(self, prices: pd.DataFrame, universe_df: pd.DataFrame,
                      T: np.ndarray, spot: float) -> np.ndarray:
        base = prices[ENSEMBLE_BASE_MODELS]
        if self._meta_learner_trained and base.notna().all(axis=None):
            try:
                features = np.column_stack([
                    base['MMAR'], base['BS'], base['HESTON'], base['MERTON'], base['BATES'],
                    np.full(len(T), self._last_volatility),
                    universe_df['strike'].values / spot,
                    T,
                ])
                return np.maximum(self._meta_learner.predict(self._feature_scaler.transform(features)), 0.0)
            except Exception as e:
                self.algorithm.debug(f"GBR ensemble prediction failed, using equal weights: {e}")
        w = np.array([self._ensemble_weights[m] for m in ENSEMBLE_BASE_MODELS])
        return (base.values * w).sum(axis=1) / w.sum()

    # ---- GBR pricing ensemble -------------------------------------------

    def calibrate_ensemble_weights(self, calibration_df):
        """
        Train the GBR pricing ensemble on predictions from PRIOR rebalance dates:
        features = the five base-model prices + realized vol, moneyness, TTM;
        target = the same-day market mid those prices were scored against
        (`actual_price`). Trained before the current date's pricing, so the
        MIXED price is out-of-sample under the same same-day scoring.
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler

        for col in GBR_FEATURE_COLS:
            if col not in calibration_df.columns:
                calibration_df[col] = np.nan

        valid = calibration_df[GBR_FEATURE_COLS + ['actual_price']].dropna().tail(GBR_TRAINING_WINDOW)

        n = len(valid)
        if n < 10:
            self.algorithm.debug(
                f"GBR ensemble: insufficient data ({n} rows, need ≥10). "
                "Keeping equal-weight fallback."
            )
            return

        X = valid[GBR_FEATURE_COLS].values
        y = valid['actual_price'].values

        # Exponential decay weights: recent observations weighted more
        raw_w    = np.exp(np.linspace(-2.0, 0.0, n))
        sample_w = (raw_w / raw_w.sum()) * n

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        depth    = 2 if n < 30 else 3
        min_leaf = max(3, n // 10)
        lr       = 0.05 if n < 30 else 0.1

        # ── OOS evaluation: temporal 70/30 split (shadow model, does not affect production GBR) ──
        oos_rmse = float('nan')
        oos_mae  = float('nan')
        n_test   = 0
        if n >= 20:
            n_train  = int(n * 0.70)
            n_test   = n - n_train
            X_tr, X_te = X_scaled[:n_train], X_scaled[n_train:]
            y_tr, y_te = y[:n_train], y[n_train:]
            raw_w_tr   = np.exp(np.linspace(-2.0, 0.0, n_train))
            sw_tr      = (raw_w_tr / raw_w_tr.sum()) * n_train
            d_tr       = 2 if n_train < 30 else 3
            ml_tr      = max(3, n_train // 10)
            lr_tr      = 0.05 if n_train < 30 else 0.1
            gbr_shadow = GradientBoostingRegressor(
                n_estimators=100, max_depth=d_tr, learning_rate=lr_tr,
                subsample=0.8, min_samples_leaf=ml_tr, random_state=42,
            )
            gbr_shadow.fit(X_tr, y_tr, sample_weight=sw_tr)
            oos_preds = gbr_shadow.predict(X_te)
            oos_rmse  = float(np.sqrt(np.mean((oos_preds - y_te) ** 2)))
            oos_mae   = float(np.mean(np.abs(oos_preds - y_te)))

        self._meta_oos_rmse = oos_rmse
        self._meta_oos_mae  = oos_mae
        self._meta_oos_n    = n_test

        # ── Production GBR: trained on full dataset ──────────────────────────
        gbr = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=depth,
            learning_rate=lr,
            subsample=0.8,
            min_samples_leaf=min_leaf,
            random_state=42,
        )
        gbr.fit(X_scaled, y, sample_weight=sample_w)

        self._meta_learner = gbr
        self._feature_scaler = scaler
        self._meta_learner_trained = True

        train_preds = gbr.predict(X_scaled)
        self._meta_train_rmse = float(np.sqrt(np.mean((train_preds - y) ** 2)))
        self._meta_importances = {k: round(float(v), 4)
                                  for k, v in zip(GBR_FEATURE_COLS, gbr.feature_importances_)}

        self.algorithm.debug(
            f"GBR meta-learner trained on {n} samples (OOS holdout: {n_test}). "
            f"Train RMSE=${self._meta_train_rmse:.4f}  "
            f"OOS RMSE=${oos_rmse:.4f}  OOS MAE=${oos_mae:.4f}  "
            f"Feature importances: {self._meta_importances}"
        )

    # ---- weekly calibration ----------------------------------------------

    def calibrate_from_cross_section(self, chain_df, spot: float,
                                      r: float, log_returns: np.ndarray,
                                      q: float = 0.0, calib_date=None) -> None:
        """
        Calibrate every model on information through t−1 and store the results
        for calculate_model_prices_batch.

        Quote-calibrated (common stratified t−1 sample, ≤ 10 × CALIB_CONTRACTS_PER_CELL
        contracts, squared-price-deviation objective):
          BS (σ), Heston (v0, κ, θ, ξ, ρ), Bates (Heston part; jumps fixed from
          Merton MLE), Merton-Q (σ, λ, μ_J, σ_J), MMAR (σ, H, s),
          MMAR static-H (σ, s; H = returns partition-function estimate).
        Return-calibrated: Merton (MLE on dividend-adjusted daily log returns).
        """
        from calibration import (filter_chain_for_calibration, stratified_sample,
                                  CALIB_MONEYNESS_EDGES, calibrate_bs_iv,
                                  calibrate_heston_nls, calibrate_merton_mle,
                                  calibrate_merton_nls, calibrate_bates_nls,
                                  calibrate_mmar_nls, bs_price_vec, merton_price_vec,
                                  heston_price_vec, bates_price_vec)

        log = self.algorithm.debug
        date_str = str(calib_date) if calib_date is not None else str(self.algorithm._current_time.date())
        date_key = int(date_str.replace('-', ''))
        filt = filter_chain_for_calibration(chain_df, spot)
        sample = stratified_sample(filt, spot, CALIB_CONTRACTS_PER_CELL, seed=date_key,
                                   moneyness_edges=CALIB_MONEYNESS_EDGES)
        sample = sample[(sample['ttm'] > 1 / 365) & (sample['mid_price'] > 0.05)].reset_index(drop=True)
        if len(sample) < 5:
            self.algorithm.debug("Calibration: too few valid contracts after filtering — skipped.")
            return
        self.algorithm.debug(f"Calibration: {len(filt)} contracts after filter, "
                             f"{len(sample)} in the stratified calibration sample (q={q:.4f}).")
        K = sample['strike'].values.astype(float)
        T = sample['ttm'].values.astype(float)
        mids = sample['mid_price'].values.astype(float)
        is_call = sample['option_type'].values == 'call'

        def _log_fit(model: str, prices: np.ndarray, seconds: float) -> None:
            err = prices - mids
            row = {'date': date_str, 'model': model, 'n': len(mids),
                   'rmse': round(float(np.sqrt(np.mean(err ** 2))), 4),
                   'mae': round(float(np.mean(np.abs(err))), 4), 'seconds': round(seconds, 2)}
            self._calib_log.append(row)
            self.algorithm.debug(f"  {model:11s} in-sample RMSE ${row['rmse']:.3f}  "
                                 f"MAE ${row['mae']:.3f}  ({seconds:.1f}s)")

        # --- Black-Scholes -------------------------------------------------
        t0 = time.perf_counter()
        log('Calibration step 1/7: Black-Scholes sigma')
        self._bs_calibrated_sigma = calibrate_bs_iv(sample, spot, r, q, log=log)
        _log_fit('BS', bs_price_vec(spot, K, T, r, q, self._bs_calibrated_sigma * np.sqrt(T), is_call),
                 time.perf_counter() - t0)

        # --- Merton MLE on returns ------------------------------------------
        t0 = time.perf_counter()
        log('Calibration step 2/7: Merton MLE on returns')
        merton_result = calibrate_merton_mle(log_returns, log=log) if len(log_returns) >= 30 else None
        if merton_result is not None:
            self.merton_params = {k: merton_result[k] for k in ('lambda_jump', 'mu_jump', 'sigma_jump')}
            self._merton_diffusion_sigma = merton_result['diffusion_sigma']
        mp = self._merton_mc_params()
        _log_fit('MERTON', merton_price_vec(spot, K, T, r, q, mp['sigma'], mp['lambda_jump'],
                                            mp['mu_jump'], mp['sigma_jump'], is_call),
                 time.perf_counter() - t0)

        # --- Merton on quotes (robustness) -----------------------------------
        t0 = time.perf_counter()
        init_mq = self._merton_q_params or merton_result
        log('Calibration step 3/7: Merton on quotes (robustness)')
        mq = calibrate_merton_nls(sample, spot, r, q, init_params=init_mq, log=log)
        if mq is not None:
            self._merton_q_params = mq
        if self._merton_q_params is not None:
            p = self._merton_q_params
            _log_fit('MERTON_Q', merton_price_vec(spot, K, T, r, q, p['diffusion_sigma'], p['lambda_jump'],
                                                  p['mu_jump'], p['sigma_jump'], is_call),
                     time.perf_counter() - t0)

        # --- Heston ---------------------------------------------------------
        t0 = time.perf_counter()
        log('Calibration step 4/7: Heston')
        heston_result = calibrate_heston_nls(sample, spot, r, init_params=self.heston_params, q=q, log=log)
        if heston_result is not None:
            self.heston_params = heston_result
        else:
            self.algorithm.debug("  Heston NLS did not converge — prior params retained.")
        h = self.heston_params
        _log_fit('HESTON', heston_price_vec(spot, K, T, r, q, h['v0'], h['kappa'], h['theta'],
                                            h['sigma'], h['rho'], is_call), time.perf_counter() - t0)

        # --- Bates (Heston part fitted, jumps from Merton MLE) ----------------
        log('Calibration step 5/7: Bates (jumps fixed from Merton MLE)')
        t0 = time.perf_counter()
        bates_result = calibrate_bates_nls(sample, spot, r, self.merton_params,
                                           init_heston=self._bates_heston_params or self.heston_params, q=q,
                                           log=log)
        if bates_result is not None:
            self._bates_heston_params = bates_result
        else:
            self.algorithm.debug("  Bates NLS did not converge — prior params retained.")
        b = self._bates_heston_params or self.heston_params
        m = self.merton_params
        _log_fit('BATES', bates_price_vec(spot, K, T, r, q, b['v0'], b['kappa'], b['theta'], b['sigma'],
                                          b['rho'], m['lambda_jump'], m['mu_jump'], m['sigma_jump'], is_call),
                 time.perf_counter() - t0)

        # --- MMAR: returns-based H, then quote calibration (free and static H) ---
        if len(log_returns) >= 100:
            h_est = estimate_hurst_partition(log_returns)['hurst']
            if np.isfinite(h_est):
                self._hurst_returns = float(h_est)
            log(f"Returns-based H (partition function, {len(log_returns)} returns): {h_est:.4f}")
        h_ret = self._hurst_returns if self._hurst_returns is not None else 0.5
        z_cal = draw_cascade_normals(MMAR_CALIB_DRAWS,
                                     np.random.default_rng([RNG_BASE_SEED, date_key, 99]))

        t0 = time.perf_counter()
        log('Calibration step 6/7: MMAR (sigma, H, s free)')
        mmar_result = calibrate_mmar_nls(sample, spot, r, z_cal, q=q, init_params=self._mmar_params,
                                         hurst_starts=(h_ret, 0.5), bs_sigma=self._bs_calibrated_sigma,
                                         log=log)
        t_mmar = time.perf_counter() - t0
        if mmar_result is not None:
            self._mmar_params = {k: mmar_result[k] for k in ('sigma', 'hurst', 'cascade_sigma')}
        else:
            self.algorithm.debug("  MMAR NLS did not converge — prior params retained.")

        t0 = time.perf_counter()
        log(f'Calibration step 7/7: MMAR static-H (H fixed at {h_ret:.4f})')
        static_result = calibrate_mmar_nls(sample, spot, r, z_cal, q=q, init_params=self._mmar_static_params,
                                           bs_sigma=self._bs_calibrated_sigma, fixed_hurst=h_ret, log=log)
        t_static = time.perf_counter() - t0
        if static_result is not None:
            self._mmar_static_params = {k: static_result[k] for k in ('sigma', 'hurst', 'cascade_sigma')}

        sets = self._mmar_param_sets()
        for name, secs in (('MMAR', t_mmar), ('MMAR_STATIC', t_static)):
            p = sets[name]
            _log_fit(name, mmar_price_vec(spot, K, T, r, q, p['sigma'], p['hurst'], p['cascade_sigma'],
                                          is_call, z_cal), secs)

        row = {'date': date_str, 'hurst_returns': h_ret}
        if mmar_result is not None:
            row.update(mmar_result)
        else:
            row.update({'converged': False})
        if static_result is not None:
            row.update({'static_sigma': static_result['sigma'],
                        'static_cascade_sigma': static_result['cascade_sigma'],
                        'static_objective': static_result['objective'],
                        'static_converged': True,
                        'static_s_at_bound': static_result['sigma_at_bound']})
        self._mmar_calib_history.append(row)
        cal = sets['MMAR']
        base = {'date': date_str, 'spot': spot, 'r': r, 'q': q}
        param_sets = {
            'BS':          {'sigma': self._bs_sigma()},
            'MERTON':      self._merton_mc_params(),
            'MERTON_Q':    self._merton_q_params or {},
            'HESTON':      self.heston_params,
            'BATES':       {**(self._bates_heston_params or self.heston_params), **self.merton_params},
            'MMAR':        sets['MMAR'],
            'MMAR_STATIC': sets['MMAR_STATIC'],
        }
        for model, params in param_sets.items():
            self._param_log.append({**base, 'model': model, **params})
        self.algorithm.debug(
            f"Calibration complete — BS σ={self._bs_sigma():.4f}  Heston v0={self.heston_params['v0']:.4f}  "
            f"Merton λ={self.merton_params['lambda_jump']:.3f}  "
            f"MMAR σ={cal['sigma']:.4f} H={cal['hurst']:.4f} s={cal['cascade_sigma']:.4f}  "
            f"H_returns={h_ret:.4f}")
