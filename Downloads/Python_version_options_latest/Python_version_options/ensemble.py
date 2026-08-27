from AlgorithmImports import *          # resolved via local stub
from models import *
from models import (
    _calculate_bs_price_worker,
    _calculate_mmar_price_worker,
    _calculate_heston_price_worker,
    _calculate_merton_price_worker,
    _calculate_bates_price_worker,
)
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
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


class ModelPerformanceTracker:
    def __init__(self, max_history=1000):
        self._prediction_history = [] 
        self._max_history = max_history
    
    def record_prediction(self, timestamp, contract_symbol, strike, expiry, option_type, 
                         model_prices_dict, actual_price, volatility=None, moneyness=None, ttm=None):

        record = {
            'timestamp': timestamp,
            'contract_symbol': contract_symbol,
            'strike': strike,
            'expiry': expiry,
            'option_type': option_type,
            'mmar_pred': model_prices_dict.get('MMAR Call' if option_type == 'call' else 'MMAR Put'),
            'bs_pred': model_prices_dict.get('BS Call' if option_type == 'call' else 'BS Put'),
            'heston_pred': model_prices_dict.get('Heston Call' if option_type == 'call' else 'Heston Put'),
            'merton_pred': model_prices_dict.get('Merton Call' if option_type == 'call' else 'Merton Put'),
            'bates_pred': model_prices_dict.get('Bates Call' if option_type == 'call' else 'Bates Put'),
            'volatility': volatility,
            'moneyness': moneyness,
            'ttm': ttm,
            'actual_price': actual_price,
            'realized_price': None 
        }
        self._prediction_history.append(record)
    
        if len(self._prediction_history) > self._max_history:
            self._prediction_history = self._prediction_history[-self._max_history:]
    
    def record_realization(self, contract_symbol: str, realized_price: float,
                           expiry=None) -> None:
        """
        Mark the most-recent unrealized prediction for this contract as realized.
        The `expiry` parameter is accepted for backwards-compatibility but is NOT
        used for matching — the stored expiry is a datetime while callers pass a
        formatted string, so the comparison silently fails.  contract_symbol alone
        is unique enough (it encodes strike, expiry date, and right in the symbol
        string generated by _generate_option_chain).
        """
        sym = str(contract_symbol)
        for record in reversed(self._prediction_history):
            if record['contract_symbol'] == sym and record['realized_price'] is None:
                record['realized_price'] = float(realized_price)
                break

    def mark_open_predictions(self, securities_prices: dict) -> int:
        """
        Mark-to-market: fill in realized_price for every still-unrealized prediction
        using the current mid price from the securities dict.  Called weekly so the
        pricing GBR always has calibration data even if positions are never closed.
        Returns the number of records newly realized.
        """
        n_filled = 0
        for record in self._prediction_history:
            if record['realized_price'] is not None:
                continue
            price = securities_prices.get(record['contract_symbol'])
            if price and price > 0:
                record['realized_price'] = float(price)
                n_filled += 1
        return n_filled
    
    def get_calibration_dataframe(self):
        realized_records = [r for r in self._prediction_history if r['realized_price'] is not None]

        if not realized_records:
            return None

        df = pd.DataFrame(realized_records)
        return df[['mmar_pred', 'bs_pred', 'heston_pred', 'merton_pred', 'bates_pred',
                   'volatility', 'moneyness', 'ttm', 'actual_price', 'realized_price']]
    
    def get_model_accuracy_metrics(self):
        from metrics import compute_mape, compute_directional_accuracy

        df = self.get_calibration_dataframe()
        if df is None or len(df) < 10:
            return None

        metrics = {}
        model_cols = ['mmar_pred', 'bs_pred', 'heston_pred', 'merton_pred', 'bates_pred']

        for model_col in model_cols:
            model_name = model_col.replace('_pred', '').upper()
            valid_mask = (
                df[model_col].notna()
                & df['actual_price'].notna()
                & df['realized_price'].notna()
            )

            if valid_mask.sum() > 5:
                preds_arr    = df.loc[valid_mask, model_col].values.astype(float)
                entry_arr    = df.loc[valid_mask, 'actual_price'].values.astype(float)
                realized_arr = df.loc[valid_mask, 'realized_price'].values.astype(float)

                mae  = float(np.mean(np.abs(preds_arr - realized_arr)))
                rmse = float(np.sqrt(np.mean((preds_arr - realized_arr) ** 2)))
                corr = float(pd.Series(preds_arr).corr(pd.Series(realized_arr))) if len(preds_arr) > 2 else 0.0

                # MAPE: model price vs market mid-price at entry — scale-invariant
                mape = compute_mape(preds_arr, entry_arr)

                # Directional accuracy: did the model correctly identify the
                # direction of mispricing? sign(model−entry)==sign(exit−entry)
                dir_acc = compute_directional_accuracy(preds_arr, entry_arr, realized_arr)

                metrics[model_name] = {
                    'MAE':                  round(mae, 6),
                    'RMSE':                 round(rmse, 6),
                    'MAPE':                 round(mape, 2) if not np.isnan(mape) else float('nan'),
                    'Directional_Accuracy': round(dir_acc, 2) if not np.isnan(dir_acc) else float('nan'),
                    'Correlation':          round(corr, 4),
                    'Sample_Size':          int(valid_mask.sum()),
                }

        return metrics

    def run_diebold_mariano_tests(self):
        from scipy.stats import t as t_dist

        df = self.get_calibration_dataframe()
        if df is None or len(df) < 20:
            return None

        results = {}
        ref_col = 'bs_pred'
        candidates = [c for c in ('mmar_pred', 'heston_pred', 'merton_pred', 'bates_pred')
                      if c in df.columns]

        for col in candidates:
            valid = df[col].notna() & df[ref_col].notna() & df['realized_price'].notna()
            if valid.sum() < 10:
                continue

            e_cand = (df.loc[valid, col] - df.loc[valid, 'realized_price']).values
            e_ref = (df.loc[valid, ref_col] - df.loc[valid, 'realized_price']).values
            d = e_cand ** 2 - e_ref ** 2
            n = len(d)
            d_bar = d.mean()

            lag = max(1, int(n ** (1 / 3)))
            nw_var = np.var(d, ddof=0)
            for k in range(1, lag + 1):
                gamma_k = np.dot(d[:-k] - d_bar, d[k:] - d_bar) / n
                nw_var += 2 * (1 - k / (lag + 1)) * gamma_k

            if nw_var <= 0:
                continue

            dm_stat = float(d_bar / np.sqrt(nw_var / n))
            p_val = float(2 * (1 - t_dist.cdf(abs(dm_stat), df=n - 1)))
            model_name = col.replace('_pred', '').upper()
            results[f'{model_name}_vs_BS'] = {
                'DM_stat': round(dm_stat, 4),
                'p_value': round(p_val, 4),
                'n': n,
                'favors': model_name if dm_stat < 0 else 'BS',
            }

        return results if results else None


class OptionPricingCalculator:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self._price_history = None
        self._performance_tracker = ModelPerformanceTracker(max_history=1000)
   
        self._ensemble_weights = {
            'MMAR': 0.2,
            'BS': 0.2,
            'Heston': 0.2,
            'Merton': 0.2,
            'Bates': 0.2,
        }
 
        self._meta_learner = None
        self._feature_scaler = None
        self._meta_learner_trained = False
        self._last_volatility = 0.2
        self._meta_importances: dict = {}
        self._meta_train_rmse: float = float('nan')
        self.heston_params = None
        self.merton_params = None
        self._initialize_params()
    
    def get_performance_tracker(self):
        return self._performance_tracker

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
        # Calibrated values (set to None until first NLS/MLE calibration runs)
        self._bs_calibrated_sigma:    float | None = None
        self._merton_diffusion_sigma: float | None = None
        self._bates_heston_params:    dict  | None = None

    def update_price_history(self, prices_df):
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

    def calculate_model_prices(self, spot_price, strike, expiration_date, current_date, num_paths=20, hurst=None):

        if self._price_history is None or self._price_history.empty:
            return self._get_default_prices(spot_price, strike)

        time_diff = (expiration_date.date() - current_date.date()).days
        T = max(time_diff / 365.25, 0.001)

        log_rets = np.log(self._price_history['SPY'].pct_change() + 1).dropna()
        if log_rets.empty:
            return self._get_default_prices(spot_price, strike)

        volatility = float(log_rets.std() * np.sqrt(252))

        if hurst is None:
            try:
                hurst_values = calculate_hurst_for_segments(self._price_history['SPY'].values, 8)
                hurst = float(np.mean(hurst_values))
            except Exception:
                hurst = 0.5

        r = 0.052
        S0 = float(spot_price)
        K  = float(strike)

        # Use calibrated params when available; fall back to realised-vol seeds
        bs_sigma      = self._bs_calibrated_sigma    if self._bs_calibrated_sigma    is not None else volatility
        merton_sigma  = self._merton_diffusion_sigma if self._merton_diffusion_sigma is not None else volatility
        bates_hp      = self._bates_heston_params    if self._bates_heston_params    is not None else self.heston_params

        try:
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(_calculate_bs_price_worker,    S0, K, r, bs_sigma,     T, num_paths): 'BS',
                    executor.submit(_calculate_mmar_price_worker,  S0, K, r, T, hurst,         num_paths): 'MMAR',
                    executor.submit(_calculate_heston_price_worker,S0, K, r, T, self.heston_params, num_paths): 'Heston',
                    executor.submit(_calculate_merton_price_worker,S0, K, r, T, self.merton_params, num_paths, merton_sigma): 'Merton',
                    executor.submit(_calculate_bates_price_worker, S0, K, r, T, bates_hp, self.merton_params, num_paths): 'Bates',
                }
                concurrent_results = {}
                for future in as_completed(futures, timeout=30):
                    model_name, (call_p, put_p) = future.result()
                    concurrent_results[model_name] = {
                        'call': float(call_p) if not np.isnan(call_p) else None,
                        'put':  float(put_p)  if not np.isnan(put_p)  else None,
                    }
        except Exception as e:
            self.algorithm.debug(f"Concurrent pricing failed: {str(e)}")
            return self._get_default_prices(spot_price, strike)
        
        prices_dict = {}
        
        for model_name in ['MMAR', 'BS', 'Heston', 'Merton', 'Bates']:
            if model_name in concurrent_results:
                prices_dict[f'{model_name} Call'] = concurrent_results[model_name]['call']
                prices_dict[f'{model_name} Put'] = concurrent_results[model_name]['put']
            else:
                prices_dict[f'{model_name} Call'] = None
                prices_dict[f'{model_name} Put'] = None

        self._last_volatility = volatility

        BASE_MODELS = ['MMAR', 'BS', 'Heston', 'Merton', 'Bates']
        moneyness = K / S0 if S0 > 0 else 1.0

        def _gbr_or_fallback(option_type):

            price_key = 'call' if option_type == 'call' else 'put'
            available = {m: concurrent_results[m][price_key]
                         for m in BASE_MODELS
                         if m in concurrent_results and concurrent_results[m][price_key] is not None}

            if not available:
                return None

            if self._meta_learner_trained and len(available) == len(BASE_MODELS):
                try:
                    features = np.array([[
                        available['MMAR'],
                        available['BS'],
                        available['Heston'],
                        available['Merton'],
                        available['Bates'],
                        volatility,
                        moneyness,
                        T,
                    ]])
                    X_scaled = self._feature_scaler.transform(features)
                    mixed = float(self._meta_learner.predict(X_scaled)[0])
                    mixed = max(mixed, 0.0) 
                    return mixed
                except Exception:
                    pass  

      
            weight_sum = sum(self._ensemble_weights.get(m, 0) for m in available)
            if weight_sum <= 0:
                return float(np.mean(list(available.values())))
            weighted = sum(available[m] * self._ensemble_weights.get(m, 0) for m in available)
            return float(weighted / weight_sum)

        mixed_call = _gbr_or_fallback('call')
        prices_dict['Mixed Call'] = mixed_call if (mixed_call is not None and not np.isnan(mixed_call)) else None

        mixed_put = _gbr_or_fallback('put')
        prices_dict['Mixed Put'] = mixed_put if (mixed_put is not None and not np.isnan(mixed_put)) else None
        
        return prices_dict
    
    def _get_default_prices(self, spot, strike):
        default_call = max(spot - strike, 0) * 1.05
        default_put = max(strike - spot, 0) * 1.05
        
        return {
            'MMAR Call': default_call,
            'MMAR Put': default_put,
            'BS Call': default_call,
            'BS Put': default_put,
            'Heston Call': default_call,
            'Heston Put': default_put,
            'Merton Call': default_call,
            'Merton Put': default_put,
            'Bates Call': default_call,
            'Bates Put': default_put,
            'Mixed Call': default_call,
            'Mixed Put': default_put,
        }
    
    def calibrate_ensemble_weights(self, calibration_df):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler

        FEATURE_COLS = [
            'mmar_pred', 'bs_pred', 'heston_pred', 'merton_pred', 'bates_pred',
            'volatility', 'moneyness', 'ttm',
        ]

        for col in FEATURE_COLS:
            if col not in calibration_df.columns:
                calibration_df[col] = np.nan

        valid = calibration_df[FEATURE_COLS + ['realized_price']].dropna()

        n = len(valid)
        if n < 10:
            self.algorithm.debug(
                f"GBR ensemble: insufficient realized data ({n} rows, need ≥10). "
                "Keeping equal-weight fallback."
            )
            return

        X = valid[FEATURE_COLS].values
        y = valid['realized_price'].values

        # Exponential decay weights: recent observations weighted more
        raw_w    = np.exp(np.linspace(-2.0, 0.0, n))
        sample_w = (raw_w / raw_w.sum()) * n

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        depth    = 2 if n < 30 else 3
        min_leaf = max(3, n // 10)
        lr       = 0.05 if n < 30 else 0.1

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
                                  for k, v in zip(FEATURE_COLS, gbr.feature_importances_)}

        self.algorithm.debug(
            f"GBR meta-learner trained on {len(valid)} samples. "
            f"Train RMSE=${self._meta_train_rmse:.4f}  "
            f"Feature importances: {self._meta_importances}"
        )

    def calibrate_from_cross_section(self, chain_df, spot: float,
                                      r: float, log_returns: np.ndarray) -> None:
        """
        Calibrate all model parameters to the t-1 option-chain cross-section
        (and the recent return series for Merton MLE).  Results are stored on
        self and picked up by calculate_model_prices on the next call.

        BS   — NLS on t-1 cross-section (single flat vol, genuinely OOS)
        Heston — NLS on t-1 cross-section (semi-closed CF prices)
        Merton — MLE on most-recent log-returns (jump params only)
        Bates  — NLS on t-1 cross-section with Merton jump params held fixed
        """
        from calibration import (filter_chain_for_calibration, calibrate_bs_iv,
                                  calibrate_heston_nls, calibrate_merton_mle,
                                  calibrate_bates_nls)

        filt = filter_chain_for_calibration(chain_df, spot)
        if filt.empty:
            self.algorithm.debug("Calibration: no valid contracts after filtering — skipped.")
            return
        self.algorithm.debug(f"Calibration: {len(filt)} contracts after filter.")

        # --- Black-Scholes: single flat IV from t-1 cross-section ---
        self.algorithm.debug("Calibration: BS IV fitting …")
        try:
            bs_sigma = calibrate_bs_iv(filt, spot, r)
            self._bs_calibrated_sigma = bs_sigma
            self.algorithm.debug(f"  BS done: sigma={bs_sigma:.4f}")
        except Exception as e:
            self.algorithm.debug(f"  BS IV calibration failed: {e}")
            bs_sigma = self._bs_calibrated_sigma or 0.20

        # --- Merton MLE: jump params from recent return history ---
        self.algorithm.debug(f"Calibration: Merton MLE ({len(log_returns)} returns) …")
        merton_result = None
        if len(log_returns) >= 30:
            try:
                merton_result = calibrate_merton_mle(log_returns)
            except Exception as e:
                self.algorithm.debug(f"  Merton MLE failed: {e}")
        if merton_result is not None:
            self.merton_params = {
                'lambda_jump': merton_result['lambda_jump'],
                'mu_jump':     merton_result['mu_jump'],
                'sigma_jump':  merton_result['sigma_jump'],
            }
            self._merton_diffusion_sigma = merton_result['diffusion_sigma']
            self.algorithm.debug(
                f"  Merton done: lambda={merton_result['lambda_jump']:.4f}  "
                f"mu_j={merton_result['mu_jump']:.4f}  "
                f"sigma_j={merton_result['sigma_jump']:.4f}")
        else:
            self.algorithm.debug("  Merton MLE skipped (insufficient returns or failed).")

        # --- Heston NLS: stochastic-vol params from t-1 cross-section ---
        self.algorithm.debug("Calibration: Heston NLS …")
        try:
            heston_result = calibrate_heston_nls(filt, spot, r,
                                                  init_params=self.heston_params)
            if heston_result is not None:
                self.heston_params = heston_result
                self.algorithm.debug(
                    f"  Heston done: v0={heston_result['v0']:.4f}  "
                    f"kappa={heston_result['kappa']:.4f}  "
                    f"rho={heston_result['rho']:.4f}")
            else:
                self.algorithm.debug("  Heston NLS did not converge — prior params retained.")
        except Exception as e:
            self.algorithm.debug(f"  Heston NLS failed: {e}")

        # --- Bates NLS: Heston params with Merton jump params fixed ---
        self.algorithm.debug("Calibration: Bates NLS …")
        try:
            bates_result = calibrate_bates_nls(filt, spot, r,
                                                self.merton_params,
                                                init_heston=self.heston_params)
            if bates_result is not None:
                self._bates_heston_params = bates_result
                self.algorithm.debug(
                    f"  Bates done: v0={bates_result['v0']:.4f}  "
                    f"kappa={bates_result['kappa']:.4f}")
            else:
                self.algorithm.debug("  Bates NLS did not converge — prior params retained.")
        except Exception as e:
            self.algorithm.debug(f"  Bates NLS failed: {e}")

        self.algorithm.debug(
            f"Calibration complete — "
            f"BS={bs_sigma:.4f}  "
            f"Heston_v0={self.heston_params['v0']:.4f}  "
            f"Merton_lambda={self.merton_params['lambda_jump']:.4f}"
        )