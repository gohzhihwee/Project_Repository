from AlgorithmImports import *          
from models import *
from modelstrats import *
from convmethods import *
from ensemble import *
from data_loader import DatabentoCacheLoader, _parse_osi_symbol
from calibration import stratified_sample, EVAL_MONEYNESS_EDGES

import os
import pathlib
from dotenv import load_dotenv
load_dotenv()          
import numpy as np
import pandas as pd
import datetime
import time
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor

OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'
CACHE_DIR  = pathlib.Path(__file__).parent / 'option_cache'
CSV_DIR    = pathlib.Path(__file__).parent / 'options_data'  # drop Databento CSV exports here

UNDERLYING_TICKER = 'SPY'
# Prevailing short-dated Treasury yield over the sample (paper §3.2).
RISK_FREE_RATE = 0.052
# Evaluation universe per rebalance: this many contracts from each of the
# 5 K/S bands × 2 maturity buckets (7–30, 31–90 days) → 150 contracts.
EVAL_CONTRACTS_PER_CELL = 15
EVAL_MIN_DTE, EVAL_MAX_DTE = 7, 90

# ---------------------------------------------------------------------------
# Shim data-structures that mirror the QC runtime API
# ---------------------------------------------------------------------------

class _PortfolioPosition:
    def __init__(self, qty, avg_price):
        self.quantity   = qty
        self.average_price = avg_price
    @property
    def invested(self):
        return self.quantity != 0

class _EmptyPosition:
    quantity      = 0
    average_price = 0.0
    invested      = False

class Portfolio:
    
    def __init__(self, cash):
        self._cash      = cash
        self._positions = {}   # str(symbol) -> _PortfolioPosition
        self._prices    = {}   # str(symbol) -> float (current mark price)

    # --- QC API ---
    @property
    def total_portfolio_value(self):
        opt_val = sum(
            p.quantity * self._prices.get(sym, 0.0) * 100
            for sym, p in self._positions.items()
        )
        return self._cash + opt_val

    @property
    def total_margin_used(self):
        return sum(
            abs(p.quantity) * self._prices.get(sym, 0.0) * 100
            for sym, p in self._positions.items()
        )

    @property
    def margin_remaining(self):
        return max(self._cash - self.total_margin_used * 0.3, 0.0)

    def __getitem__(self, symbol):
        return self._positions.get(str(symbol), _EmptyPosition())

    def __contains__(self, symbol):
        return str(symbol) in self._positions

    def keys(self):
        return self._positions.keys()

    def __bool__(self):
        return len(self._positions) > 0

    # --- Internal order helpers ---
    def _buy(self, sym_str, qty, price):
        cost = price * qty * 100
        if cost > self._cash:
            qty = int(self._cash / (price * 100 + 1e-9))
        if qty <= 0:
            return 0
        self._cash -= price * qty * 100
        if sym_str in self._positions:
            pos   = self._positions[sym_str]
            total = pos.quantity + qty
            if total == 0:
                del self._positions[sym_str]
            elif abs(total) > 0:
                avg = (pos.average_price * abs(pos.quantity) + price * qty) / abs(total)
                self._positions[sym_str] = _PortfolioPosition(total, avg)
        else:
            self._positions[sym_str] = _PortfolioPosition(qty, price)
        self._prices[sym_str] = price
        return qty

    def _sell(self, sym_str, qty, price):
        if sym_str not in self._positions:
            return 0
        pos      = self._positions[sym_str]
        sell_qty = min(abs(qty), pos.quantity)
        self._cash += price * sell_qty * 100
        new_qty = pos.quantity - sell_qty
        if new_qty == 0:
            del self._positions[sym_str]
        else:
            self._positions[sym_str] = _PortfolioPosition(new_qty, pos.average_price)
        return sell_qty

    def _write(self, sym_str, qty, price):
        """Open a short option position (write). Receives premium upfront."""
        if qty <= 0:
            return 0
        self._cash += price * qty * 100
        if sym_str in self._positions:
            pos       = self._positions[sym_str]
            new_qty   = pos.quantity - qty
            total_abs = abs(pos.quantity) + qty
            avg       = (pos.average_price * abs(pos.quantity) + price * qty) / total_abs
            self._positions[sym_str] = _PortfolioPosition(new_qty, avg)
        else:
            self._positions[sym_str] = _PortfolioPosition(-qty, price)
        self._prices[sym_str] = price
        return qty

    def _cover(self, sym_str, qty, price):
        """Close a short option position (cover). Pays buyback cost."""
        if sym_str not in self._positions:
            return 0
        pos = self._positions[sym_str]
        if pos.quantity >= 0:
            return 0
        cover_qty = min(qty, abs(pos.quantity))
        self._cash -= price * cover_qty * 100
        new_qty = pos.quantity + cover_qty
        if new_qty == 0:
            del self._positions[sym_str]
        else:
            self._positions[sym_str] = _PortfolioPosition(new_qty, pos.average_price)
        return cover_qty


class _SecurityData:
    def __init__(self, price):
        self.price = price

class Securities:
    def __init__(self):
        self._data = {}

    def __getitem__(self, symbol):
        return _SecurityData(self._data.get(str(symbol), 0.0))

    def __contains__(self, symbol):
        return str(symbol) in self._data

    def keys(self):
        return self._data.keys()

    def update(self, d):
        self._data.update({str(k): float(v) for k, v in d.items()})


class _OptionContract:
    
    def __init__(self, symbol, strike, expiry_dt, right, bid, ask, last):
        self.symbol        = symbol
        self.strike        = float(strike)
        self.expiry        = expiry_dt          # datetime.datetime
        self.right         = right              # OptionRight.CALL / .PUT
        self.bid_price     = float(bid)
        self.ask_price     = float(ask)
        self.last_price    = float(last)
        self.volume        = 100
        self.open_interest = 500

class _OptionChain:
    def __init__(self, contracts):
        self._contracts = contracts
    def __iter__(self):
        return iter(self._contracts)
    def __len__(self):
        return len(self._contracts)

class _OptionChains:
    def __init__(self):
        self._data = {}
    def get(self, symbol):
        return self._data.get(str(symbol))
    def _set(self, symbol, chain):
        self._data[str(symbol)] = chain

class CurrentSlice:
    def __init__(self):
        self.option_chains = _OptionChains()


# ---------------------------------------------------------------------------
# Mock backtester used for hypothetical signal generation in _rebalance
# ---------------------------------------------------------------------------

class _MockBacktester:


    def __init__(self, algo):
        self._algo = algo
        self.portfolio = {}          
        self._underlying_symbol = algo._underlying_symbol

    def history(self, symbol, periods, resolution=None):
        return self._algo.history(symbol, periods, resolution)


# ---------------------------------------------------------------------------
# Main algorithm class
# ---------------------------------------------------------------------------

class OptionsArbitrageAlgorithm:

    # ==== entry point ======================================================

    def run_backtest(self):
        self._wall_t0 = time.perf_counter()
        print("Downloading SPY daily data via yfinance …")
        import yfinance as yf
        
        # Two series: the UNADJUSTED close is the tradable spot used for option
        # pricing, moneyness and settlement; the dividend-ADJUSTED close is used
        # only for return-based quantities (realized vol, Merton MLE, returns-H,
        # regime features, the SPY total-return benchmark). Adjusted closes are
        # back-adjusted for every dividend up to the download date and were
        # 2.5–3.9% below the traded price over this sample.
        raw = yf.download(UNDERLYING_TICKER, start="2021-06-01", end="2024-09-01",
                          progress=False, auto_adjust=False)
        raw.index = pd.to_datetime(raw.index).normalize()
        close = raw['Close'].squeeze()
        adj_close = raw['Adj Close'].squeeze()
        self._all_prices = pd.DataFrame({'SPY': close, 'SPY_TR': adj_close})
        divs = yf.Ticker(UNDERLYING_TICKER).dividends
        divs.index = pd.to_datetime(divs.index).tz_localize(None).normalize()
        self._dividends = divs
        print(f"  {len(self._all_prices)} daily bars loaded; {len(divs)} dividend records.")
        using_real_data = bool(os.environ.get('DATABENTO_API_KEY', '').strip())
        print(f"  Options data source: {'Databento (real)' if using_real_data else 'synthetic (fallback — set DATABENTO_API_KEY for real data)'}")

        self._start_date = datetime.date(2023, 8, 25)
        self._end_date   = datetime.date(2024, 8, 16)
        
        self._warmup_cutoff = self._start_date - datetime.timedelta(days=30)

        self.initialize()

        trading_dates = [
            d.date() for d in self._all_prices.index
            if self._warmup_cutoff <= d.date() <= self._end_date
        ]
        n_days = len(trading_dates)

        print(f"\n{'Date':<14} {'Portfolio $':>14} {'Active Strategy':>22}")
        print("─" * 55)

        self._equity_curve = []

        for day_idx, sim_date in enumerate(trading_dates):
            self._current_time = datetime.datetime.combine(
                sim_date, datetime.time(10, 30))

           
            row = self._all_prices[self._all_prices.index.date == sim_date]  # type: ignore
            if row.empty:
                continue
            spy_price = float(row['SPY'].iloc[0])
            self._securities.update({'SPY': spy_price,
                                     self._underlying_symbol: spy_price})
            self._settle_expired_positions(sim_date, spy_price)
            if day_idx % 10 == 0:
                print(f"[{sim_date}] Day {day_idx + 1}/{n_days}  "
                      f"SPY ${spy_price:.2f}  "
                      f"PV ${self._portfolio.total_portfolio_value:,.0f}")

            
            chain, calib_df = self._load_option_chain(spy_price, sim_date)
            self._current_slice.option_chains._set(self._option_symbol, chain)
            self._today_calib_df = calib_df
            # _prev_calib_df intentionally NOT updated yet: it still holds
            # yesterday's cross-section so _calibrate_model_params (called
            # inside _rebalance) calibrates on t-1 and evaluates OOS on t.

            is_rebalance_day = (sim_date.weekday() == 0
                                and sim_date >= self._start_date)
            held_symbols = set(self._portfolio._positions.keys())
            for c in chain:
                sym = str(c.symbol)
                # On non-rebalance days update only held positions so margin
                # monitoring uses fresh prices without iterating 6k+ contracts.
                if not is_rebalance_day and sym not in held_symbols:
                    continue
                mid = (c.bid_price + c.ask_price) / 2.0
                self._securities.update({sym: mid})
                self._portfolio._prices[sym] = mid

            # Daily scheduled event
            self._monitor_margin_health()

            # Weekly rebalance on Mondays after warmup
            if sim_date.weekday() == 0 and sim_date >= self._start_date:
                self._rebalance()
                pv = self._portfolio.total_portfolio_value
                strat_name = type(self._active_strategy).__name__
                self._equity_curve.append({
                    'date':              sim_date,
                    'value':             pv,
                    'notional_exposure': self._portfolio.total_margin_used,
                    'strategy':          strat_name,
                })
                print(f"{str(sim_date):<14} {pv:>14,.2f} {strat_name:>22}")
                # Live interim chart — overwrite every 4 rebalances (~monthly)
                if len(self._equity_curve) % 4 == 0:
                    from visualization import save_interim_chart
                    save_interim_chart(self._equity_curve, OUTPUT_DIR)

            # Update t-1 cache AFTER rebalance so tomorrow's calibration
            # uses today's cross-section (genuine out-of-sample protocol).
            self._prev_calib_df = calib_df
            self._prev_spot     = spy_price
            self._prev_date     = sim_date

        self._print_summary()
        self._generate_output()

    # ==== synthetic option chain ==========================================

    def _generate_option_chain(self, spot, current_date):

        hist = self._all_prices[self._all_prices.index.date <= current_date]  # type: ignore
        log_rets = np.log(hist['SPY_TR'].pct_change() + 1).dropna()
        sigma = float(log_rets.tail(20).std() * np.sqrt(252)) if len(log_rets) >= 5 else 0.20
        sigma = max(sigma, 0.05)
        r = RISK_FREE_RATE

        atm      = round(spot / 5.0) * 5.0
        strikes  = [atm + i * 5.0 for i in range(-2, 3)]
        expiries = [
            current_date + datetime.timedelta(days=14),
            current_date + datetime.timedelta(days=21),
        ]


        VRP_FACTOR  = 1.20
        SKEW_SLOPE  = -0.25  

        contracts = []
        for expiry in expiries:
            T = max((expiry - current_date).days / 365.25, 0.001)
            for K in strikes:
                log_moneyness  = np.log(K / spot)
                skew_adj       = SKEW_SLOPE * log_moneyness * sigma
                implied_sigma  = max(sigma * VRP_FACTOR + skew_adj, 0.05)
                for right in [OptionRight.CALL, OptionRight.PUT]:
                    sq  = implied_sigma * np.sqrt(T)
                    d1  = (np.log(spot / K) + (r + 0.5 * implied_sigma ** 2) * T) / sq
                    d2  = d1 - sq
                    if right == OptionRight.CALL:
                        mid = spot * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                    else:
                        mid = K * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
                    mid    = max(float(mid), 0.01)
                    spread = max(mid * 0.02, 0.05)
                    bid    = round(mid - spread / 2, 2)
                    ask    = round(mid + spread / 2, 2)
                    tag    = 'C' if right == OptionRight.CALL else 'P'
                    sym    = (f"SPY {expiry.strftime('%y%m%d')} "
                              f"{tag}{int(K):05d}")
                    expiry_dt = datetime.datetime.combine(
                        expiry, datetime.time(16, 0))
                    contracts.append(
                        _OptionContract(sym, K, expiry_dt, right, bid, ask, mid))
        return _OptionChain(contracts)

    # ==== summary ==========================================================

    def _print_summary(self):
        if not self._equity_curve:
            print("\nNo equity curve data — no rebalances executed.")
            return
        curve   = pd.DataFrame(self._equity_curve).set_index('date')
        ret     = (curve['value'].iloc[-1] / 100_000 - 1) * 100
        wrets   = curve['value'].pct_change().dropna()
        sharpe  = (wrets.mean() / wrets.std() * np.sqrt(52)
                   if wrets.std() > 0 else 0.0)
        mdd     = ((curve['value'] - curve['value'].cummax())
                   / curve['value'].cummax()).min() * 100
        print("\n" + "═" * 55)
        print(f"  Total Return : {ret:+.2f}%")
        print(f"  Sharpe Ratio : {sharpe:.3f}  (annualised, weekly)")
        print(f"  Max Drawdown : {mdd:.2f}%")
        print(f"  Final Value  : ${curve['value'].iloc[-1]:,.2f}")
        print("═" * 55)

    def _generate_output(self):
        
        if not self._equity_curve:
            print("[VIZ] No equity curve — skipping output generation.")
            return

        from metrics import compute_portfolio_metrics
        from visualization import generate_visualizations

        curve_df = pd.DataFrame(self._equity_curve)
        curve_df['date'] = pd.to_datetime(curve_df['date'])
        curve_df = curve_df.set_index('date')

        # Weekly portfolio returns for all reported risk/return metrics (Sharpe,
        # Sortino, VaR/CVaR, alpha/beta, rolling Sharpe, the return-distribution
        # chart) — derived from the same post-trade equity curve that CAGR,
        # total_return, max drawdown and the equity-curve/drawdown charts use,
        # so every reported number traces back to one consistent series.
        # self._all_weekly_returns is a *different*, pre-trade series (mark-to-
        # market of the prior week's book, measured before that week's own
        # trades execute) used causally during the walk-forward loop for
        # regime classification (_regime_returns) — that's a legitimate,
        # separate purpose and is intentionally left untouched here.
        equity_weekly_rets = curve_df['value'].pct_change().dropna().tolist()

        # SPY weekly returns aligned to rebalance dates for alpha/beta
        spy_weekly_rets = None
        try:
            ec_dates = pd.to_datetime([r['date'] for r in self._equity_curve])
            spy_at_dates = self._all_prices['SPY_TR'].reindex(ec_dates, method='ffill').dropna()
            spy_weekly_rets = spy_at_dates.pct_change().dropna()
        except Exception as e:
            self.debug(f"SPY benchmark alignment failed: {e}")

        pm = compute_portfolio_metrics(
            equity_curve=curve_df,
            weekly_returns=equity_weekly_rets,
            trade_log=self._trade_log,
            spy_returns=spy_weekly_rets,
            initial_capital=100_000.0,
            risk_free_rate=RISK_FREE_RATE,
        )
        self.debug(f"Portfolio metrics: {pm}")

        tracker      = self._price_calculator.get_performance_tracker()
        model_metrics = tracker.get_model_accuracy_metrics()
        dm_results    = tracker.run_diebold_mariano_tests('same_day')
        dm_next       = tracker.run_diebold_mariano_tests('next_week')

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        bucketed_df = tracker.compute_bucketed_metrics()
        if bucketed_df is not None:
            out_path = OUTPUT_DIR / 'bucketed_model_errors.csv'
            bucketed_df.to_csv(out_path, index=False)
            self.debug(f"[OUTPUT] Bucketed metrics written → {out_path}  ({len(bucketed_df)} rows)")
        else:
            self.debug("[OUTPUT] Bucketed metrics: insufficient data, CSV not written.")

        pred_df = tracker.get_prediction_dataframe()
        if pred_df is not None:
            pred_df.to_csv(OUTPUT_DIR / 'predictions.csv', index=False)
        if model_metrics:
            metrics_df = pd.DataFrame(model_metrics).T
            metrics_df.index.name = 'model'
            same_cols = ['RMSE', 'MAE', 'MAPE', 'MedAPE', 'Correlation', 'Sample_Size', 'n_dates']
            next_cols = ['RMSE_next', 'MAE_next', 'Correlation_next', 'Directional_Accuracy', 'Sample_Size_next']
            metrics_df[[c for c in same_cols if c in metrics_df]].join(
                pd.DataFrame(dm_results or {}).T.rename(index=lambda k: k.split('_vs_')[0]).add_prefix('dm_'), how='left'
            ).to_csv(OUTPUT_DIR / 'model_errors_sameday.csv')
            metrics_df[[c for c in next_cols if c in metrics_df]].join(
                pd.DataFrame(dm_next or {}).T.rename(index=lambda k: k.split('_vs_')[0]).add_prefix('dm_'), how='left'
            ).to_csv(OUTPUT_DIR / 'model_errors_nextweek.csv')

        pc = self._price_calculator
        mmar_hist_df = pc.get_mmar_calibration_history_df()
        if mmar_hist_df is not None:
            mmar_hist_df.to_csv(OUTPUT_DIR / 'mmar_calibration_history.csv', index=False)
            self.debug(f"[OUTPUT] MMAR calibration history written ({len(mmar_hist_df)} dates)")
        for name, frame in (('calibration_log.csv', pc.get_calibration_log_df()),
                            ('calibration_params.csv', pc.get_param_log_df()),
                            ('mc_convergence_log.csv', pc.get_mc_check_log_df()),
                            ('dm_pairwise_sameday.csv', tracker.run_pairwise_dm_tests('same_day')),
                            ('dm_pairwise_nextweek.csv', tracker.run_pairwise_dm_tests('next_week'))):
            if frame is not None:
                frame.to_csv(OUTPUT_DIR / name, index=False)

        # MMAR calibration-protocol diagnostics: convergence count and the DM
        # test of the quote-calibrated MMAR against the static-H MMAR.
        mmar_calib_summary = pc.get_mmar_calibration_summary()
        mmar_protocol_dm   = tracker.run_mmar_calibration_protocol_dm_test('same_day')
        mmar_protocol_dm_next = tracker.run_mmar_calibration_protocol_dm_test('next_week')
        event_counts = self._event_counts(pred_df)
        # Figure 4 and the headline JSON keys show the Monte Carlo (reported)
        # prices; the closed-form robustness rows are stored separately.
        mc_metrics = {k: v for k, v in (model_metrics or {}).items() if k not in CLOSED_FORM_MODELS} or None
        cf_metrics = {k: v for k, v in (model_metrics or {}).items() if k in CLOSED_FORM_MODELS} or None
        mc_dm = {k: v for k, v in (dm_results or {}).items() if not k.endswith('_vs_BS_CF')} or None
        cf_dm = {k: v for k, v in (dm_results or {}).items() if k.endswith('_vs_BS_CF')} or None
        self.debug(f"[OUTPUT] Event counts: {event_counts}")
        if mmar_calib_summary is not None:
            self.debug(
                f"[MMAR] Calibration convergence: "
                f"{mmar_calib_summary['n_converged']} of {mmar_calib_summary['n_dates']} dates converged  "
                f"(H median={mmar_calib_summary['hurst_median']}, SD={mmar_calib_summary['hurst_std']})"
            )
        if mmar_protocol_dm is not None:
            self.debug(
                f"[MMAR] Quote-calibrated vs static-H DM test: "
                f"stat={mmar_protocol_dm['DM_stat']}  p={mmar_protocol_dm['p_value']}  "
                f"favors={mmar_protocol_dm['favors']}  n={mmar_protocol_dm['n']}"
            )
        ss = self._strategy_selector

        _nan = float('nan')
        generate_visualizations(
            equity_curve=self._equity_curve,
            weekly_returns=equity_weekly_rets,
            spy_prices=self._all_prices[['SPY_TR']].rename(columns={'SPY_TR': 'SPY'}),
            model_accuracy_metrics=mc_metrics,
            dm_results=mc_dm,
            trade_log=self._trade_log,
            gbr_meta_importances=pc._meta_importances if pc._meta_importances else None,
            gbr_meta_train_rmse=pc._meta_train_rmse if pc._meta_train_rmse == pc._meta_train_rmse else None,
            strategy_selector_importances=ss.feature_importances if ss.feature_importances else None,
            portfolio_metrics=pm,
            gbr_oos_rmse=pc._meta_oos_rmse if pc._meta_oos_rmse == pc._meta_oos_rmse else None,
            gbr_oos_mae=pc._meta_oos_mae if pc._meta_oos_mae == pc._meta_oos_mae else None,
            gbr_oos_n=pc._meta_oos_n if pc._meta_oos_n > 0 else None,
            mmar_calibration_summary=mmar_calib_summary,
            mmar_protocol_dm_test=mmar_protocol_dm,
            output_dir=OUTPUT_DIR,
            extra_summary={
                'scoring_note': ('model_accuracy RMSE/MAE/MAPE/Correlation and diebold_mariano are '
                                 'same-day OOS pricing errors (model price from t-1 parameters vs '
                                 'day-t mid). *_next fields and diebold_mariano_next_week are '
                                 'one-week forecast errors vs the next rebalance mid; contracts not '
                                 'quoted then are excluded. DM_stat is date-clustered.'),
                'diebold_mariano_next_week': dm_next,
                'model_accuracy_closed_form': cf_metrics,
                'diebold_mariano_closed_form': cf_dm,
                'mmar_protocol_dm_next_week': mmar_protocol_dm_next,
                'event_counts': event_counts,
                'config': {'n_mc_paths': N_MC_PATHS, 'risk_free_rate': RISK_FREE_RATE,
                           'eval_contracts_per_cell': EVAL_CONTRACTS_PER_CELL,
                           'eval_moneyness_edges': list(EVAL_MONEYNESS_EDGES)},
            },
        )

    def _event_counts(self, pred_df) -> dict:
        """One definitive table of how many dates enter each part of the study."""
        mondays = pd.date_range(self._start_date, self._end_date, freq='W-MON').date
        trading = set(self._all_prices.index.date)
        return {
            'mondays_in_window':          int(len(mondays)),
            'monday_market_holidays':     int(sum(1 for d in mondays if d not in trading)),
            'rebalance_dates':            int(self._n_rebalances),
            'calibration_attempts':       int(self._n_calibration_attempts),
            'mmar_calibrations_recorded': int(len(self._price_calculator._mmar_calib_history)),
            'evaluated_pricing_dates':    int(pred_df['date'].nunique()) if pred_df is not None else 0,
            'evaluated_contract_dates':   int(len(pred_df)) if pred_df is not None else 0,
            'next_week_realized':         int(pred_df['realized_price'].notna().sum()) if pred_df is not None else 0,
            'margin_halted_weeks':        int(self._n_halted_weeks),
            'strategy_return_weeks':      int(len(self._equity_curve) - 1),
        }

    # ==== QC API shims =====================================================

    def initialize(self):
        self._trading_halted  = False
        self._halt_reason     = None

        self._underlying_symbol = 'SPY'
        self._option_symbol     = 'SPY_OPT'

        self._data_loader   = DatabentoCacheLoader(
            csv_dir   = CSV_DIR,
            cache_dir = CACHE_DIR,
            symbol    = os.environ.get('DATABENTO_SYMBOL', 'SPY'),
        )
        self._prev_calib_df = None   # t-1 cross-section for model calibration
        self._prev_spot     = None
        self._prev_date     = None
        self._today_calib_df = None  # day-t quoted chain (evaluation + next-week realization)
        self._last_eval_date = None  # rebalance date of the most recent priced cross-section
        if not hasattr(self, '_dividends'):
            self._dividends = pd.Series(dtype=float)
        self._n_rebalances           = 0
        self._n_calibration_attempts = 0
        self._n_halted_weeks         = 0

        self._price_history   = pd.DataFrame()
        self._price_calculator = OptionPricingCalculator(self)

        self._portfolio     = Portfolio(100000)
        self._securities    = Securities()
        self._current_slice = CurrentSlice()
        self._current_time  = datetime.datetime.now()

        # One week before the first rebalance so that the first evaluated date
        # (2023-08-28) is already priced from a t−1 calibration.
        self._last_calibration_date        = datetime.date(2023, 8, 18)
        self._calibration_frequency_days   = 7
        self._min_samples_for_calibration  = 10

        self._spy_hedge_quantity    = 0
        self._regime_returns        = {'high_vol': [], 'low_vol': []}
        self._all_weekly_returns    = []
        self._last_portfolio_value  = 100_000.0
        self._trade_log             = []   # {date, contract, qty, entry_price, exit_price, pnl}

        self._strategy_instances = {
            'BuyAndHold':    BuyAndHoldStrategy(),
            'Momentum':      MomentumStrategy(),
            'MeanReversion': MeanReversionStrategy(),
            'MMAR':          MMARStrategy(),
            'BS':            BlackScholesStrategy(),
            'Heston':        HestonStrategy(),
            'Merton':        MertonStrategy(),
            'Bates':         BatesStrategy(),
            'Mixed':         MixedStrategy(),
        }
        self._strategy_selector      = StrategySelector()
        self._pending_hypo_positions = None
        self._pending_hypo_features  = None
        # Pre-seed with synthetic prior rows so the GBR has something to start from
        # on week 1; exponential decay in train() quickly discounts these vs real data
        self._strategy_history       = seed_strategy_history()
        self._active_strategy        = self._strategy_instances['Mixed']

    @property
    def time(self):
        return self._current_time

    @property
    def portfolio(self):
        return self._portfolio

    @property
    def securities(self):
        return self._securities

    @property
    def current_slice(self):
        return self._current_slice

    @property
    def is_warming_up(self):
        return self._current_time.date() < self._start_date

    def debug(self, msg):
        # Simulated time, then wall-clock time elapsed since the run started.
        elapsed = int(time.perf_counter() - getattr(self, '_wall_t0', time.perf_counter()))
        h, rem = divmod(elapsed, 3600)
        print(f"[{self._current_time.strftime('%Y-%m-%d %H:%M')} | +{h:d}:{rem // 60:02d}:{rem % 60:02d}] {msg}",
              flush=True)

    def history(self, symbol, n_bars, resolution=None):
        
        cur = self._current_time.date()
        hist = self._all_prices[self._all_prices.index.date <= cur]  # type: ignore
        # Dividend-adjusted closes: every consumer of history() uses returns.
        result = hist[['SPY_TR']].tail(n_bars).copy()
        result.columns = ['close']
        return result

    def market_order(self, symbol, quantity):
        
        sym_str  = str(symbol)
        price    = self._securities._data.get(sym_str, 0.0)
        if price <= 0:
            return None
        self._portfolio._prices[sym_str] = price
        if quantity > 0:
            filled = self._portfolio._buy(sym_str, quantity, price)
            return filled if filled > 0 else None
        elif quantity < 0:
            filled = self._portfolio._sell(sym_str, abs(quantity), price)
            return filled if filled > 0 else None
        return None

    # No-op stubs for QC methods not needed in standalone mode
    def set_start_date(self, *a):    pass
    def set_end_date(self, *a):      pass
    def set_cash(self, c):           self._portfolio._cash = float(c)
    def set_warm_up(self, *a):       pass
    def add_equity(self, ticker, resolution=None):
        class _A:
            symbol = ticker
        return _A()
    def add_option(self, ticker, resolution=None):
        class _O:
            symbol = f"{ticker}_OPT"
        return _O()
    class _ScheduleStub:
        def on(self, *a, **kw): pass
    schedule = _ScheduleStub()

    # ==== Real / synthetic option chain loading ============================

    def _load_option_chain(self, spot: float,
                           date: datetime.date) -> tuple:

        # Only hit the Databento API for dates inside the paid backtest window.
        # Warmup days use the synthetic fallback so we don't pay for data we
        # never calibrate or trade on.
        if date >= self._start_date:
            calib_df = self._data_loader.get_chain_df(date, spot)
            if calib_df is not None and not calib_df.empty:
                chain = self._df_to_option_chain(calib_df)
                return chain, calib_df
        # Synthetic fallback — covers warmup period and any failed real fetches
        chain = self._generate_option_chain(spot, date)
        return chain, self._chain_to_calib_df(chain, date)

    def _df_to_option_chain(self, chain_df) -> '_OptionChain':

        contracts = []
        for _, row in chain_df.iterrows():
            right = OptionRight.CALL if row['option_type'] == 'call' else OptionRight.PUT
            expiry_dt = datetime.datetime.combine(
                row['expiry'], datetime.time(16, 0))
            contracts.append(_OptionContract(
                symbol    = row['contract_symbol'],
                strike    = row['strike'],
                expiry_dt = expiry_dt,
                right     = right,
                bid       = row['bid_price'],
                ask       = row['ask_price'],
                last      = row['last_price'],
            ))
        return _OptionChain(contracts)

    def _chain_to_calib_df(self, chain: '_OptionChain',
                            date: datetime.date) -> pd.DataFrame:

        rows = []
        for c in chain:
            ttm = max((c.expiry.date() - date).days / 365.25, 0.001)
            mid = (c.bid_price + c.ask_price) / 2.0
            rows.append({
                'contract_symbol': str(c.symbol),
                'strike':          c.strike,
                'expiry':          c.expiry.date(),
                'option_type':     'call' if c.right == OptionRight.CALL else 'put',
                'bid_price':       c.bid_price,
                'ask_price':       c.ask_price,
                'last_price':      c.last_price,
                'mid_price':       mid,
                'ttm':             ttm,
                'volume':          c.volume,
                'open_interest':   c.open_interest,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _calibrate_model_params(self) -> None:

        if self._prev_calib_df is None or self._prev_spot is None:
            return
        self._n_calibration_attempts += 1
        # Dividend-adjusted daily log returns strictly before today (through t−1),
        # expanding window from the start of the downloaded history.
        adj = self._all_prices['SPY_TR']
        adj = adj[adj.index.date < self.time.date()]
        log_rets = np.log(adj).diff().dropna().values
        try:
            self._price_calculator.calibrate_from_cross_section(
                self._prev_calib_df, self._prev_spot, r=RISK_FREE_RATE,
                log_returns=log_rets, q=self._dividend_yield(self._prev_date, self._prev_spot),
                calib_date=self._prev_date)
        except Exception as e:
            self.debug(f"Model calibration error: {e}")

    def _dividend_yield(self, date: datetime.date, spot: float) -> float:
        """Trailing-12-month cash dividends (ex-dates ≤ date) / unadjusted spot."""
        if self._dividends is None or len(self._dividends) == 0 or not spot:
            return 0.0
        end = pd.Timestamp(date)
        window = self._dividends[(self._dividends.index > end - pd.Timedelta(days=365))
                                 & (self._dividends.index <= end)]
        return float(window.sum() / spot)

    # ==== All existing algorithm methods (unchanged from QC version) =======

    def _close_position_with_log(self, symbol) -> tuple:

        pos = self.portfolio[symbol]
        if not pos.invested:
            return 0, 0.0
        qty        = pos.quantity      # negative for shorts
        avg_entry  = pos.average_price
        exit_price = float(self.securities[symbol].price)
        sym_str    = str(symbol)

        if qty > 0:
            filled = self._portfolio._sell(sym_str, qty, exit_price)
            pnl    = (exit_price - avg_entry) * qty * 100
        else:
            filled = self._portfolio._cover(sym_str, abs(qty), exit_price)
            # Short P&L: received avg_entry premium, pay exit_price to cover
            pnl    = (avg_entry - exit_price) * abs(qty) * 100

        if filled:
            self._trade_log.append({
                'date':        self.time.date(),
                'contract':    sym_str,
                'qty':         qty,
                'entry_price': avg_entry,
                'exit_price':  exit_price,
                'pnl':         pnl,
            })
            return abs(qty), exit_price
        return 0, 0.0

    @staticmethod
    def _parse_contract(symbol_str: str):
        """
        (option_type, strike, expiry_date) from an OSI symbol such as
        'SPY   231020P00425000' (Databento), or from the synthetic warm-up
        format 'SPY 231020 P00425'. Returns None if unparseable.
        """
        parsed = _parse_osi_symbol(str(symbol_str))
        if parsed is not None:
            _, expiry, option_type, strike = parsed
            return option_type, strike, expiry
        parts = str(symbol_str).split()
        try:
            expiry = datetime.datetime.strptime(parts[1], '%y%m%d').date()
            return ('call' if parts[2][0] == 'C' else 'put'), float(parts[2][1:]), expiry
        except (ValueError, IndexError):
            return None

    def _days_to_expiry(self, symbol_str: str) -> int:
        parsed = self._parse_contract(symbol_str)
        if parsed is None:
            return 999
        return max((parsed[2] - self.time.date()).days, 0)

    def _settle_expired_positions(self, date: datetime.date, spot: float) -> None:
        """
        Cash-settle every held option whose expiry is on or before `date` at
        intrinsic value against the unadjusted close on its expiry date (SPY
        options are physically settled; intrinsic value is the equivalent
        exercise value for an ITM contract and zero otherwise).
        """
        for sym in list(self._portfolio._positions.keys()):
            parsed = self._parse_contract(sym)
            if parsed is None or parsed[2] > date:
                continue
            option_type, strike, expiry = parsed
            closes = self._all_prices['SPY'][self._all_prices.index.date <= expiry]
            s_exp = float(closes.iloc[-1]) if len(closes) else spot
            intrinsic = max(s_exp - strike, 0.0) if option_type == 'call' else max(strike - s_exp, 0.0)
            pos = self._portfolio._positions.pop(sym)
            self._portfolio._cash += pos.quantity * intrinsic * 100
            self._portfolio._prices.pop(sym, None)
            self._trade_log.append({
                'date':        date,
                'contract':    sym,
                'qty':         pos.quantity,
                'entry_price': pos.average_price,
                'exit_price':  intrinsic,
                'pnl':         (intrinsic - pos.average_price) * pos.quantity * 100,
                'settled':     True,
            })
            self.debug(f"SETTLED {pos.quantity} {sym} at expiry, intrinsic ${intrinsic:.2f} (S={s_exp:.2f})")

    def _selective_close_positions(self):

        TAKE_PROFIT  =  0.50   # close if option gained 50%
        STOP_LOSS    = -0.80   # close if option lost 80%
        DTE_CUTOFF   = 3

        for symbol in list(self.portfolio.keys()):
            pos = self.portfolio[symbol]
            if not pos.invested:
                continue
            cur_price  = float(self.securities[symbol].price)
            avg_entry  = pos.average_price
            dte        = self._days_to_expiry(symbol)
            if avg_entry <= 0:
                continue

            pct       = (cur_price - avg_entry) / avg_entry
            should_close = dte <= DTE_CUTOFF or pct >= TAKE_PROFIT or pct <= STOP_LOSS
            if should_close:
                self._close_position_with_log(symbol)

    def _option_filter(self, universe):
        return universe.strikes(-2, 2).expiration(14, 21)

    def _update_price_history(self):
        history = self.history(self._underlying_symbol, 60)
        if not history.empty:
            self._price_history = history[['close']].rename(columns={'close': 'SPY'})
            self._price_calculator.update_price_history(self._price_history)

    def _get_option_chain(self):
        slice_data = self.current_slice.option_chains.get(self._option_symbol)
        if slice_data is None:
            return None
        contracts = []
        for contract in slice_data:
            contracts.append({
                'symbol':        contract.symbol,
                'strike':        contract.strike,
                'expiry':        contract.expiry,
                'right':         contract.right,
                'bid':           contract.bid_price,
                'ask':           contract.ask_price,
                'last':          contract.last_price,
                'volume':        contract.volume,
                'open_interest': contract.open_interest,
            })
        if not contracts:
            return None
        return pd.DataFrame(contracts)

    def _select_evaluation_universe(self, chain_df: pd.DataFrame, spot: float) -> pd.DataFrame:
        """
        Stratified OOS evaluation sample for one rebalance date: up to
        EVAL_CONTRACTS_PER_CELL contracts from each K/S band (0.85–1.15) ×
        maturity bucket (7–30, 31–90 days), seeded by the date.
        """
        today = self.time.date()
        df = chain_df.copy()
        days = df['expiry'].map(lambda e: (e - today).days)
        df = df[(days >= EVAL_MIN_DTE) & (days <= EVAL_MAX_DTE)
                & (df['bid_price'] > 0) & (df['ask_price'] >= df['bid_price'])]
        df['ttm'] = days[df.index] / 365.25
        return stratified_sample(df, spot, EVAL_CONTRACTS_PER_CELL,
                                 seed=int(today.strftime('%Y%m%d')),
                                 moneyness_edges=EVAL_MONEYNESS_EDGES)

    def _prepare_option_data(self, universe_df: pd.DataFrame) -> pd.DataFrame:
        """
        Price the day-t evaluation universe under every model (parameters from
        t−1), record each contract-date for OOS scoring, and return the
        prepared-options table the trading strategies consume.
        """
        spot = float(self.securities[self._underlying_symbol].price)
        q = self._dividend_yield(self.time.date(), spot)
        prices = self._price_calculator.calculate_model_prices_batch(
            universe_df, spot, RISK_FREE_RATE, q, self.time)
        tracker = self._price_calculator.get_performance_tracker()
        prepared = []
        for i, row in universe_df.iterrows():
            option_type = row['option_type']
            cp = 'Call' if option_type == 'call' else 'Put'
            mid_price = float(row['mid_price'])
            model_prices = {f'{PRICE_LABELS[m]} {cp}': (None if pd.isna(prices.at[i, m]) else float(prices.at[i, m]))
                            for m in MODEL_COLUMNS}
            moneyness = float(row['strike']) / spot
            tracker.record_prediction(
                timestamp=self.time,
                contract_symbol=str(row['contract_symbol']),
                strike=row['strike'],
                expiry=row['expiry'],
                option_type=option_type,
                model_prices_dict=model_prices,
                actual_price=mid_price,
                volatility=self._price_calculator._last_volatility,
                moneyness=moneyness,
                ttm=float(row['ttm']),
                spot=spot,
                dividend_yield=q,
            )
            record = {
                'contract_symbol': str(row['contract_symbol']),
                'Model Strike':    row['strike'],
                'Expiration Date': row['expiry'].strftime('%Y-%m-%d'),
                'option_type':     option_type,
                'spread_cost':     self._compute_spread_cost(mid_price, moneyness, float(row['ttm'])),
                'Actual Call Price': mid_price if option_type == 'call' else None,
                'Actual Put Price':  mid_price if option_type == 'put'  else None,
            }
            for m in MODEL_COLUMNS:
                for side in ('Call', 'Put'):
                    record[f'{PRICE_LABELS[m]} {side}'] = model_prices[f'{PRICE_LABELS[m]} {cp}'] if side == cp else None
            prepared.append(record)
        self._last_eval_date = self.time.date()
        return pd.DataFrame(prepared)

    MAX_CONCURRENT_POSITIONS = 6

    def _execute_signal(self, signal, prepared_options):
        if self._trading_halted and signal['type'] in ('buy', 'write'):
            return 0
        contract_symbol_str = signal['contract_symbol']
        option_data = prepared_options[
            prepared_options['contract_symbol'] == contract_symbol_str]
        if option_data.empty:
            return 0
        matching_contracts = [s for s in self.securities.keys()
                              if str(s) == contract_symbol_str]
        if not matching_contracts:
            return 0
        symbol        = matching_contracts[0]
        requested_qty = signal['quantity']
        option_price  = signal['price']
        option_row    = option_data.iloc[0]

        if signal['type'] == 'buy':
            if self.portfolio[symbol].invested:
                return 0  # don't pyramid into existing position
            if len(self._portfolio._positions) >= self.MAX_CONCURRENT_POSITIONS:
                return 0
            if not self._has_sufficient_margin(option_price, requested_qty):
                self.debug(f"SKIPPED BUY {symbol}: Insufficient margin.")
                return 0
            max_qty    = self._calculate_max_position_quantity(option_price, 0.3)
            actual_qty = min(requested_qty, max_qty)
            if actual_qty <= 0:
                return 0
            order = self.market_order(symbol, actual_qty)
            if order:
                self.debug(f"BUY {actual_qty} {symbol} @ ${option_price:.2f}")
                delta = self._compute_option_delta(option_row)
                return -round(delta * actual_qty * 100)
            return 0

        elif signal['type'] == 'write':
            pos = self.portfolio[symbol]
            if pos.invested and pos.quantity > 0:
                # Close the long that is now deemed overpriced
                exit_price = float(self.securities[symbol].price)
                qty_sold, _ = self._close_position_with_log(symbol)
                if qty_sold > 0:
                    self.debug(f"CLOSED LONG (overpriced) {qty_sold} {symbol} @ ${exit_price:.2f}")
                return 0  # don't also open a short in the same rebalance
            if pos.invested and pos.quantity < 0:
                return 0  # already short, don't double up
            if len(self._portfolio._positions) >= self.MAX_CONCURRENT_POSITIONS:
                return 0
            if not self._has_sufficient_margin(option_price, requested_qty):
                self.debug(f"SKIPPED WRITE {symbol}: Insufficient margin.")
                return 0
            max_qty    = self._calculate_max_position_quantity(option_price, 0.3)
            actual_qty = min(requested_qty, max_qty)
            if actual_qty <= 0:
                return 0
            filled = self._portfolio._write(str(symbol), actual_qty, option_price)
            if filled:
                self.debug(f"WRITE {actual_qty} {symbol} @ ${option_price:.2f}")
                self._portfolio._prices[str(symbol)] = option_price
                delta = self._compute_option_delta(option_row)
                # Short position: hedge in the opposite direction to a long
                return round(delta * actual_qty * 100)
            return 0

        elif signal['type'] == 'sell':
            if self.portfolio[symbol].invested:
                exit_price = float(self.securities[symbol].price)
                qty_sold, _ = self._close_position_with_log(symbol)
                if qty_sold > 0:
                    self.debug(f"SELL {qty_sold} {symbol} @ ${exit_price:.2f}")
            return 0
        return 0

    def _liquidate_all_positions(self):
        self.debug("Emergency liquidation …")
        for symbol in list(self.portfolio.keys()):
            self._close_position_with_log(symbol)

    def _run_pricing_evaluation(self):
        """
        Pricing experiment for this rebalance date — independent of the trading
        account (runs even in a margin-halted week):
          1. calibrate every model on the t−1 cross-section,
          2. realize last rebalance's predictions from today's quoted chain,
          3. train the GBR pricing ensemble on prior dates only,
          4. price the stratified day-t universe and record it for OOS scoring.
        Returns the prepared-options table (or None if no chain is available).
        """
        days_since = (self.time.date() - self._last_calibration_date).days
        if days_since >= self._calibration_frequency_days:
            self._calibrate_model_params()
            self._last_calibration_date = self.time.date()
        self._mark_to_market_realizations()

        tracker = self._price_calculator.get_performance_tracker()
        pred_df = tracker.get_prediction_dataframe()
        if pred_df is not None and len(pred_df) >= self._min_samples_for_calibration:
            try:
                self._price_calculator.calibrate_ensemble_weights(pred_df[pred_df['date'] < self.time.date()])
            except Exception as e:
                self.debug(f"Pricing GBR failed: {e}")

        if self._today_calib_df is None or self._today_calib_df.empty:
            self.debug(f"No option chain at {self.time}")
            return None
        spot = float(self.securities[self._underlying_symbol].price)
        universe = self._select_evaluation_universe(self._today_calib_df, spot)
        if universe.empty:
            self.debug("No contracts in the evaluation universe after filtering")
            return None
        self.debug(f"Evaluation universe: {len(universe)} contracts (stratified K/S × maturity)")
        prepared = self._prepare_option_data(universe)
        self._n_rebalances += 1

        dm = tracker.run_diebold_mariano_tests('same_day')
        if dm:
            self.debug("DM (same-day, date-clustered) vs BS: " + "  ".join(
                f"{k.replace('_vs_BS', '')} {v['DM_stat']:+.2f} (p={v['p_value']:.3f}, T={v['n_dates']})"
                for k, v in dm.items()))
        return prepared

    def _rebalance(self):
        if self.is_warming_up:
            return

        self._update_price_history()
        prepared_options = self._run_pricing_evaluation()

        # ---- trading (does not feed back into the pricing experiment) ----
        if self._trading_halted:
            self._n_halted_weeks += 1
            self.debug(f"TRADING HALTED – {self._halt_reason}")
            if self._spy_hedge_quantity != 0:
                self.market_order(self._underlying_symbol, -self._spy_hedge_quantity)
                self._spy_hedge_quantity = 0
            self._liquidate_all_positions()
            # Liquidating frees essentially all margin, so re-check and resume
            # once positions are flat and margin is available again.
            if self._margin_has_recovered():
                self.debug(f"Trading resumed — margin recovered (halt was: {self._halt_reason}).")
                self._trading_halted = False
                self._halt_reason    = None
            return

        current_pv = self.portfolio.total_portfolio_value
        if self._last_portfolio_value > 0:
            wr  = (current_pv - self._last_portfolio_value) / self._last_portfolio_value
            vol = self._price_calculator._last_volatility
            self._regime_returns['high_vol' if vol > 0.20 else 'low_vol'].append(wr)
            self._all_weekly_returns.append(wr)
        self._last_portfolio_value = current_pv

        if self._spy_hedge_quantity != 0:
            self.market_order(self._underlying_symbol, -self._spy_hedge_quantity)
            self._spy_hedge_quantity = 0

        self.debug("Selective close: near-expiry / P&L threshold positions…")
        self._selective_close_positions()

        current_features = self._compute_strategy_features()

        if self._pending_hypo_positions is not None:
            hypo_returns = self._compute_hypo_returns(self._pending_hypo_positions)
            row = {**self._pending_hypo_features, **hypo_returns}
            self._strategy_history.append(row)
            self.debug(f"Hypo returns: { {k: round(v,5) for k,v in hypo_returns.items()} }")

        self._pending_hypo_features = current_features

        if len(self._strategy_history) >= StrategySelector.MIN_SAMPLES:
            try:
                self._strategy_selector.train(self._strategy_history, self)
            except Exception as e:
                self.debug(f"StrategySelector training failed: {e}")

        if prepared_options is None or prepared_options.empty:
            self.debug("No valid options after preparation")
            return

        current_price = self.securities[self._underlying_symbol].price
        market_data   = pd.Series({'SPY': current_price})

        mock_bt = _MockBacktester(self)
        new_hypo_positions = {}
        for strat_name, strat_obj in self._strategy_instances.items():
            try:
                hypo_sigs = strat_obj.generate_signals(
                    self.time, market_data, prepared_options, {}, mock_bt)
                new_hypo_positions[strat_name] = [
                    (s['contract_symbol'], s['type'],
                     s.get('quantity', 1), s.get('price'))
                    for s in hypo_sigs
                    if s.get('price') is not None and s.get('price', 0) > 0
                ]
            except Exception as e:
                self.debug(f"Hypo signal gen failed [{strat_name}]: {e}")
                new_hypo_positions[strat_name] = []

        self._pending_hypo_positions = new_hypo_positions

        if self._strategy_selector.is_trained:
            best_name = self._strategy_selector.select(current_features)
            self._active_strategy = self._strategy_instances.get(
                best_name, self._strategy_instances['Mixed'])
            self.debug(f"StrategySelector: {best_name}")

        signals = self._active_strategy.generate_signals(
            self.time, market_data, prepared_options, self.portfolio, self)

        net_delta_hedge = 0
        for signal in signals:
            net_delta_hedge += self._execute_signal(signal, prepared_options)

        if net_delta_hedge != 0:
            self.market_order(self._underlying_symbol, net_delta_hedge)
            self._spy_hedge_quantity = net_delta_hedge

    def _mark_to_market_realizations(self):
        """
        Next-week realization: fill predictions made on the previous rebalance
        date with the mids of today's quoted chain only. Contracts not quoted
        today (e.g. expired) stay unrealized and are excluded from the
        next-week metrics rather than back-filled with stale prices.
        """
        tracker = self._price_calculator.get_performance_tracker()
        if self._last_eval_date is None or self._today_calib_df is None or self._today_calib_df.empty:
            return
        day_mids = dict(zip(self._today_calib_df['contract_symbol'].astype(str),
                            self._today_calib_df['mid_price'].astype(float)))
        n_filled = tracker.mark_open_predictions(day_mids, from_date=self._last_eval_date)
        n_prev = sum(1 for r in tracker._prediction_history
                     if r['timestamp'].date() == self._last_eval_date)
        self.debug(f"Next-week realization: {n_filled} of {n_prev} predictions from "
                   f"{self._last_eval_date} quoted today")

    def _calculate_max_position_quantity(self, option_price, margin_buffer=0.3):
        avail   = self.portfolio.margin_remaining
        req_per = option_price * 100
        usable  = avail * (1 - margin_buffer)
        if usable <= 0 or req_per <= 0:
            return 0
        return int(usable / req_per)

    def _compute_spread_cost(self, mid_price, moneyness, ttm):
        otm_deg = abs(np.log(moneyness)) if moneyness > 0 else 0.0
        mf      = 1.0 + max(0.0, (otm_deg - 0.02) * 10.0)
        tf      = 1.0 + max(0.0, (0.04 - ttm) / 0.04)
        return 0.01 * mid_price * mf * tf

    def _has_sufficient_margin(self, option_price, quantity=1):
        required  = option_price * 100 * quantity * 1.3
        available = self.portfolio.margin_remaining
        if available < required:
            if not self._trading_halted:
                self._trading_halted = True
                self._halt_reason = (
                    f"insufficient margin (required ${required:,.0f} > "
                    f"available ${available:,.0f})"
                )
            return False
        return True

    def _margin_has_recovered(self) -> bool:
        """
        True once open positions are flat and margin is available again — the
        condition under which a margin-triggered halt can safely lift. Liquidating
        on entry to the halted branch of _rebalance() closes every position, so
        margin_remaining reverts to available cash; this only stays False if cash
        itself has been driven to zero or negative by realized losses.
        """
        return len(self._portfolio._positions) == 0 and self.portfolio.margin_remaining > 0

    def _compute_option_delta(self, option_row):
        import datetime as dt_mod
        current_spot = float(self.securities[self._underlying_symbol].price)
        strike       = float(option_row['Model Strike'])
        option_type  = option_row['option_type']
        try:
            expiry = dt_mod.datetime.strptime(
                option_row['Expiration Date'], '%Y-%m-%d').date()
            ttm = max((expiry - self.time.date()).days / 365.25, 0.001)
        except (ValueError, AttributeError):
            ttm = 0.1
        sigma = max(self._price_calculator._last_volatility, 0.05)
        q = self._dividend_yield(self.time.date(), current_spot)
        return bs_delta(current_spot, strike, RISK_FREE_RATE, sigma, ttm, option_type, q)

    def _compute_strategy_features(self):
        vol = self._price_calculator._last_volatility
        if self._price_history is None or self._price_history.empty or len(self._price_history) < 6:
            return {'volatility': vol, 'momentum_5d': 0.0,
                    'ma_deviation_20': 0.0, 'hurst': 0.5}
        prices = self._price_history['SPY']
        momentum_5d = float(
            (prices.iloc[-1] - prices.iloc[-6]) / prices.iloc[-6]
        ) if len(prices) >= 6 else 0.0
        if len(prices) >= 20:
            ma20 = float(prices.rolling(20).mean().iloc[-1])
            ma_dev = (float(prices.iloc[-1]) - ma20) / ma20 if ma20 > 0 else 0.0
        else:
            ma_dev = 0.0
        try:
            hurst_vals = calculate_hurst_for_segments(prices.values, 8)
            hurst      = float(np.mean(hurst_vals))
        except Exception:
            hurst = 0.5
        return {'volatility': vol, 'momentum_5d': float(momentum_5d),
                'ma_deviation_20': float(ma_dev), 'hurst': float(hurst)}

    def _compute_hypo_returns(self, hypo_positions):
        """
        Approximate one-week return each strategy would have earned on last
        week's signals (the strategy selector's learning target):
            P&L_i = δ_i · r_SPY · S · q_i · 100 + Θ_i · (7/365) · q_i · 100
        with δ_i and Θ_i the Black-Scholes delta and theta (per year, with the
        dividend yield) at each contract's actual strike, type and time to
        expiry, using the calibrated BS σ. Sign flips for written positions.
        """
        returns = {}
        current_spy = float(self.securities[self._underlying_symbol].price)
        sigma       = self._price_calculator._bs_sigma() or 0.20
        q           = self._dividend_yield(self.time.date(), current_spy)

        # 1-week SPY return computed from the last 6 price bars (~5 trading days)
        if self._price_history is not None and len(self._price_history) >= 6:
            prev_spy    = float(self._price_history['SPY'].iloc[-6])
            spy_return  = (float(self._price_history['SPY'].iloc[-1]) - prev_spy) / prev_spy if prev_spy > 0 else 0.0
        else:
            spy_return = 0.0

        for name, positions in hypo_positions.items():
            if not positions:
                returns[name] = 0.0
                continue
            total_pnl = total_cost = 0.0
            for contract_symbol, sig_type, qty, entry_price in positions:
                if entry_price is None or entry_price <= 0:
                    continue
                parsed = self._parse_contract(contract_symbol)
                if parsed is None:
                    continue
                opt_type, strike, expiry = parsed
                ttm = max((expiry - self.time.date()).days / 365.25, 1 / 365.25)
                delta = bs_delta(current_spy, strike, RISK_FREE_RATE, sigma, ttm, opt_type, q)
                theta = bs_theta(current_spy, strike, RISK_FREE_RATE, sigma, ttm, opt_type, q)

                delta_pnl = delta * spy_return * current_spy * qty * 100
                theta_pnl = theta * (7.0 / 365.0) * qty * 100
                approx_pnl = delta_pnl + theta_pnl
                if sig_type == 'write':
                    approx_pnl = -approx_pnl   # short: opposite exposure
                total_pnl  += approx_pnl
                total_cost += abs(entry_price) * qty * 100
            returns[name] = float(total_pnl / total_cost) if total_cost > 0 else 0.0
        return returns

    def _monitor_margin_health(self):
        used      = self.portfolio.total_margin_used
        available = self.portfolio.margin_remaining
        leverage  = ((used + available) / (available + 1e-9)
                     if available > 0 else 0)
        self.debug(f"Margin – Used: ${used:.0f}  Avail: ${available:.0f}  "
                   f"Lev: {leverage:.2f}x")
        if available < (used + available) * 0.3:
            self.debug("WARNING: margin approaching 70% limit!")


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    algo = OptionsArbitrageAlgorithm()
    algo.run_backtest()
