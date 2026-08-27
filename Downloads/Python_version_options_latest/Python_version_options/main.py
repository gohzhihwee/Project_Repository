from AlgorithmImports import *          
from models import *
from modelstrats import *
from convmethods import *
from ensemble import *
from data_loader import DatabentoCacheLoader

import os
import pathlib
from dotenv import load_dotenv
load_dotenv()          
import numpy as np
import pandas as pd
import datetime
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor

OUTPUT_DIR = pathlib.Path('/app/output')
CACHE_DIR  = pathlib.Path(__file__).parent / 'option_cache'
CSV_DIR    = pathlib.Path(__file__).parent / 'options_data'  # drop Databento CSV exports here

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
        print("Downloading SPY daily data via yfinance …")
        import yfinance as yf
        
        raw = yf.download("SPY", start="2021-06-01", end="2024-09-01",
                          progress=False, auto_adjust=True)
        raw.index = pd.to_datetime(raw.index).normalize()
        self._all_prices = raw[['Close']].rename(columns={'Close': 'SPY'})
        print(f"  {len(self._all_prices)} daily bars loaded.")
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
            if day_idx % 10 == 0:
                print(f"[{sim_date}] Day {day_idx + 1}/{n_days}  "
                      f"SPY ${spy_price:.2f}  "
                      f"PV ${self._portfolio.total_portfolio_value:,.0f}")

            
            chain, calib_df = self._load_option_chain(spy_price, sim_date)
            self._current_slice.option_chains._set(self._option_symbol, chain)
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

        self._print_summary()
        self._generate_output()

    # ==== synthetic option chain ==========================================

    def _generate_option_chain(self, spot, current_date):

        hist = self._all_prices[self._all_prices.index.date <= current_date]  # type: ignore
        log_rets = np.log(hist['SPY'].pct_change() + 1).dropna()
        sigma = float(log_rets.tail(20).std() * np.sqrt(252)) if len(log_rets) >= 5 else 0.20
        sigma = max(sigma, 0.05)
        r = 0.052

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

        # SPY weekly returns aligned to rebalance dates for alpha/beta
        spy_weekly_rets = None
        try:
            ec_dates = pd.to_datetime([r['date'] for r in self._equity_curve])
            spy_at_dates = self._all_prices['SPY'].reindex(ec_dates, method='ffill').dropna()
            spy_weekly_rets = spy_at_dates.pct_change().dropna()
        except Exception as e:
            self.debug(f"SPY benchmark alignment failed: {e}")

        pm = compute_portfolio_metrics(
            equity_curve=curve_df,
            weekly_returns=self._all_weekly_returns,
            trade_log=self._trade_log,
            spy_returns=spy_weekly_rets,
            initial_capital=100_000.0,
        )
        self.debug(f"Portfolio metrics: {pm}")

        tracker      = self._price_calculator.get_performance_tracker()
        model_metrics = tracker.get_model_accuracy_metrics()
        dm_results    = tracker.run_diebold_mariano_tests()

        pc = self._price_calculator
        ss = self._strategy_selector

        generate_visualizations(
            equity_curve=self._equity_curve,
            weekly_returns=self._all_weekly_returns,
            spy_prices=self._all_prices,
            model_accuracy_metrics=model_metrics,
            dm_results=dm_results,
            trade_log=self._trade_log,
            gbr_meta_importances=pc._meta_importances if pc._meta_importances else None,
            gbr_meta_train_rmse=pc._meta_train_rmse if not (
                pc._meta_train_rmse != pc._meta_train_rmse) else None,  # nan check
            strategy_selector_importances=ss.feature_importances if ss.feature_importances else None,
            portfolio_metrics=pm,
            output_dir=OUTPUT_DIR,
        )

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

        self._price_history   = pd.DataFrame()
        self._price_calculator = OptionPricingCalculator(self)

        self._portfolio     = Portfolio(100000)
        self._securities    = Securities()
        self._current_slice = CurrentSlice()
        self._current_time  = datetime.datetime.now()

        self._last_calibration_date        = datetime.date(2023, 8, 25)
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
        print(f"[{self._current_time.strftime('%Y-%m-%d %H:%M')}] {msg}")

    def history(self, symbol, n_bars, resolution=None):
        
        cur = self._current_time.date()
        hist = self._all_prices[self._all_prices.index.date <= cur]  # type: ignore
        result = hist.tail(n_bars).copy()
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
        log_rets = np.array([])
        if (self._price_history is not None
                and not self._price_history.empty
                and len(self._price_history) >= 30):
            log_rets = np.log(
                self._price_history['SPY'].pct_change() + 1).dropna().values
        try:
            self._price_calculator.calibrate_from_cross_section(
                self._prev_calib_df, self._prev_spot, r=0.052,
                log_returns=log_rets)
        except Exception as e:
            self.debug(f"Model calibration error: {e}")

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

    def _days_to_expiry(self, symbol_str: str) -> int:

        try:
            parts = str(symbol_str).split()
            if len(parts) >= 2:
                expiry = datetime.datetime.strptime(parts[1], '%y%m%d').date()
                return max((expiry - self.time.date()).days, 0)
        except (ValueError, IndexError):
            pass
        return 999

    def _selective_close_positions(self):

        TAKE_PROFIT  =  0.50   # close if option gained 50%
        STOP_LOSS    = -0.80   # close if option lost 80%
        DTE_CUTOFF   = 3

        tracker = self._price_calculator.get_performance_tracker()
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
                qty_closed, exit_px = self._close_position_with_log(symbol)
                if qty_closed > 0:
                    tracker.record_realization(str(symbol), exit_px)

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

    def _prepare_option_data(self, chain, precomputed_hurst=0.5):
        current_price = self.securities[self._underlying_symbol].price
        current_time  = self.time
        n_total  = len(chain)
        prepared = []
        for idx, (_, row) in enumerate(chain.iterrows()):
            if idx % 25 == 0:
                self.debug(f"  Pricing {idx + 1}/{n_total} contracts …")
            option_type = 'call' if row['right'] == OptionRight.CALL else 'put'
            mid_price   = ((row['bid'] + row['ask']) / 2
                           if row['ask'] > 0 else row['last'])
            if mid_price <= 0:
                continue
            model_prices = self._price_calculator.calculate_model_prices(
                current_price, row['strike'], row['expiry'],
                current_time, num_paths=50, hurst=precomputed_hurst)
            current_spot    = float(current_price)
            option_ttm      = max((row['expiry'].date() - current_time.date()).days / 365.25, 0.001)
            option_moneyness = float(row['strike']) / current_spot if current_spot > 0 else 1.0
            self._price_calculator.get_performance_tracker().record_prediction(
                timestamp=current_time,
                contract_symbol=str(row['symbol']),
                strike=row['strike'],
                expiry=row['expiry'],
                option_type=option_type,
                model_prices_dict=model_prices,
                actual_price=mid_price,
                volatility=self._price_calculator._last_volatility,
                moneyness=option_moneyness,
                ttm=option_ttm,
            )
            spread_cost = self._compute_spread_cost(mid_price, option_moneyness, option_ttm)
            record = {
                'contract_symbol': str(row['symbol']),
                'Model Strike':    row['strike'],
                'Expiration Date': row['expiry'].strftime('%Y-%m-%d'),
                'option_type':     option_type,
                'spread_cost':     spread_cost,
                'Actual Call Price': mid_price if option_type == 'call' else None,
                'Actual Put Price':  mid_price if option_type == 'put'  else None,
                'MMAR Call':   model_prices.get('MMAR Call')   if option_type == 'call' else None,
                'MMAR Put':    model_prices.get('MMAR Put')    if option_type == 'put'  else None,
                'BS Call':     model_prices.get('BS Call')     if option_type == 'call' else None,
                'BS Put':      model_prices.get('BS Put')      if option_type == 'put'  else None,
                'Merton Call': model_prices.get('Merton Call') if option_type == 'call' else None,
                'Merton Put':  model_prices.get('Merton Put')  if option_type == 'put'  else None,
                'Heston Call': model_prices.get('Heston Call') if option_type == 'call' else None,
                'Heston Put':  model_prices.get('Heston Put')  if option_type == 'put'  else None,
                'Bates Call':  model_prices.get('Bates Call')  if option_type == 'call' else None,
                'Bates Put':   model_prices.get('Bates Put')   if option_type == 'put'  else None,
                'Mixed Call':  model_prices.get('Mixed Call')  if option_type == 'call' else None,
                'Mixed Put':   model_prices.get('Mixed Put')   if option_type == 'put'  else None,
            }
            prepared.append(record)
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
                self._price_calculator.get_performance_tracker().record_realization(
                    contract_symbol_str, exit_price)
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
                self._price_calculator.get_performance_tracker().record_realization(
                    contract_symbol_str, exit_price)
                qty_sold, _ = self._close_position_with_log(symbol)
                if qty_sold > 0:
                    self.debug(f"SELL {qty_sold} {symbol} @ ${exit_price:.2f}")
            return 0
        return 0

    def _liquidate_all_positions(self):
        self.debug("Emergency liquidation …")
        for symbol in list(self.portfolio.keys()):
            self._close_position_with_log(symbol)

    def _rebalance(self):
        if self.is_warming_up:
            return
        if self._trading_halted:
            self.debug(f"TRADING HALTED – {self._halt_reason}")
            if self._spy_hedge_quantity != 0:
                self.market_order(self._underlying_symbol, -self._spy_hedge_quantity)
                self._spy_hedge_quantity = 0
            self._liquidate_all_positions()
            return

        self._update_price_history()
        precomputed_hurst = 0.5
        try:
            hurst_values = calculate_hurst_for_segments(
                self._price_history['SPY'].values, 8)
            precomputed_hurst = float(np.mean(hurst_values))
        except Exception as e:
            self.debug(f"Hurst calculation failed: {e}, using 0.5")

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

        days_since = (self.time.date() - self._last_calibration_date).days
        if days_since >= self._calibration_frequency_days:
            self._attempt_ensemble_calibration()
            self._last_calibration_date = self.time.date()

        chain = self._get_option_chain()
        if chain is None or chain.empty:
            self.debug(f"No option chain at {self.time}")
            return
        self.debug(f"Chain: {len(chain)} contracts")

        # Restrict to near-ATM contracts with reasonable TTM before pricing.
        # Pricing all ~6k contracts per rebalance is prohibitive; this mirrors
        # the moneyness/TTM window used for calibration.
        today = self.time.date()
        spot  = float(self.securities[self._underlying_symbol].price)
        chain = chain[
            (chain['strike'] >= spot * 0.85) &
            (chain['strike'] <= spot * 1.15) &
            (chain['expiry'].apply(
                lambda e: 7 <= (e.date() - today).days <= 90))
        ].assign(_dist=lambda df: abs(df['strike'] - spot)
        ).nsmallest(150, '_dist').drop(columns='_dist').reset_index(drop=True)
        if chain.empty:
            self.debug("No contracts in tradeable universe after ATM/TTM filter")
            return
        self.debug(f"Filtered to {len(chain)} tradeable contracts")

        prepared_options = self._prepare_option_data(chain, precomputed_hurst)
        if prepared_options.empty:
            self.debug("No valid options after preparation")
            return
        self.debug(f"Prepared {len(prepared_options)} options")

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

        if self._trading_halted:
            signals = [s for s in signals if s['type'] == 'sell']
            if not signals:
                return

        net_delta_hedge = 0
        for signal in signals:
            net_delta_hedge += self._execute_signal(signal, prepared_options)

        if net_delta_hedge != 0:
            self.market_order(self._underlying_symbol, net_delta_hedge)
            self._spy_hedge_quantity = net_delta_hedge

    def _mark_to_market_realizations(self):

        tracker   = self._price_calculator.get_performance_tracker()
        n_total   = len(tracker._prediction_history)
        n_filled  = tracker.mark_open_predictions(self._securities._data)
        n_realized = sum(1 for r in tracker._prediction_history
                         if r['realized_price'] is not None)
        self.debug(
            f"MTM: history={n_total} predictions, "
            f"+{n_filled} newly realized, "
            f"{n_realized} total realized, "
            f"{len(self._securities._data)} securities in dict"
        )

    def _attempt_ensemble_calibration(self):
        # Calibrate pricing model params (NLS/MLE) on t-1 cross-section first
        self._calibrate_model_params()
        self.debug("Checking GBR ensemble calibration…")
        tracker = self._price_calculator.get_performance_tracker()
        # Mark-to-market before counting — ensures calibration data from week 1
        self._mark_to_market_realizations()
        calibration_df = tracker.get_calibration_dataframe()
        if calibration_df is None or len(calibration_df) < self._min_samples_for_calibration:
            self.debug(f"GBR skipped: {0 if calibration_df is None else len(calibration_df)} samples")
            return
        metrics = tracker.get_model_accuracy_metrics()
        if metrics:
            self.debug(f"Model accuracy: {metrics}")
        dm_results = tracker.run_diebold_mariano_tests()
        if dm_results:
            self.debug(f"DM tests: {dm_results}")
        for regime, returns in self._regime_returns.items():
            if len(returns) >= 10:
                bs_result = block_bootstrap_sharpe(np.array(returns))
                if bs_result:
                    self.debug(
                        f"Regime [{regime}] Sharpe={bs_result['observed_sharpe']:.3f} "
                        f"95% CI=[{bs_result['ci_95_lower']:.3f},{bs_result['ci_95_upper']:.3f}]")
        if len(self._all_weekly_returns) >= 10:
            overall = block_bootstrap_sharpe(np.array(self._all_weekly_returns))
            if overall:
                self.debug(f"Overall Sharpe={overall['observed_sharpe']:.3f}")
        try:
            self._price_calculator.calibrate_ensemble_weights(calibration_df)
            self.debug("Pricing GBR calibration complete")
        except Exception as e:
            self.debug(f"Pricing GBR failed: {e}")
        if len(self._strategy_history) >= StrategySelector.MIN_SAMPLES:
            try:
                self._strategy_selector.train(self._strategy_history, self)
                self.debug("StrategySelector GBR training complete")
            except Exception as e:
                self.debug(f"StrategySelector training failed: {e}")

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
            return False
        return True

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
        return bs_delta(current_spot, strike, 0.052, sigma, ttm, option_type)

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

        returns = {}
        current_spy = float(self.securities[self._underlying_symbol].price)
        sigma       = self._price_calculator._last_volatility or 0.20

        # 1-week SPY return computed from the last 6 price bars (~5 trading days)
        if self._price_history is not None and len(self._price_history) >= 6:
            prev_spy    = float(self._price_history['SPY'].iloc[-6])
            spy_return  = (current_spy - prev_spy) / prev_spy if prev_spy > 0 else 0.0
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
                # Parse option type and strike from symbol ('SPY YYMMDD C/PXXXXX')
                sym_parts   = str(contract_symbol).split()
                right_field = sym_parts[-1] if sym_parts else ''
                opt_type    = 'call' if right_field.startswith('C') else 'put'
                try:
                    strike = float(right_field[1:])
                except (ValueError, IndexError):
                    strike = current_spy
                ttm   = 14.0 / 365.25   # assume midpoint of the 14-21 DTE range
                delta = bs_delta(current_spy, strike, 0.052, sigma, ttm, opt_type)

                weekly_theta_decay = entry_price * 0.10
                delta_pnl = delta * spy_return * current_spy * qty * 100
                approx_pnl = delta_pnl - weekly_theta_decay * qty * 100
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
