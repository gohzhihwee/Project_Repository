# unit_tests_calibration.py — run with: pytest unit_tests_calibration.py -v
import datetime
import math
import sys
import os
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Stub nolds so ensemble/models import cleanly without the package installed
if "nolds" not in sys.modules:
    _nolds = types.ModuleType("nolds")
    _nolds.hurst_rs = lambda x: 0.5
    sys.modules["nolds"] = _nolds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bs_call(S, K, r, sigma, T):
    from scipy.stats import norm
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sq
    d2 = d1 - sq
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def _bs_put(S, K, r, sigma, T):
    call = _bs_call(S, K, r, sigma, T)
    return call - S + K * math.exp(-r * T)


def _make_chain(spot=450.0, sigma=0.20, r=0.02,
                strikes=None, ttms=None) -> pd.DataFrame:
    """Synthetic option chain priced at a known flat vol."""
    if strikes is None:
        strikes = [430, 440, 445, 450, 455, 460, 470]
    if ttms is None:
        ttms = [30 / 365.25, 60 / 365.25]
    rows = []
    for T in ttms:
        expiry = datetime.date.today() + datetime.timedelta(days=int(T * 365.25))
        for K in strikes:
            for otype, fn in [('call', _bs_call), ('put', _bs_put)]:
                mid = fn(spot, K, r, sigma, T)
                if mid < 0.01:
                    continue
                rows.append({
                    'contract_symbol': f"SPY{expiry.strftime('%y%m%d')}{'C' if otype=='call' else 'P'}{int(K*1000):08d}",
                    'strike':      K,
                    'expiry':      expiry,
                    'option_type': otype,
                    'bid_price':   round(mid * 0.99, 4),
                    'ask_price':   round(mid * 1.01, 4),
                    'last_price':  round(mid, 4),
                    'mid_price':   round(mid, 4),
                    'ttm':         T,
                    'volume':      100,
                    'open_interest': 500,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. filter_chain_for_calibration
# ---------------------------------------------------------------------------

class TestFilterChain:

    def test_keeps_valid_contracts(self):
        from calibration import filter_chain_for_calibration
        df = _make_chain()
        filtered = filter_chain_for_calibration(df, spot=450.0)
        assert len(filtered) > 0

    def test_removes_deep_otm(self):
        from calibration import filter_chain_for_calibration
        df = _make_chain(strikes=[200, 800])   # moneyness 0.44 and 1.78
        filtered = filter_chain_for_calibration(df, spot=450.0)
        assert filtered.empty

    def test_removes_short_ttm(self):
        from calibration import filter_chain_for_calibration
        df = _make_chain(ttms=[3 / 365.25])    # 3 days — below 7-day floor
        filtered = filter_chain_for_calibration(df, spot=450.0)
        assert filtered.empty

    def test_removes_long_ttm(self):
        from calibration import filter_chain_for_calibration
        df = _make_chain(ttms=[120 / 365.25])  # 120 days — above 90-day ceiling
        filtered = filter_chain_for_calibration(df, spot=450.0)
        assert filtered.empty

    def test_removes_zero_mid_price(self):
        from calibration import filter_chain_for_calibration
        df = _make_chain()
        df['mid_price'] = 0.0
        filtered = filter_chain_for_calibration(df, spot=450.0)
        assert filtered.empty


# ---------------------------------------------------------------------------
# 2. calibrate_bs_iv  — recovers known flat vol from synthetic chain
# ---------------------------------------------------------------------------

class TestCalibrateBsIv:

    def test_recovers_known_vol(self):
        from calibration import calibrate_bs_iv
        TRUE_SIGMA = 0.20
        df = _make_chain(sigma=TRUE_SIGMA)
        recovered = calibrate_bs_iv(df, spot=450.0, r=0.02)
        assert abs(recovered - TRUE_SIGMA) < 0.03, (
            f"Expected ~{TRUE_SIGMA}, got {recovered:.4f}")

    def test_higher_vol_chain(self):
        from calibration import calibrate_bs_iv
        TRUE_SIGMA = 0.35
        df = _make_chain(sigma=TRUE_SIGMA)
        recovered = calibrate_bs_iv(df, spot=450.0, r=0.02)
        assert abs(recovered - TRUE_SIGMA) < 0.05

    def test_fallback_on_empty_chain(self):
        from calibration import calibrate_bs_iv
        # Pass a properly-columned but empty DataFrame to exercise the early-return path
        empty = pd.DataFrame(columns=['strike', 'ttm', 'mid_price', 'option_type'])
        result = calibrate_bs_iv(empty, spot=450.0, r=0.02)
        assert result == 0.20

    def test_returns_positive_float(self):
        from calibration import calibrate_bs_iv
        df = _make_chain()
        result = calibrate_bs_iv(df, spot=450.0, r=0.02)
        assert isinstance(result, float) and result > 0


# ---------------------------------------------------------------------------
# 3. calibrate_merton_mle
# ---------------------------------------------------------------------------

class TestCalibrateMertonMle:

    def test_returns_dict_on_sufficient_data(self):
        from calibration import calibrate_merton_mle
        rng = np.random.default_rng(42)
        log_returns = rng.normal(0.0, 0.01, 60)
        result = calibrate_merton_mle(log_returns)
        assert result is not None
        assert set(result.keys()) == {
            'lambda_jump', 'mu_jump', 'sigma_jump', 'diffusion_sigma'}

    def test_returns_none_on_insufficient_data(self):
        from calibration import calibrate_merton_mle
        result = calibrate_merton_mle(np.zeros(20))  # < 30 required
        assert result is None

    def test_params_are_in_valid_range(self):
        from calibration import calibrate_merton_mle
        rng = np.random.default_rng(7)
        # Mix of normal days + occasional jump
        log_returns = np.concatenate([
            rng.normal(0.0, 0.01, 50),
            rng.normal(-0.04, 0.03, 5),   # simulated jump days
        ])
        result = calibrate_merton_mle(log_returns)
        assert result is not None
        assert result['lambda_jump'] >= 0
        assert result['sigma_jump']  > 0
        assert result['diffusion_sigma'] > 0


# ---------------------------------------------------------------------------
# 4. _parse_osi_symbol (data_loader)
# ---------------------------------------------------------------------------

class TestParseOsiSymbol:

    def test_standard_symbol(self):
        from data_loader import _parse_osi_symbol
        root, expiry, otype, strike = _parse_osi_symbol('SPY231215C00450000')
        assert root   == 'SPY'
        assert expiry == datetime.date(2023, 12, 15)
        assert otype  == 'call'
        assert abs(strike - 450.0) < 0.001

    def test_symbol_with_spaces(self):
        from data_loader import _parse_osi_symbol
        result = _parse_osi_symbol('SPY   231215C00450000')
        assert result is not None
        assert result[0] == 'SPY'

    def test_put_right(self):
        from data_loader import _parse_osi_symbol
        _, _, otype, _ = _parse_osi_symbol('SPY231215P00420000')
        assert otype == 'put'

    def test_fractional_strike(self):
        from data_loader import _parse_osi_symbol
        _, _, _, strike = _parse_osi_symbol('SPY231215C00450500')
        assert abs(strike - 450.5) < 0.001

    def test_invalid_returns_none(self):
        from data_loader import _parse_osi_symbol
        assert _parse_osi_symbol('INVALID') is None
        assert _parse_osi_symbol('') is None
        assert _parse_osi_symbol('SPY231215X00450000') is None  # X is not C/P


# ---------------------------------------------------------------------------
# 5. DatabentoCacheLoader._parse_chain  (no filesystem access needed)
# ---------------------------------------------------------------------------

def _make_raw_df(spot=450.0, n_contracts=4, n_bars_each=3) -> pd.DataFrame:
    """Simulate a raw Databento CBBO-1m DataFrame with multiple bars per contract."""
    rows = []
    base = datetime.datetime(2023, 12, 15, 19, 40)  # 15:40 ET as UTC
    expiry = '240315'
    symbols = [
        f"SPY{expiry}C{int((450 + i * 5) * 1000):08d}"
        for i in range(n_contracts)
    ]
    for sym in symbols:
        for b in range(n_bars_each):
            ts = base + datetime.timedelta(minutes=b)
            mid = 5.0 + b * 0.10   # price rises each bar — last bar is highest
            rows.append({
                'ts_recv':   ts.isoformat() + 'Z',
                'ts_event':  ts.isoformat() + 'Z',
                'symbol':    sym,
                'bid_px_00': round(mid - 0.05, 4),
                'ask_px_00': round(mid + 0.05, 4),
                'bid_sz_00': 10,
                'ask_sz_00': 10,
            })
    return pd.DataFrame(rows)


class TestParseChain:

    def _loader(self):
        import pathlib, tempfile
        from data_loader import DatabentoCacheLoader
        tmp = pathlib.Path(tempfile.mkdtemp())
        return DatabentoCacheLoader(csv_dir=tmp, cache_dir=tmp, symbol='SPY')

    def test_eod_snapshot_takes_last_bar(self):
        loader = self._loader()
        raw = _make_raw_df(n_bars_each=3)
        date = datetime.date(2023, 12, 15)
        result = loader._parse_chain(raw, date)
        assert result is not None
        # Last bar has bid = 5.20-0.05 = 5.15, ask = 5.20+0.05 = 5.25 → mid=5.20
        assert all(result['mid_price'] > 5.10), (
            "Expected last-bar mid prices (~5.20), got lower values — "
            "EOD snapshot is not taking the last record per contract.")

    def test_filters_zero_bid(self):
        loader = self._loader()
        raw = _make_raw_df(n_bars_each=1)
        raw.loc[0, 'bid_px_00'] = 0.0
        date = datetime.date(2023, 12, 15)
        result = loader._parse_chain(raw, date)
        # The zero-bid contract should be dropped
        n_input  = raw['symbol'].nunique()
        n_output = len(result) if result is not None else 0
        assert n_output < n_input

    def test_filters_expired_contracts(self):
        loader = self._loader()
        raw = _make_raw_df(n_bars_each=1)
        # Use a date AFTER the contract expiry (2024-03-15)
        future_date = datetime.date(2024, 4, 1)
        result = loader._parse_chain(raw, future_date)
        assert result is None or result.empty

    def test_output_columns_present(self):
        loader = self._loader()
        raw = _make_raw_df(n_bars_each=1)
        date = datetime.date(2023, 12, 15)
        result = loader._parse_chain(raw, date)
        assert result is not None
        required = {'contract_symbol', 'strike', 'expiry',
                    'option_type', 'bid_price', 'ask_price', 'mid_price'}
        assert required.issubset(set(result.columns))

    def test_non_spy_symbols_excluded(self):
        loader = self._loader()
        raw = _make_raw_df(n_bars_each=1)
        # Inject a non-SPY row
        raw.loc[len(raw)] = {
            'ts_recv':   '2023-12-15T19:40:00Z',
            'ts_event':  '2023-12-15T19:40:00Z',
            'symbol':    'QQQ231215C00380000',
            'bid_px_00': 4.0,
            'ask_px_00': 4.2,
            'bid_sz_00': 5,
            'ask_sz_00': 5,
        }
        date = datetime.date(2023, 12, 15)
        result = loader._parse_chain(raw, date)
        assert result is not None
        assert all(s.startswith('SPY') for s in result['contract_symbol'])


# ---------------------------------------------------------------------------
# 6. t-1 calibration ordering — guard against regression
# ---------------------------------------------------------------------------

def test_prev_calib_df_updated_after_rebalance():
    """
    Verify the t-1 protocol: _prev_calib_df must not be overwritten with
    today's data before _rebalance() is called.  We check this structurally
    by inspecting the source of run_backtest.
    """
    import inspect
    from main import OptionsArbitrageAlgorithm
    src = inspect.getsource(OptionsArbitrageAlgorithm.run_backtest)

    # The assignment to _prev_calib_df must appear AFTER the _rebalance() call
    idx_rebalance = src.find('self._rebalance()')
    idx_prev_upd  = src.find('self._prev_calib_df = calib_df')
    assert idx_rebalance != -1, "_rebalance() not found in run_backtest"
    assert idx_prev_upd  != -1, "_prev_calib_df assignment not found in run_backtest"
    assert idx_prev_upd > idx_rebalance, (
        "t-1 violation: _prev_calib_df is updated BEFORE _rebalance() runs. "
        "Calibration uses same-day (t) data instead of previous-day (t-1).")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
