# unit_tests_pricing.py — run with: pytest unit_tests_pricing.py -v
import datetime
import math
import os
import sys
import time
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

from calibration import (bs_price_vec, heston_price_vec, bates_price_vec, merton_price_vec,
                         heston_call_analytical, stratified_sample, calibrate_mmar_nls,
                         EVAL_MONEYNESS_EDGES)
from models import (mc_price_contracts, mmar_price_vec, draw_cascade_normals,
                    cascade_cell_masses, mmar_trading_time, estimate_hurst_partition,
                    bs_delta, bs_theta, N_MC_PATHS)

S, R, Q = 450.0, 0.052, 0.013
HESTON = dict(v0=0.02, kappa=2.0, theta=0.03, sigma=0.4, rho=-0.7)
JUMPS = dict(lambda_jump=1.5, mu_jump=-0.06, sigma_jump=0.08)


def _grid(n=150, seed=0):
    rng = np.random.default_rng(seed)
    K = S * rng.uniform(0.85, 1.15, n)
    T = np.round(rng.uniform(7, 90, n)) / 365.25
    return K, T, rng.random(n) < 0.5


def _parity_gap(call, put, K, T):
    return call - put - (S * np.exp(-Q * T) - K * np.exp(-R * T))


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------

def test_bs_put_call_parity_with_dividends():
    K, T, _ = _grid()
    sd = 0.15 * np.sqrt(T)
    gap = _parity_gap(bs_price_vec(S, K, T, R, Q, sd, True), bs_price_vec(S, K, T, R, Q, sd, False), K, T)
    assert np.max(np.abs(gap)) < 1e-10


def test_heston_vec_matches_scalar_reference():
    K, T, _ = _grid(60)
    ref = np.array([heston_call_analytical(S, k, R, t, **HESTON)[0] for k, t in zip(K, T)])
    vec = heston_price_vec(S, K, T, R, 0.0, **HESTON, is_call=np.ones(len(K), bool))
    assert np.max(np.abs(vec - ref)) < 1e-3


def test_heston_reduces_to_bs_without_vol_of_vol():
    K, T, c = _grid()
    h = heston_price_vec(S, K, T, R, Q, 0.02, 2.0, 0.02, 1e-3, 0.0, c)
    b = bs_price_vec(S, K, T, R, Q, math.sqrt(0.02) * np.sqrt(T), c)
    assert np.max(np.abs(h - b)) < 1e-3


def test_bates_reduces_to_merton_without_vol_of_vol():
    K, T, c = _grid()
    bt = bates_price_vec(S, K, T, R, Q, 0.02, 2.0, 0.02, 1e-3, 0.0,
                         JUMPS['lambda_jump'], JUMPS['mu_jump'], JUMPS['sigma_jump'], c)
    mt = merton_price_vec(S, K, T, R, Q, math.sqrt(0.02), JUMPS['lambda_jump'],
                          JUMPS['mu_jump'], JUMPS['sigma_jump'], c)
    assert np.max(np.abs(bt - mt)) < 2e-3


def test_merton_parity_and_zero_jump_limit():
    K, T, c = _grid()
    call = merton_price_vec(S, K, T, R, Q, 0.13, 1.5, -0.06, 0.08, np.ones(len(K), bool))
    put = merton_price_vec(S, K, T, R, Q, 0.13, 1.5, -0.06, 0.08, np.zeros(len(K), bool))
    assert np.max(np.abs(_parity_gap(call, put, K, T))) < 1e-9
    m0 = merton_price_vec(S, K, T, R, Q, 0.13, 0.0, -0.06, 0.08, c)
    assert np.max(np.abs(m0 - bs_price_vec(S, K, T, R, Q, 0.13 * np.sqrt(T), c))) < 1e-10


# ---------------------------------------------------------------------------
# 10k-path Monte Carlo vs closed forms, and speed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,params,cf", [
    ('bs', dict(sigma=0.15),
     lambda K, T, c: bs_price_vec(S, K, T, R, Q, 0.15 * np.sqrt(T), c)),
    ('merton', dict(sigma=0.13, **JUMPS),
     lambda K, T, c: merton_price_vec(S, K, T, R, Q, 0.13, 1.5, -0.06, 0.08, c)),
    ('heston', HESTON,
     lambda K, T, c: heston_price_vec(S, K, T, R, Q, **HESTON, is_call=c)),
    ('bates', dict(**HESTON, **JUMPS),
     lambda K, T, c: bates_price_vec(S, K, T, R, Q, **HESTON, lam=1.5, mu_j=-0.06, sigma_j=0.08, is_call=c)),
])
def test_mc_10k_matches_closed_form_and_is_fast(model, params, cf):
    K, T, c = _grid()
    t0 = time.perf_counter()
    mc, se = mc_price_contracts(model, params, S, R, Q, K, T, c, N_MC_PATHS, np.random.SeedSequence([7]))
    elapsed = time.perf_counter() - t0
    ref = cf(K, T, c)
    # 4 SE (SE floored at 1 cent for near-zero prices) + a 2-cent allowance for
    # the Euler discretisation of the variance process in Heston/Bates.
    tol = 4 * np.maximum(se, 0.01) + (0.02 if model in ('heston', 'bates') else 0.0)
    assert N_MC_PATHS == 10_000
    assert np.mean(np.abs(mc - ref) <= tol) >= 0.98
    assert elapsed < 1.0


# ---------------------------------------------------------------------------
# MMAR
# ---------------------------------------------------------------------------

def test_mmar_reduces_to_bs_at_half_and_no_intermittency():
    K, T, c = _grid()
    z = draw_cascade_normals(2000, np.random.default_rng(1))
    m = mmar_price_vec(S, K, T, R, Q, 0.15, 0.5, 0.0, c, z)
    assert np.max(np.abs(m - bs_price_vec(S, K, T, R, Q, 0.15 * np.sqrt(T), c))) < 1e-9


def test_mmar_put_call_parity_exact():
    K, T, _ = _grid()
    z = draw_cascade_normals(2000, np.random.default_rng(2))
    call = mmar_price_vec(S, K, T, R, Q, 0.4, 0.8, 0.6, np.ones(len(K), bool), z)
    put = mmar_price_vec(S, K, T, R, Q, 0.4, 0.8, 0.6, np.zeros(len(K), bool), z)
    assert np.max(np.abs(_parity_gap(call, put, K, T))) < 1e-9


def test_trading_time_is_unbiased_and_random():
    z = draw_cascade_normals(20_000, np.random.default_rng(3))
    T = np.array([7, 30, 90]) / 365.25
    theta = mmar_trading_time(T, cascade_cell_masses(z, 0.6))
    assert np.allclose(theta.mean(axis=1) / T, 1.0, atol=0.02)
    assert np.all(theta.std(axis=1) > 0)


def test_mmar_price_increases_with_sigma():
    K, T, c = _grid(40)
    z = draw_cascade_normals(2000, np.random.default_rng(4))
    lo = mmar_price_vec(S, K, T, R, Q, 0.3, 0.8, 0.5, c, z)
    hi = mmar_price_vec(S, K, T, R, Q, 0.4, 0.8, 0.5, c, z)
    assert np.all(hi >= lo - 1e-12)


def test_mmar_calibration_recovers_parameters():
    K, T, c = _grid(200, seed=9)
    z = draw_cascade_normals(2000, np.random.default_rng(5))
    true = dict(sigma=0.35, hurst=0.75, cascade_sigma=0.5)
    mids = mmar_price_vec(S, K, T, R, Q, true['sigma'], true['hurst'], true['cascade_sigma'], c, z)
    chain = pd.DataFrame({'strike': K, 'ttm': T, 'mid_price': mids,
                          'option_type': np.where(c, 'call', 'put')})
    fit = calibrate_mmar_nls(chain, S, R, z, q=Q, hurst_starts=(0.6, 0.5), bs_sigma=0.15)
    assert fit is not None
    assert abs(fit['hurst'] - true['hurst']) < 0.05
    assert abs(fit['sigma'] - true['sigma']) / true['sigma'] < 0.15


def test_static_mmar_keeps_hurst_fixed():
    K, T, c = _grid(80, seed=11)
    z = draw_cascade_normals(1000, np.random.default_rng(6))
    mids = mmar_price_vec(S, K, T, R, Q, 0.35, 0.75, 0.5, c, z)
    chain = pd.DataFrame({'strike': K, 'ttm': T, 'mid_price': mids,
                          'option_type': np.where(c, 'call', 'put')})
    fit = calibrate_mmar_nls(chain, S, R, z, q=Q, bs_sigma=0.15, fixed_hurst=0.6)
    assert fit is not None and fit['hurst'] == 0.6


def test_returns_hurst_is_half_for_iid_gaussian():
    est = [estimate_hurst_partition(np.random.default_rng(s).standard_normal(2000) * 0.01)['hurst']
           for s in range(10)]
    assert abs(np.mean(est) - 0.5) < 0.05


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def test_bs_theta_matches_finite_difference():
    for opt in ('call', 'put'):
        T, dt = 30 / 365.25, 1e-5
        price = lambda t: bs_price_vec(S, np.array([460.0]), np.array([t]), R, Q,
                                       np.array([0.2 * math.sqrt(t)]), opt == 'call')[0]
        fd = -(price(T) - price(T - dt)) / dt
        assert abs(bs_theta(S, 460.0, R, 0.2, T, opt, Q) - fd) < 1e-2
    assert 0 < bs_delta(S, 450.0, R, 0.2, 0.1, 'call', Q) < 1


# ---------------------------------------------------------------------------
# Evaluation machinery
# ---------------------------------------------------------------------------

def _chain(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    K = S * rng.uniform(0.8, 1.2, n)
    days = rng.integers(5, 100, n)
    return pd.DataFrame({'strike': K, 'ttm': days / 365.25, 'mid_price': 1.0,
                         'option_type': np.where(rng.random(n) < 0.5, 'call', 'put')})


def test_stratified_sample_fills_cells_and_is_deterministic():
    df = _chain()
    a = stratified_sample(df, S, 15, seed=20240101)
    b = stratified_sample(df, S, 15, seed=20240101)
    assert len(a) == 150
    pd.testing.assert_frame_equal(a, b)
    ks = a['strike'] / S
    assert ks.min() >= EVAL_MONEYNESS_EDGES[0] and ks.max() <= EVAL_MONEYNESS_EDGES[-1]
    for lo, hi in zip(EVAL_MONEYNESS_EDGES[:-1], EVAL_MONEYNESS_EDGES[1:]):
        assert ((ks >= lo) & (ks <= hi)).sum() >= 15


def test_dm_date_clustering_controls_size_under_common_shocks():
    from ensemble import _dm_test
    rng = np.random.default_rng(0)
    rej_contract = rej_date = 0
    n_sims, n_dates, per_date = 200, 40, 50
    for _ in range(n_sims):
        dates = np.repeat(np.arange(n_dates), per_date)
        d = np.repeat(rng.standard_normal(n_dates), per_date) + 0.2 * rng.standard_normal(n_dates * per_date)
        res = _dm_test(d, dates)
        rej_contract += res['p_value_contract'] < 0.05
        rej_date += res['p_value'] < 0.05
    assert rej_contract / n_sims > 0.3        # pooled test over-rejects a true null
    assert rej_date / n_sims < 0.12           # date-clustered test holds its size


def _record(tracker, sym, day, mid=5.0):
    tracker.record_prediction(
        timestamp=datetime.datetime.combine(day, datetime.time(10, 30)),
        contract_symbol=sym, strike=450, expiry=day + datetime.timedelta(days=30),
        option_type='call', model_prices_dict={'BS Call': 5.1, 'MMAR Call': 5.0},
        actual_price=mid, volatility=0.2, moneyness=1.0, ttm=30 / 365.25)


def test_tracker_is_uncapped_and_realizes_only_from_given_date():
    from ensemble import ModelPerformanceTracker
    tracker = ModelPerformanceTracker()
    d1, d2 = datetime.date(2024, 1, 8), datetime.date(2024, 1, 16)
    for i in range(1200):
        _record(tracker, f"SPY240208C{i:08d}", d1)
    _record(tracker, "SPY240110C00450000", d2)          # expires before next week
    assert len(tracker._prediction_history) == 1201
    today_chain = {f"SPY240208C{i:08d}": 6.0 for i in range(1200)}
    n = tracker.mark_open_predictions(today_chain, from_date=d1)
    assert n == 1200
    assert tracker._prediction_history[-1]['realized_price'] is None


def test_same_day_metrics_use_the_day_t_mid():
    from ensemble import ModelPerformanceTracker
    tracker = ModelPerformanceTracker()
    for i in range(20):
        _record(tracker, f"SPY{i}", datetime.date(2024, 1, 8) + datetime.timedelta(days=7 * (i % 4)), mid=5.0)
    m = tracker.get_model_accuracy_metrics()
    assert m['BS']['RMSE'] == pytest.approx(0.1)
    assert m['MMAR']['RMSE'] == pytest.approx(0.0)
    assert 'RMSE_next' not in m['BS']


def test_pairwise_dm_covers_every_pair_and_orients_sign():
    from ensemble import ModelPerformanceTracker
    tracker = ModelPerformanceTracker()
    rng = np.random.default_rng(1)
    for i in range(400):
        day = datetime.date(2024, 1, 8) + datetime.timedelta(days=7 * (i % 20))
        tracker.record_prediction(
            timestamp=datetime.datetime.combine(day, datetime.time(10, 30)),
            contract_symbol=f"SPY{i}", strike=450, expiry=day + datetime.timedelta(days=30),
            option_type='call',
            model_prices_dict={'BS Call': 5.0 + rng.normal(0, 1.0), 'Heston Call': 5.0 + rng.normal(0, 0.3),
                               'MMAR Call': 5.0 + rng.normal(0, 0.6)},
            actual_price=5.0, volatility=0.2, moneyness=1.0, ttm=30 / 365.25)
    pw = tracker.run_pairwise_dm_tests('same_day')
    assert len(pw) == 3                                   # BS, MMAR, HESTON → 3 pairs
    row = pw[(pw['model_a'] == 'MMAR') & (pw['model_b'] == 'HESTON')].iloc[0]
    assert row['DM_stat'] > 0 and row['favors'] == 'HESTON' and row['p_value'] < 0.05
    assert row['n_dates'] == 20


def test_portfolio_metrics_use_excess_returns():
    from metrics import compute_portfolio_metrics
    rets = np.array([0.002, 0.001, 0.003, 0.0, 0.002, 0.001] * 8)
    values = 100_000 * np.cumprod(1 + rets)
    curve = pd.DataFrame({'value': values}, index=pd.date_range('2024-01-01', periods=len(rets), freq='W'))
    rf = 0.052
    m = compute_portfolio_metrics(curve, list(rets), [], risk_free_rate=rf)
    rf_w = (1 + rf) ** (1 / 52) - 1
    expected = (rets.mean() - rf_w) / rets.std(ddof=1) * np.sqrt(52)
    assert m['sharpe'] == pytest.approx(round(expected, 3))
    assert m['sharpe_raw'] > m['sharpe']
    spy = pd.Series(rets * 2)
    m2 = compute_portfolio_metrics(curve, list(rets), [], spy_returns=spy, risk_free_rate=rf)
    # r = 0.5·r_spy exactly → β = 0.5 and Jensen's α = (r̄ − r_f) − 0.5(2r̄ − r_f) = −0.5·r_f
    assert m2['beta'] == pytest.approx(0.5)
    assert m2['alpha_pct'] == pytest.approx(round(-0.5 * rf_w * 52 * 100, 2))


# ---------------------------------------------------------------------------
# Contract parsing and expiry settlement
# ---------------------------------------------------------------------------

def test_parse_contract_osi_and_synthetic():
    from main import OptionsArbitrageAlgorithm
    p = OptionsArbitrageAlgorithm._parse_contract
    assert p("SPY   231020P00425000") == ('put', 425.0, datetime.date(2023, 10, 20))
    assert p("SPY 231020 C00425") == ('call', 425.0, datetime.date(2023, 10, 20))
    assert p("garbage") is None


def _algo_with_position(qty, sym="SPY   240119C00470000"):
    from main import OptionsArbitrageAlgorithm, Portfolio, _PortfolioPosition
    algo = OptionsArbitrageAlgorithm.__new__(OptionsArbitrageAlgorithm)
    algo._current_time = datetime.datetime(2024, 1, 22, 10, 30)
    algo._portfolio = Portfolio(100_000)
    algo._portfolio._positions[sym] = _PortfolioPosition(qty, 2.0)
    algo._portfolio._prices[sym] = 2.0
    algo._trade_log = []
    idx = pd.to_datetime(['2024-01-18', '2024-01-19', '2024-01-22'])
    algo._all_prices = pd.DataFrame({'SPY': [476.0, 482.5, 484.0], 'SPY_TR': [470.0, 476.0, 478.0]}, index=idx)
    return algo


@pytest.mark.parametrize("qty,cash_change", [(2, 2 * 12.5 * 100), (-1, -12.5 * 100)])
def test_expired_itm_positions_settle_at_intrinsic(qty, cash_change):
    algo = _algo_with_position(qty)
    algo._settle_expired_positions(datetime.date(2024, 1, 22), 484.0)
    assert algo._portfolio._positions == {}
    assert algo._portfolio._cash == pytest.approx(100_000 + cash_change)   # S(expiry)=482.5, K=470
    assert algo._trade_log[0]['settled']


def test_unexpired_position_is_kept_and_dte_is_real():
    algo = _algo_with_position(1, sym="SPY   240216C00470000")
    algo._settle_expired_positions(datetime.date(2024, 1, 22), 484.0)
    assert "SPY   240216C00470000" in algo._portfolio._positions
    assert algo._days_to_expiry("SPY   240216C00470000") == 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
