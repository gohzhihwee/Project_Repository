# unit_tests_gbr.py  —  run with: pytest unit_tests_gbr.py -v
import datetime
import sys
import os
import types
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── stub missing optional package so ensemble/models import cleanly ───────────
if "nolds" not in sys.modules:
    _nolds = types.ModuleType("nolds")
    _nolds.hurst_rs = lambda x: 0.5
    sys.modules["nolds"] = _nolds


class _MockAlgo:
    def debug(self, msg):
        pass


# ──────────────────────────────────────────────────────────────────────────────
# 1. BUG 1 FIX — seed_strategy_history must be exported by `from ensemble import *`
# ──────────────────────────────────────────────────────────────────────────────

def test_seed_strategy_history_importable_via_star():
    ns = {}
    exec("from ensemble import *", ns)
    assert "seed_strategy_history" in ns, (
        "'seed_strategy_history' missing from `from ensemble import *`. "
        "The function name must NOT start with an underscore."
    )


def test_seed_strategy_history_returns_eight_valid_rows():
    from ensemble import seed_strategy_history, StrategySelector
    rows = seed_strategy_history()
    assert len(rows) == 8
    required = set(StrategySelector.FEATURE_COLS + StrategySelector.STRATEGY_NAMES)
    for i, row in enumerate(rows):
        missing = required - row.keys()
        assert not missing, f"Seed row {i} is missing columns: {missing}"
        for k, v in row.items():
            assert isinstance(v, (int, float)), (
                f"Non-numeric value in seed row {i}: {k}={v!r}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# 2. BUG 0 FIX — record_realization must match by contract_symbol, not by expiry
# ──────────────────────────────────────────────────────────────────────────────

def _add_prediction(tracker, sym="SPY 220315 C00470"):
    tracker.record_prediction(
        timestamp=datetime.datetime(2022, 1, 3),
        contract_symbol=sym,
        strike=470,
        expiry=datetime.datetime(2022, 3, 15, 16, 0),   # stored as datetime
        option_type="call",
        model_prices_dict={
            "MMAR Call": 5.0, "BS Call": 5.1,
            "Heston Call": 4.9, "Merton Call": 5.0, "Bates Call": 5.1,
        },
        actual_price=5.0,
        volatility=0.20, moneyness=1.0, ttm=0.1,
    )


def test_record_realization_works_with_string_expiry():
    from ensemble import ModelPerformanceTracker
    tracker = ModelPerformanceTracker()
    _add_prediction(tracker)
    # Old code compared datetime == "2022-03-15" → always False; bug is fixed
    tracker.record_realization("SPY 220315 C00470", 5.5, expiry="2022-03-15")
    assert tracker._prediction_history[0]["realized_price"] == 5.5


def test_record_realization_works_with_no_expiry():
    from ensemble import ModelPerformanceTracker
    tracker = ModelPerformanceTracker()
    _add_prediction(tracker)
    tracker.record_realization("SPY 220315 C00470", 6.0)
    assert tracker._prediction_history[0]["realized_price"] == 6.0


def test_record_realization_does_not_overwrite_existing():
    from ensemble import ModelPerformanceTracker
    tracker = ModelPerformanceTracker()
    _add_prediction(tracker)
    tracker.record_realization("SPY 220315 C00470", 5.5)
    tracker.record_realization("SPY 220315 C00470", 99.9)   # must be ignored
    assert tracker._prediction_history[0]["realized_price"] == 5.5


# ──────────────────────────────────────────────────────────────────────────────
# 3. mark_open_predictions — fills realized_price from securities dict
# ──────────────────────────────────────────────────────────────────────────────

def _make_tracker(n: int):
    from ensemble import ModelPerformanceTracker
    tracker = ModelPerformanceTracker()
    for i in range(n):
        sym = f"SPY 220317 C{470 + i * 5:05d}"
        tracker.record_prediction(
            timestamp=datetime.datetime(2022, 1, 3),
            contract_symbol=sym,
            strike=470 + i * 5,
            expiry=datetime.datetime(2022, 3, 17, 16),
            option_type="call",
            model_prices_dict={
                "MMAR Call": 5.0, "BS Call": 5.1,
                "Heston Call": 4.9, "Merton Call": 5.0, "Bates Call": 5.1,
            },
            actual_price=5.0 + i * 0.1,
            volatility=0.20, moneyness=1.0, ttm=0.05,
        )
    return tracker


def test_mark_open_predictions_fills_all_matching_records():
    tracker = _make_tracker(10)
    securities = {f"SPY 220317 C{470 + i * 5:05d}": 5.5 + i * 0.1 for i in range(10)}
    n_filled = tracker.mark_open_predictions(securities)
    assert n_filled == 10
    for rec in tracker._prediction_history:
        assert rec["realized_price"] is not None


def test_mark_open_predictions_skips_already_realized():
    tracker = _make_tracker(3)
    tracker._prediction_history[0]["realized_price"] = 9.0   # pre-realized
    securities = {f"SPY 220317 C{470 + i * 5:05d}": 5.5 for i in range(3)}
    n_filled = tracker.mark_open_predictions(securities)
    assert n_filled == 2
    assert tracker._prediction_history[0]["realized_price"] == 9.0


def test_calibration_df_none_before_any_realization():
    tracker = _make_tracker(10)
    assert tracker.get_calibration_dataframe() is None


def test_calibration_df_populated_after_mark_to_market():
    tracker = _make_tracker(15)
    securities = {f"SPY 220317 C{470 + i * 5:05d}": 5.5 + i * 0.1 for i in range(15)}
    tracker.mark_open_predictions(securities)
    df = tracker.get_calibration_dataframe()
    assert df is not None
    assert len(df) == 15


# ──────────────────────────────────────────────────────────────────────────────
# 4. BUG 2 FIX — calibrate_ensemble_weights trains at n=10 (threshold was 20)
# ──────────────────────────────────────────────────────────────────────────────

def _make_calibration_df(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "mmar_pred":      rng.uniform(3, 7, n),
        "bs_pred":        rng.uniform(3, 7, n),
        "heston_pred":    rng.uniform(3, 7, n),
        "merton_pred":    rng.uniform(3, 7, n),
        "bates_pred":     rng.uniform(3, 7, n),
        "volatility":     rng.uniform(0.15, 0.30, n),
        "moneyness":      rng.uniform(0.97, 1.03, n),
        "ttm":            rng.uniform(0.04, 0.08, n),
        "actual_price":   rng.uniform(3, 7, n),
        "realized_price": rng.uniform(3, 7, n),
    })


def test_pricing_gbr_does_not_train_with_9_rows():
    from ensemble import OptionPricingCalculator
    calc = OptionPricingCalculator(_MockAlgo())
    calc.calibrate_ensemble_weights(_make_calibration_df(9))
    assert not calc._meta_learner_trained, "Should NOT train with 9 rows (< 10)"


def test_pricing_gbr_trains_with_exactly_10_rows():
    from ensemble import OptionPricingCalculator
    calc = OptionPricingCalculator(_MockAlgo())
    calc.calibrate_ensemble_weights(_make_calibration_df(10))
    assert calc._meta_learner_trained, (
        "GBR must train with 10 rows; threshold was 20 (Bug 2) and should now be 10"
    )


def test_pricing_gbr_trains_with_20_rows():
    from ensemble import OptionPricingCalculator
    calc = OptionPricingCalculator(_MockAlgo())
    calc.calibrate_ensemble_weights(_make_calibration_df(20))
    assert calc._meta_learner_trained
    assert calc._meta_learner is not None
    assert calc._feature_scaler is not None


# ──────────────────────────────────────────────────────────────────────────────
# 5. StrategySelector — regime fallback and GBR training
# ──────────────────────────────────────────────────────────────────────────────

def _make_history_rows(n: int):
    from ensemble import StrategySelector
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n):
        row = {
            "volatility":      float(rng.uniform(0.10, 0.35)),
            "momentum_5d":     float(rng.uniform(-0.05, 0.05)),
            "ma_deviation_20": float(rng.uniform(-0.04, 0.04)),
            "hurst":           float(rng.uniform(0.35, 0.65)),
        }
        for s in StrategySelector.STRATEGY_NAMES:
            row[s] = float(rng.uniform(-0.01, 0.02))
        rows.append(row)
    return rows


def test_selector_untrained_initially():
    from ensemble import StrategySelector
    assert not StrategySelector().is_trained


def test_regime_fallback_high_vol_returns_bates():
    from ensemble import StrategySelector
    sel = StrategySelector()
    result = sel.select({"volatility": 0.32, "momentum_5d": 0.02,
                         "ma_deviation_20": 0.01, "hurst": 0.52})
    assert result == "Bates"


def test_regime_fallback_trending_returns_momentum():
    from ensemble import StrategySelector
    sel = StrategySelector()
    result = sel.select({"volatility": 0.20, "momentum_5d": 0.03,
                         "ma_deviation_20": 0.01, "hurst": 0.57})
    assert result == "Momentum"


def test_regime_fallback_low_hurst_returns_mean_reversion():
    from ensemble import StrategySelector
    sel = StrategySelector()
    result = sel.select({"volatility": 0.19, "momentum_5d": 0.00,
                         "ma_deviation_20": 0.01, "hurst": 0.43})
    assert result == "MeanReversion"


def test_regime_fallback_low_vol_returns_bs():
    from ensemble import StrategySelector
    sel = StrategySelector()
    result = sel.select({"volatility": 0.13, "momentum_5d": 0.01,
                         "ma_deviation_20": 0.005, "hurst": 0.50})
    assert result == "BS"


def test_selector_trains_at_min_samples():
    from ensemble import StrategySelector
    sel = StrategySelector()
    sel.train(_make_history_rows(StrategySelector.MIN_SAMPLES), _MockAlgo())
    assert sel.is_trained


def test_selector_does_not_train_below_min_samples():
    from ensemble import StrategySelector
    sel = StrategySelector()
    sel.train(_make_history_rows(StrategySelector.MIN_SAMPLES - 1), _MockAlgo())
    assert not sel.is_trained


def test_selector_trained_output_is_valid_strategy_name():
    from ensemble import StrategySelector
    sel = StrategySelector()
    sel.train(_make_history_rows(20), _MockAlgo())
    assert sel.is_trained
    result = sel.select({"volatility": 0.20, "momentum_5d": 0.01,
                         "ma_deviation_20": 0.01, "hurst": 0.50})
    assert result in StrategySelector.STRATEGY_NAMES


# ──────────────────────────────────────────────────────────────────────────────
# 6. FULL PIPELINE — predictions → MTM → calibration_df → pricing GBR trains
# ──────────────────────────────────────────────────────────────────────────────

def test_full_pipeline_pricing_gbr_trains_after_mtm():
    from ensemble import ModelPerformanceTracker, OptionPricingCalculator

    tracker = ModelPerformanceTracker()
    calc = OptionPricingCalculator(_MockAlgo())

    for i in range(20):
        sym = f"SPY 220317 C{450 + i * 5:05d}"
        tracker.record_prediction(
            timestamp=datetime.datetime(2022, 1, 3),
            contract_symbol=sym,
            strike=450 + i * 5,
            expiry=datetime.datetime(2022, 3, 17, 16),
            option_type="call",
            model_prices_dict={
                "MMAR Call": 5.0 + i * 0.1, "BS Call": 5.1 + i * 0.1,
                "Heston Call": 4.9 + i * 0.1, "Merton Call": 5.0 + i * 0.1,
                "Bates Call": 5.1 + i * 0.1,
            },
            actual_price=5.0 + i * 0.1,
            volatility=0.20, moneyness=1.0 + i * 0.01, ttm=0.05,
        )

    securities = {f"SPY 220317 C{450 + i * 5:05d}": 5.3 + i * 0.1 for i in range(20)}
    n_filled = tracker.mark_open_predictions(securities)
    assert n_filled == 20

    df = tracker.get_calibration_dataframe()
    assert df is not None
    assert len(df) == 20

    calc.calibrate_ensemble_weights(df)
    assert calc._meta_learner_trained, (
        "Pricing GBR should be trained after MTM provides 20 realized records"
    )


def test_full_pipeline_strategy_selector_trains_on_seeded_plus_one_row():
    from ensemble import StrategySelector, seed_strategy_history

    rows = seed_strategy_history()          # 8 prior rows
    assert len(rows) == 8

    real_row = {"volatility": 0.22, "momentum_5d": 0.015,
                "ma_deviation_20": 0.010, "hurst": 0.51}
    for s in StrategySelector.STRATEGY_NAMES:
        real_row[s] = 0.002
    rows.append(real_row)                   # 9 rows total ≥ MIN_SAMPLES=8

    sel = StrategySelector()
    sel.train(rows, _MockAlgo())
    assert sel.is_trained, (
        "Strategy selector must train when 8 seeded + 1 real row ≥ MIN_SAMPLES=8"
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
