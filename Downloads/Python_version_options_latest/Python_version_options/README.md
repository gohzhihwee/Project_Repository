# Options Arbitrage Ensemble Backtester

A standalone Python backtester that prices SPY options with five stochastic models, combines them via a Gradient Boosting meta-learner (the pricing GBR), and selects among nine trading strategies using a second GBR trained on live regime features. It runs without the QuantConnect (LEAN) runtime — a local stub (`AlgorithmImports.py`) provides the QC API surface.

---

## File Map

| File | Role |
|---|---|
| `main.py` | Backtester entry point, portfolio shims, scheduling loop |
| `ensemble.py` | GBR meta-learner, strategy selector, prediction tracker |
| `models.py` | Five option-pricing models + concurrent dispatcher |
| `modelstrats.py` | Six model-based trading strategy classes |
| `convmethods.py` | Three conventional trading strategy classes |
| `metrics.py` | Pure metric functions (Sharpe, Sortino, MAPE, DM-test, …) |
| `visualization.py` | End-of-run PNG chart suite (9 charts) |
| `AlgorithmImports.py` | Stub that exposes `OptionRight`, `Resolution`, etc. |
| `unit_tests_gbr.py` | pytest suite verifying the GBR calibration pipeline |

---

## High-Level Architecture

```
SPY daily prices (yfinance)
        |
        v
  run_backtest()
  |
  +-- initialize()              set up portfolio, models, seed strategy history
  |
  +-- Daily loop (every trading day)
        |
        +-- generate_option_chain()   build synthetic BS chain with VRP skew
        +-- update securities dict    accumulate ALL contract mid-prices
        +-- _monitor_margin_health()  daily margin check / warning
        |
        +-- [Monday only, post-warmup]
              |
              v
          _rebalance()
              |
              +-- _update_price_history()
              +-- _selective_close_positions()    near-expiry / P&L threshold
              +-- _compute_hypo_returns()         log last week's strategy performance
              +-- _attempt_ensemble_calibration() train/retrain both GBRs
              +-- _prepare_option_data()          price options, record predictions
              +-- strategy.generate_signals()     active strategy emits buy/write/sell
              +-- _execute_signal()               place orders, delta-hedge SPY
```

---

## Execution Flow — Step by Step

### 1. Startup

```
run_backtest()
  └─ yf.download("SPY", 2021-06-01 → 2024-12-31)
  └─ _start_date  = 2022-01-01
     _warmup_cutoff = 2021-12-02   (30-day warmup)
  └─ initialize()
       ├─ Portfolio(cash=100 000)
       ├─ OptionPricingCalculator(self)
       │     └─ ModelPerformanceTracker()   empty prediction history
       ├─ StrategySelector()                untrained
       └─ _strategy_history = seed_strategy_history()   8 synthetic prior rows
```

### 2. Warmup (Dec 2 – Dec 31 2021, daily)

```
For each trading day:
  generate_option_chain()  →  adds contract prices to _securities._data
  _monitor_margin_health()
  [no _rebalance() — date < _start_date]
```

During warmup, option prices accumulate in `_securities._data` but no predictions are recorded and no trades are placed.

### 3. Week 1 — First Real Rebalance (Monday Jan 3 2022)

```
_rebalance()
  ├─ days_since calibration = 2  (<7)  →  calibration SKIPPED
  ├─ _prepare_option_data()
  │     For each of 20 contracts (5 strikes × 2 rights × 2 expiries):
  │       calculate_model_prices()  →  MMAR, BS, Heston, Merton, Bates prices
  │       record_prediction()       →  stores {contract, model_prices, realized_price=None}
  └─ strategy.generate_signals()   →  regime fallback selects strategy
     _execute_signal()             →  places first trades
```

After week 1: `_prediction_history` holds 20 records, all with `realized_price = None`.

### 4. Week 2 — First GBR Calibration (Monday Jan 10 2022)

```
_rebalance()
  ├─ _compute_hypo_returns()   →  appends 1 real row to _strategy_history (total: 9)
  ├─ days_since calibration = 9  (≥7)
  │
  └─ _attempt_ensemble_calibration()
        ├─ _mark_to_market_realizations()
        │     mark_open_predictions(_securities._data)
        │       For each unrealized prediction:
        │         looks up contract_symbol in _securities._data
        │         fills realized_price  →  20 records realized
        │
        ├─ get_calibration_dataframe()  →  20-row DataFrame
        │
        ├─ calibrate_ensemble_weights()   ← PRICING GBR TRAINS HERE
        │     .dropna()  →  remove NaN model rows
        │     GradientBoostingRegressor.fit(X_scaled, realized_price, sample_weight)
        │
        └─ StrategySelector.train()       ← STRATEGY GBR TRAINS HERE
              MultiOutputRegressor(GBR).fit(features, strategy_returns, sample_weight)
```

From week 2 onwards both GBRs are live. All subsequent rebalances retrain them weekly on a rolling window of the last 52 observations.

### 5. Every Subsequent Week

```
_rebalance()
  ├─ _selective_close_positions()
  │     Close if:  DTE ≤ 3  OR  gain ≥ 50%  OR  loss ≥ 80%
  │     record_realization(symbol, exit_price)   →  fills realized_price immediately
  │
  ├─ _attempt_ensemble_calibration()
  │     MTM fills any still-open predictions from securities prices
  │     Pricing GBR retrains on rolling window (last 1000 records, ≥10 valid after .dropna())
  │     Strategy GBR retrains on rolling window (last 52 rows, ≥8 rows)
  │
  ├─ _prepare_option_data()   records fresh predictions for this week's chain
  │
  └─ StrategySelector.select(features)   picks best strategy
       If trained: GBR argmax over 9 predicted returns
       If untrained: _regime_based_fallback() rule-based heuristic
```

---

## The GBR Calibration Pipeline

The data must flow through five stages before either GBR can train.

```
Stage 1: Prediction recording
  _prepare_option_data()
    └─ record_prediction(contract_symbol, model_prices, actual_price, volatility, moneyness, ttm)
         Appends to _prediction_history with realized_price = None

Stage 2: Realization (two paths)
  Path A — active close:
    _execute_signal() or _selective_close_positions()
      └─ record_realization(contract_symbol, exit_price)
           Finds the most-recent unrealized record for this symbol → sets realized_price

  Path B — mark-to-market (every rebalance, even if no position closed):
    _mark_to_market_realizations()
      └─ mark_open_predictions(_securities._data)
           For each unrealized record, looks up contract_symbol in the securities dict
           → sets realized_price to the current mid-price

Stage 3: Calibration DataFrame
  get_calibration_dataframe()
    Returns all records where realized_price is not None
    Columns: mmar_pred, bs_pred, heston_pred, merton_pred, bates_pred,
             volatility, moneyness, ttm, actual_price, realized_price

Stage 4: Pricing GBR training
  calibrate_ensemble_weights(calibration_df)
    .dropna() removes rows with any NaN model price
    Requires ≥10 valid rows
    Exponential decay weights (recent observations weighted more)
    Trains GradientBoostingRegressor to predict realized_price from model features

Stage 5: Strategy GBR training
  StrategySelector.train(_strategy_history, algo)
    Requires ≥8 rows (MIN_SAMPLES)
    Rolling window: last 52 rows (MAX_HISTORY)
    Exponential decay weights
    MultiOutputRegressor(GBR) → predicts hypothetical return for each of 9 strategies
    select() returns the strategy name with the highest predicted return
```

---

## Mathematical Detail: How the GBRs Fit

There are two independent Gradient Boosting Regressors. They share the same boosting algorithm but differ in what they learn, how many outputs they produce, and what their inputs are.

---

### 1. The Boosting Algorithm

Both GBRs are instances of scikit-learn's `GradientBoostingRegressor`, which fits an **additive model of decision trees** by minimising a loss function in function space via gradient descent.

**The additive model:**

$$F_M(\mathbf{x}) = F_0(\mathbf{x}) + \eta \sum_{m=1}^{M} h_m(\mathbf{x})$$

where:
- $F_0(\mathbf{x}) = \bar{y}$ — constant initialisation (mean of training targets)
- $h_m(\mathbf{x})$ — the $m$-th shallow regression tree (weak learner)
- $\eta$ — learning rate (shrinkage parameter)
- $M = 100$ — number of boosting rounds (`n_estimators`)

Both GBRs use **squared-error (L2) loss**:

$$\mathcal{L}(y, F) = \tfrac{1}{2}(y - F(\mathbf{x}))^2$$

**At each round $m$, the algorithm does three things:**

**Step 1 — Compute pseudo-residuals (negative gradient of the loss).**  
For L2 loss the negative gradient is simply the residual:

$$r_i^{(m)} = -\left.\frac{\partial \mathcal{L}(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right|_{F = F_{m-1}} = y_i - F_{m-1}(\mathbf{x}_i)$$

**Step 2 — Fit a weighted regression tree $h_m$ to the pseudo-residuals.**  
Each node split is chosen to minimise the **weighted MSE** of the residuals in the resulting child nodes. With sample weights $w_i$, the weighted mean and impurity in a leaf region $R$ are:

$$\hat{c}_R = \frac{\sum_{i \in R} w_i \, r_i^{(m)}}{\sum_{i \in R} w_i}, \qquad \text{Impurity}(R) = \frac{\sum_{i \in R} w_i \left(r_i^{(m)} - \hat{c}_R\right)^2}{\sum_{i \in R} w_i}$$

The best split $(j^*, t^*)$ on feature $j$ at threshold $t$ maximises the weighted impurity reduction:

$$\Delta(j, t) = \text{Impurity}(R) - \frac{|R_L|}{|R|}\,\text{Impurity}(R_L) - \frac{|R_R|}{|R|}\,\text{Impurity}(R_R)$$

Tree depth is capped at `max_depth` and each leaf must contain at least `min_samples_leaf` training points.

**Step 3 — Update the ensemble.**  
With stochastic subsampling (80% of rows drawn without replacement each round):

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

After $M = 100$ rounds the final model is:

$$F_{100}(\mathbf{x}) = \bar{y} + \eta \sum_{m=1}^{100} h_m(\mathbf{x})$$

---

### 2. Input Standardisation

Both GBRs receive standardised inputs. A `StandardScaler` is fit on the training data and stored alongside the model so the same transformation is applied at inference time.

$$\tilde{x}_j = \frac{x_j - \hat{\mu}_j}{\hat{\sigma}_j}$$

where $\hat{\mu}_j$ and $\hat{\sigma}_j$ are the feature-wise mean and standard deviation over the training window. This is important because the tree-splitting criterion is scale-invariant, but the regularisation parameter `min_samples_leaf` and the exponential decay weights (below) interact better with normalised inputs.

---

### 3. Exponential Decay Sample Weights

Both GBRs weight recent observations more heavily than older ones. Given $n$ training rows ordered oldest-to-newest:

$$w_i^{\text{raw}} = \exp\!\left(-2 + \frac{2i}{n-1}\right) = \exp\!\left(\text{linspace}(-2,\, 0,\, n)_i\right), \quad i = 0, \ldots, n-1$$

These are normalised so that the weights sum to $n$ (scikit-learn's convention for sample weights in a regression context):

$$w_i = \frac{w_i^{\text{raw}}}{\sum_{j=0}^{n-1} w_j^{\text{raw}}} \cdot n$$

The ratio of the newest to the oldest weight is $e^2 \approx 7.4$, meaning the most recent observation counts roughly **7× more** than the oldest one in the window. The decay is smooth and exponential; data from 26 weeks ago receives approximately $e^{-1} \approx 0.37$ of the weight of the current week.

---

### 4. Pricing GBR (`OptionPricingCalculator._meta_learner`)

**Purpose:** Learn how to combine the five model prices into a single price estimate that is closest to the option's realised mid-price.

**Training data construction:**

Each week, $\sim$20 option price predictions are recorded (5 strikes × 2 rights × 2 expiries). After MTM or position close, `realized_price` is filled. The calibration DataFrame has one row per realised option and the following columns:

| Column | Description |
|---|---|
| `mmar_pred` | MMAR Monte Carlo call or put price |
| `bs_pred` | Black-Scholes Monte Carlo price |
| `heston_pred` | Heston stochastic-vol price |
| `merton_pred` | Merton jump-diffusion price |
| `bates_pred` | Bates (stoch-vol + jumps) price |
| `volatility` | Realised 20-day annualised vol at prediction time |
| `moneyness` | $K / S_0$ — ratio of strike to spot |
| `ttm` | Time-to-maturity in years |
| `realized_price` | **Target**: actual mid-price when the option was marked or closed |

Rows with any NaN model price are dropped before fitting. Requires $n \geq 10$ valid rows.

**Feature matrix and target:**

$$\mathbf{X} \in \mathbb{R}^{n \times 8} = \left[\tilde{\mathbf{p}}_{\text{MMAR}},\ \tilde{\mathbf{p}}_{\text{BS}},\ \tilde{\mathbf{p}}_{\text{Heston}},\ \tilde{\mathbf{p}}_{\text{Merton}},\ \tilde{\mathbf{p}}_{\text{Bates}},\ \tilde{\boldsymbol{\sigma}},\ \widetilde{K/S},\ \tilde{\boldsymbol{\tau}}\right], \qquad \mathbf{y} \in \mathbb{R}^n$$

where tildes denote standardised values.

**What the GBR learns:**

$$\hat{p}^{\text{mixed}} = F_{100}(\tilde{\mathbf{x}}) \approx p^{\text{realized}}$$

The model effectively learns a **data-driven interpolation** over the five model prices, weighted by the current volatility regime, moneyness, and time-to-maturity. In a low-vol, near-ATM regime the weights may concentrate on BS; in high-vol regimes with jumps they may shift toward Bates or Merton. The GBR discovers this mapping from the historical record without the programmer specifying it.

**Adaptive hyperparameters** (set in `calibrate_ensemble_weights`):

| Condition | `max_depth` | `learning_rate` | `min_samples_leaf` |
|---|---|---|---|
| $n < 30$ | 2 | 0.05 | $\max(3,\ n / 10)$ |
| $n \geq 30$ | 3 | 0.10 | $\max(3,\ n / 10)$ |

Shallow trees and a low learning rate when data is scarce prevent the model from overfitting to the small early sample. As the window grows, the model is allowed to fit slightly more complex patterns.

**At inference** (`_gbr_or_fallback` inside `calculate_model_prices`):

$$\hat{p}^{\text{mixed}} = F_{100}\!\left(\text{StandardScaler.transform}(\mathbf{x}_{\text{new}})\right)$$

The result is clipped at 0 and used as the 'Mixed' price that `MixedStrategy` compares against the chain's implied price.

If the GBR is not yet trained, the fallback is an equal-weight average of the available model prices:

$$\hat{p}^{\text{mixed}}_{\text{fallback}} = \frac{1}{|\mathcal{A}|} \sum_{m \in \mathcal{A}} p_m, \quad \mathcal{A} = \{\text{available models}\}$$

---

### 5. Strategy Selector GBR (`StrategySelector._multi_gbr`)

**Purpose:** Given the current market regime, predict which of the nine strategies would have produced the highest return *last week*, then trade with that strategy *this week*.

This is a **multi-output regression** problem: one target per strategy, fitted simultaneously using scikit-learn's `MultiOutputRegressor`, which trains nine independent GBR models sharing the same feature matrix.

**Feature matrix:**

$$\mathbf{X} \in \mathbb{R}^{n \times 4} = \left[\tilde{\boldsymbol{\sigma}},\ \widetilde{\Delta S_{5d}},\ \widetilde{\Delta\text{MA}_{20d}},\ \tilde{H}\right]$$

| Feature | Symbol | Description |
|---|---|---|
| `volatility` | $\sigma$ | 20-day realised annualised volatility |
| `momentum_5d` | $\Delta S_{5d}$ | $(S_t - S_{t-5}) / S_{t-5}$ — 5-day price return |
| `ma_deviation_20` | $\Delta\text{MA}_{20d}$ | $(S_t - \overline{S}_{20}) / \overline{S}_{20}$ — deviation from 20-day MA |
| `hurst` | $H$ | Hurst exponent estimated by R/S analysis on the SPY series |

**Target matrix:**

$$\mathbf{Y} \in \mathbb{R}^{n \times 9}$$

Each column $s$ is the **hypothetical weekly return** that strategy $s$ would have earned the previous week, estimated via delta + theta decomposition:

$$R_s^{(t)} = \frac{\sum_{\text{positions}} \left(\delta_i \cdot r_{\text{SPY}} \cdot S_t - \theta_i^{\text{weekly}}\right) \cdot q_i \cdot 100}{\sum_{\text{positions}} |p_i| \cdot q_i \cdot 100}$$

where $\delta_i$ is the Black-Scholes delta, $r_{\text{SPY}} = (S_t - S_{t-5})/S_{t-5}$ is the observed SPY return, $\theta_i^{\text{weekly}} = 0.10 \cdot p_i$ approximates theta decay, and the sign flips for written (short) positions.

**Fitting — nine independent GBRs:**

For each strategy $s \in \{1, \ldots, 9\}$ the GBR solves:

$$F_{100}^{(s)}(\mathbf{x}) \approx R_s, \qquad \text{loss} = \sum_{i=1}^{n} w_i \left(R_s^{(i)} - F_{100}^{(s)}(\mathbf{x}_i)\right)^2$$

using the same exponential decay weights $w_i$ as above.

**Adaptive hyperparameters** (set in `StrategySelector.train`):

| Condition | `max_depth` | `learning_rate` | `min_samples_leaf` |
|---|---|---|---|
| $n < 20$ | 2 | 0.05 | $\max(3,\ n / 8)$ |
| $n \geq 20$ | 3 | 0.08 | $\max(3,\ n / 8)$ |

**Rolling window:** only the most recent $\min(n, 52)$ rows are used in any one training call, capping the effective memory at roughly one year of weekly data.

**Seeded prior:** `seed_strategy_history()` prepends 8 synthetic rows encoding regime–strategy economic priors (e.g., high-vol regime → Bates earns 0.5%, BS earns −0.1%). With `MIN_SAMPLES = 8`, the GBR can train immediately on week 2 (8 seeded + 1 real row = 9 ≥ 8). Exponential decay weights quickly dilute the synthetic rows as real data accumulates: after 8 real weeks the newest real observation outweighs the oldest seeded one by $e^2 \approx 7.4\times$.

**At inference:**

$$s^* = \underset{s \in \{1,\ldots,9\}}{\arg\max}\ F_{100}^{(s)}\!\left(\text{StandardScaler.transform}(\mathbf{x}_t)\right)$$

The algorithm selects the strategy with the highest predicted return and sets it as the active strategy for the coming week.

**Feature importances** are averaged across all nine output models after each training call:

$$\text{FI}_j = \frac{1}{9}\sum_{s=1}^{9} \text{FI}_j^{(s)}, \qquad \text{FI}_j^{(s)} = \sum_{m=1}^{100} \sum_{\substack{\text{split on} \\ \text{feature } j \\ \text{in tree } m}} \Delta_{\text{MSE}}$$

These are logged and written to the end-of-run output charts.

---

### 6. Summary Comparison

| Property | Pricing GBR | Strategy Selector GBR |
|---|---|---|
| **Inputs** | 5 model prices + vol, moneyness, TTM (8 features) | volatility, momentum, MA-deviation, Hurst (4 features) |
| **Output** | 1 scalar — predicted realised option price | 9 scalars — predicted weekly return per strategy |
| **Architecture** | Single `GradientBoostingRegressor` | `MultiOutputRegressor` wrapping 9 independent GBRs |
| **Target** | `realized_price` from MTM or position close | `_compute_hypo_returns()` delta+theta estimates |
| **Min training rows** | 10 (after `.dropna()`) | 8 (including seeded priors) |
| **Rolling window** | Last 1 000 prediction records | Last 52 weekly rows |
| **Decay ratio (newest/oldest)** | $e^2 \approx 7.4\times$ | $e^2 \approx 7.4\times$ |
| **First trains on** | Week 2 (≥10 MTM-realized records) | Week 2 (8 seeded + 1 real row) |
| **Used by** | `MixedStrategy` — generates 'Mixed' signal price | `_rebalance()` — selects active strategy |

---

## The Nine Strategies

### Conventional (no pricing model)

| Class | File | Logic |
|---|---|---|
| `BuyAndHoldStrategy` | `convmethods.py` | Buys the nearest-ATM call; holds until expiry or close |
| `MomentumStrategy` | `convmethods.py` | Buys ATM calls on upward momentum (>2%), ATM puts on downward |
| `MeanReversionStrategy` | `convmethods.py` | Buys nearest OTM call when price is below 20-day MA; OTM put when above |

### Model-Based (mispricing signals)

Each model-based strategy computes a signal price and compares it to the synthetic chain's implied price. If the model price exceeds the chain price by more than `MISPRICING_THRESHOLD = 3%`, it buys; if below by more than that threshold, it writes (shorts) the option.

| Class | File | Model |
|---|---|---|
| `MMARStrategy` | `modelstrats.py` | Multifractal Model of Asset Returns |
| `BlackScholesStrategy` | `modelstrats.py` | Standard Black-Scholes Monte Carlo |
| `HestonStrategy` | `modelstrats.py` | Stochastic volatility (CIR variance process) |
| `MertonStrategy` | `modelstrats.py` | Jump-diffusion (Poisson jumps) |
| `BatesStrategy` | `modelstrats.py` | Stochastic vol + jumps (Heston + Merton) |
| `MixedStrategy` | `modelstrats.py` | Weighted ensemble of all five models (pricing GBR once trained) |

---

## The Five Pricing Models

All five models run concurrently via `ThreadPoolExecutor` (timeout 30 s). Results are combined for the 'Mixed' price. If any model throws or times out, `_get_default_prices()` is returned (intrinsic value × 1.05) so the pipeline never stalls.

```
calculate_all_model_prices_concurrent()
  ├─ BS      : Monte Carlo with lognormal GBM         (closed-form analytic)
  ├─ MMAR    : Multifractal time-subordinated fBM
  │             _sample_cascade()  →  2^5=32 lognormal cascade weights
  │             generate_fbm_path()  →  fractional Brownian motion
  │             Subordinates fBM to multifractal trading time
  ├─ Heston  : Monte Carlo, CIR variance + correlated price
  ├─ Merton  : Monte Carlo, GBM + Poisson jump component
  └─ Bates   : Monte Carlo, Heston volatility + Merton jumps

All → option_pricer()  →  E[max(S_T − K, 0)] · e^{−rT}
```

---

## Key Functions Reference

### `main.py — OptionsArbitrageAlgorithm`

| Function | Purpose |
|---|---|
| `run_backtest()` | Downloads data, runs the daily simulation loop, calls `_print_summary()` and `_generate_output()` |
| `initialize()` | Creates portfolio, pricing calculator, strategy instances, seeds strategy history |
| `_generate_option_chain(spot, date)` | Builds a synthetic BS chain with VRP factor (×1.20) and negative skew on put strikes |
| `_prepare_option_data(chain, hurst)` | Prices each contract with all five models; stores predictions in the tracker |
| `_rebalance()` | Weekly orchestrator: close positions → calibrate → price → signal → execute |
| `_selective_close_positions()` | Closes positions with DTE ≤ 3, gain ≥ 50%, or loss ≥ 80%; records realization |
| `_attempt_ensemble_calibration()` | Triggers MTM, then retrains the pricing GBR and strategy GBR if enough data |
| `_mark_to_market_realizations()` | Fills `realized_price` for all open predictions from the current securities dict |
| `_execute_signal(signal, options)` | Places buy / write / sell orders; prevents pyramiding; caps at 6 concurrent positions |
| `_compute_hypo_returns(positions)` | Estimates last week's per-strategy P&L using delta + theta approximation |
| `_compute_strategy_features()` | Returns `{volatility, momentum_5d, ma_deviation_20, hurst}` for the strategy GBR |
| `_close_position_with_log(symbol)` | Closes long (sell) or short (cover); logs P&L to `_trade_log` |
| `_days_to_expiry(symbol_str)` | Parses DTE from symbol format `"SPY YYMMDD C/PXXXXX"` |
| `_has_sufficient_margin(price, qty)` | Checks `margin_remaining ≥ price × qty × 100 × 1.3`; halts trading if not |
| `_compute_option_delta(option_row)` | Black-Scholes delta used for SPY hedge sizing |

### `ensemble.py`

| Function / Class | Purpose |
|---|---|
| `seed_strategy_history()` | Returns 8 synthetic prior rows encoding regime → strategy economic intuitions; enables strategy GBR to train from week 2 |
| `StrategySelector` | Wraps `MultiOutputRegressor(GBR)` over 9 outputs; trains on rolling 52-row window with exponential decay weights |
| `StrategySelector.select(features)` | Returns the best strategy name; falls back to `_regime_based_fallback()` if untrained |
| `StrategySelector._regime_based_fallback(features)` | Rule-based heuristic: vol>0.28→Bates, high Hurst + momentum→Momentum, low Hurst→MeanReversion, low vol→BS, else Mixed |
| `ModelPerformanceTracker` | Stores prediction records and fills `realized_price` when positions close or MTM runs |
| `ModelPerformanceTracker.record_prediction(...)` | Appends one record per contract per week; `realized_price = None` until filled |
| `ModelPerformanceTracker.record_realization(symbol, price)` | Matches by `contract_symbol` only (not expiry); fills the most-recent unrealized record |
| `ModelPerformanceTracker.mark_open_predictions(securities_dict)` | MTM pass: fills `realized_price` for all still-open predictions from the securities mid-price dict |
| `ModelPerformanceTracker.get_calibration_dataframe()` | Returns DataFrame of all realized records; `None` if none exist yet |
| `OptionPricingCalculator` | Owns the pricing GBR (`_meta_learner`), the performance tracker, and Heston/Merton parameters |
| `OptionPricingCalculator.calibrate_ensemble_weights(df)` | Trains `GradientBoostingRegressor` on realized prices; requires ≥10 valid rows after `.dropna()` |
| `OptionPricingCalculator.calculate_model_prices(...)` | Dispatches to `calculate_all_model_prices_concurrent`; uses GBR for 'Mixed' price once trained |
| `block_bootstrap_sharpe(returns)` | Block-bootstrap confidence interval on the Sharpe ratio; used for regime-level statistical testing |

### `models.py`

| Function | Purpose |
|---|---|
| `calculate_all_model_prices_concurrent(...)` | Runs all five models in parallel threads; collects results via `as_completed(timeout=30)` |
| `calculate_mmar_price(S, K, r, T, hurst, paths)` | MMAR: lognormal cascade → multifractal trading time → fBM subordination → Monte Carlo payoffs |
| `_sample_cascade(layers, v, ...)` | Recursive lognormal cascade; returns flat array of 2^layers leaf weights |
| `calculate_heston_price(...)` | Monte Carlo: CIR variance process + correlated Brownian price process |
| `calculate_merton_price(...)` | Monte Carlo: GBM drift + Poisson-counted log-normal jumps |
| `calculate_bates_price(...)` | Monte Carlo: Heston stochastic vol combined with Merton jump component |
| `calculate_bs_price(...)` | Monte Carlo Black-Scholes (analytic-equivalent with `num_paths` samples) |
| `calculate_hurst_for_segments(data, n)` | Segments the price series; computes Hurst exponent on each segment via `nolds.hurst_rs` |
| `option_pricer(paths, K, r, T, type)` | Discounted expected payoff across all Monte Carlo terminal prices |
| `bs_delta(S, K, r, sigma, T, type)` | Black-Scholes delta; used as universal hedge ratio for all models |
| `generate_fbm_path(n, hurst, dt, s0)` | Fractional Brownian motion path via scaled Gaussian increments |

---

## Regime-Based Strategy Fallback

Before the strategy GBR has trained (weeks 1–2), `_regime_based_fallback()` maps market features to a strategy deterministically:

```
volatility > 0.28                           →  Bates   (jumps dominate)
hurst > 0.54  AND  |momentum_5d| > 0.012   →  Momentum
hurst < 0.46  OR   |ma_deviation_20| > 0.02 →  MeanReversion
volatility < 0.15                           →  BS      (smooth, low-vol)
else                                         →  Mixed
```

---

## Portfolio and Risk Controls

| Mechanism | Setting | Location |
|---|---|---|
| Max concurrent positions | 6 | `MAX_CONCURRENT_POSITIONS` in `main.py` |
| Take-profit close | +50% | `_selective_close_positions()` |
| Stop-loss close | −80% | `_selective_close_positions()` |
| DTE close | ≤ 3 days | `_selective_close_positions()` |
| Margin requirement | 1.3× option cost | `_has_sufficient_margin()` |
| Margin buffer on sizing | 30% | `_calculate_max_position_quantity()` |
| Delta hedge | SPY shares via `market_order` | `_rebalance()` after signals |
| Capital | $100 000 | `Portfolio(100000)` in `initialize()` |

---

## Output

The backtester writes all output to `/app/output` (Docker path). At the end of the run `_generate_output()` produces:

- `backtest_summary.json` — CAGR, Sharpe, Sortino, Calmar, max drawdown, win rate, alpha, beta
- Equity curve chart
- Drawdown chart
- Weekly returns distribution
- Strategy allocation over time
- Model accuracy metrics (MAE, RMSE, MAPE, directional accuracy per model)
- Diebold-Mariano test results (statistical comparison of model forecast accuracy)
- GBR feature importance charts
- Rolling Sharpe chart

An interim equity curve is also written every 4 rebalances (~monthly) during the run.

---

## Running the Backtester

```bash
# Locally (requires nolds, yfinance, scikit-learn, scipy, tqdm, seaborn, matplotlib)
python main.py

# Docker (recommended — sets OUTPUT_DIR and MPLBACKEND automatically)
docker build -t options-backtester .
docker run -v $(pwd)/output:/app/output options-backtester

# Unit tests (GBR pipeline verification)
pytest unit_tests_gbr.py -v
```

> **Important**: if you edit any `.py` file, rebuild the Docker image before running again. The container does not pick up host-side file changes automatically.
