
import math
import time
from typing import Callable, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import roots_laguerre, ndtr
from scipy.stats import norm

# Gauss-Laguerre quadrature nodes/weights precomputed once — used only by the
# scalar reference pricer heston_call_analytical (kept for regression tests).
# ∫₀^∞ f(u) du ≈ ∑_k (w_k · e^(x_k)) · f(x_k)
_GL_NODES, _GL_WEIGHTS = roots_laguerre(64)
_GL_WEIGHTS_EXP = _GL_WEIGHTS * np.exp(_GL_NODES)

# Gauss-Legendre nodes on (0, _CF_X_MAX] for the vectorised characteristic-
# function pricers. The integration variable is rescaled per contract by the
# total standard deviation sqrt(v̄·T) (see _gil_pelaez_probs), so a fixed
# dimensionless range covers every maturity in the 7-90 day universe.
_CF_N_NODES = 128
_CF_X_MAX = 12.0
_leg_x, _leg_w = np.polynomial.legendre.leggauss(_CF_N_NODES)
_CF_X = 0.5 * _CF_X_MAX * (_leg_x + 1.0)
_CF_W = 0.5 * _CF_X_MAX * _leg_w

# Strike/spot band edges and maturity buckets (calendar days) used by the
# stratified samplers. Calibration uses a wider outer band than evaluation.
EVAL_MONEYNESS_EDGES = (0.85, 0.95, 0.99, 1.01, 1.05, 1.15)
CALIB_MONEYNESS_EDGES = (0.80, 0.95, 0.99, 1.01, 1.05, 1.20)
MATURITY_BUCKETS_DAYS = ((7, 30), (31, 90))

# Seconds of wall time between progress lines emitted during an optimisation.
PROGRESS_LOG_EVERY_S = 5.0


class _ProgressLogger:
    """
    Wraps an objective function to report optimisation progress through `log`
    (e.g. the algorithm's debug): a start line, a progress line every
    PROGRESS_LOG_EVERY_S seconds (evaluations, elapsed time, best objective and
    its parameters), and a closing summary. A no-op when `log` is None.
    """

    def __init__(self, log: Optional[Callable[[str], None]], name: str,
                 param_names: tuple, n_contracts: int):
        self.log, self.name, self.param_names = log, name, param_names
        self.n_evals, self.best_f, self.best_x = 0, float('inf'), None
        self.t0 = self.t_last = time.perf_counter()
        if log:
            log(f"[{name}] start: {n_contracts} contracts, params {', '.join(param_names)}")

    def _fmt(self, x) -> str:
        if x is None:
            return '—'
        return ' '.join(f"{n}={float(v):.4g}" for n, v in zip(self.param_names, np.atleast_1d(x)))

    def wrap(self, fn: Callable) -> Callable:
        def wrapped(x, *args):
            f = fn(x, *args)
            self.n_evals += 1
            if f < self.best_f:
                self.best_f, self.best_x = f, np.array(x, dtype=float, copy=True)
            now = time.perf_counter()
            if self.log and now - self.t_last >= PROGRESS_LOG_EVERY_S:
                self.t_last = now
                self.log(f"[{self.name}] eval {self.n_evals}, {now - self.t0:.1f}s, "
                         f"best obj {self.best_f:.4g} at {self._fmt(self.best_x)}")
            return f
        return wrapped

    def done(self, res=None, note: str = '') -> None:
        if not self.log:
            return
        status = ''
        if res is not None:
            status = f"{'converged' if getattr(res, 'success', True) else 'NOT converged'}"
            if getattr(res, 'nit', None) is not None:
                status += f" in {res.nit} iterations"
            status += ', '
        self.log(f"[{self.name}] done: {status}{self.n_evals} evals, "
                 f"{time.perf_counter() - self.t0:.1f}s, obj {self.best_f:.4g} at {self._fmt(self.best_x)}"
                 f"{' ' + note if note else ''}")


# ---------------------------------------------------------------------------
# Black-Scholes (with continuous dividend yield q)
# ---------------------------------------------------------------------------

def _bs_call(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0) -> float:
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sq
    d2 = d1 - sq
    return float(S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))


def _bs_price(S: float, K: float, r: float, sigma: float,
              T: float, option_type: str, q: float = 0.0) -> float:
    call = _bs_call(S, K, r, sigma, T, q)
    if option_type == 'call':
        return call
    return call - S * math.exp(-q * T) + K * math.exp(-r * T)


def bs_price_vec(S: float, K: np.ndarray, T: np.ndarray, r: float, q: float,
                 total_std: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """
    Vectorised Black-Scholes price. `total_std` is σ·√T (the standard deviation
    of ln S_T), so callers with maturity-dependent variance (the MMAR's
    conditional variance) can pass it directly. Arrays broadcast together.
    """
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sd = np.maximum(np.asarray(total_std, dtype=float), 1e-12)
    fwd = S * np.exp((r - q) * T)
    d1 = (np.log(fwd / K) + 0.5 * sd ** 2) / sd
    d2 = d1 - sd
    disc = np.exp(-r * T)
    call = disc * (fwd * ndtr(d1) - K * ndtr(d2))
    put = disc * (K * ndtr(-d2) - fwd * ndtr(-d1))
    return np.where(is_call, call, put)


def _chain_arrays(rows: pd.DataFrame) -> tuple:
    return (rows['strike'].values.astype(float),
            rows['ttm'].values.astype(float),
            rows['mid_price'].values.astype(float),
            (rows['option_type'].values == 'call'))


def calibrate_bs_iv(chain_df: pd.DataFrame, spot: float, r: float, q: float = 0.0,
                    log: Optional[Callable[[str], None]] = None) -> float:

    rows = chain_df.dropna(subset=['strike', 'ttm', 'mid_price', 'option_type'])
    rows = rows[(rows['ttm'] > 0) & (rows['mid_price'] > 0)]
    if rows.empty:
        return 0.20

    K_arr, T_arr, m_arr, is_call = _chain_arrays(rows)
    sqrt_T = np.sqrt(T_arr)
    progress = _ProgressLogger(log, 'BS NLS', ('sigma',), len(rows))

    def objective(sigma: float) -> float:
        model = bs_price_vec(spot, K_arr, T_arr, r, q, sigma * sqrt_T, is_call)
        return float(np.sum((model - m_arr) ** 2))

    result = minimize_scalar(progress.wrap(objective), bounds=(0.01, 2.0), method='bounded',
                             options={'xatol': 1e-4})
    progress.done(result)
    return float(result.x) if result.success else 0.20


# ---------------------------------------------------------------------------
# Heston semi-closed-form pricing — scalar reference (Gil-Pelaez, original
# Heston 1993 two-CF form, no dividend). Retained only as a regression
# reference for heston_price_vec; not used by the pipeline.
# ---------------------------------------------------------------------------

def _heston_cf(u: complex, S: float, K: float, r: float, T: float,
               v0: float, kappa: float, theta: float,
               sigma: float, rho: float, j: int) -> complex:

    i = 1j
    b_j = (kappa - rho * sigma) if j == 1 else kappa
    u_j = 0.5 if j == 1 else -0.5

    d = np.sqrt((rho * sigma * i * u - b_j) ** 2
                - sigma ** 2 * (2.0 * u_j * i * u - u ** 2))
    g = (b_j - rho * sigma * i * u + d) / (b_j - rho * sigma * i * u - d)
    exp_dT = np.exp(d * T)
    log_term = np.log((1.0 - g * exp_dT) / (1.0 - g))

    C = (r * i * u * T
         + (kappa * theta / sigma ** 2)
         * ((b_j - rho * sigma * i * u + d) * T - 2.0 * log_term))
    D = ((b_j - rho * sigma * i * u + d) / sigma ** 2
         * (1.0 - exp_dT) / (1.0 - g * exp_dT))
    return np.exp(C + D * v0 + i * u * np.log(S / K))


def heston_call_analytical(S: float, K: float, r: float, T: float,
                            v0: float, kappa: float, theta: float,
                            sigma: float, rho: float) -> tuple:
    # _heston_cf already encodes exp(iu·log(S/K)), so the Gil-Pelaez integrand
    # is cf/(iu) — no additional exp(-iu·log(K)) factor.
    def _P(j: int) -> float:
        cf_vals = np.array([
            _heston_cf(complex(u), S, K, r, T, v0, kappa, theta, sigma, rho, j)
            for u in _GL_NODES
        ])
        integrand_vals = np.real(cf_vals / (1j * _GL_NODES))
        return 0.5 + float(np.dot(_GL_WEIGHTS_EXP, integrand_vals)) / math.pi

    P1, P2 = _P(1), _P(2)
    call = max(S * P1 - K * math.exp(-r * T) * P2, 0.0)
    put = max(call - S + K * math.exp(-r * T), 0.0)
    return float(call), float(put)


# ---------------------------------------------------------------------------
# Vectorised Heston / Bates pricing from the risk-neutral characteristic
# function of x = ln S_T ("little trap" form, Albrecher et al. 2007), with an
# optional Merton log-normal compound-Poisson jump term (Bates 1996).
#   P2 = Q(S_T > K)            = ½ + (1/π)∫ Re[e^{-iu ln K} φ(u)     / (iu)]      du
#   P1 = Q^S(S_T > K)          = ½ + (1/π)∫ Re[e^{-iu ln K} φ(u − i) / (iu φ(−i))] du
#   call = S e^{-qT} P1 − K e^{-rT} P2,   φ(−i) = S e^{(r−q)T}
# ---------------------------------------------------------------------------

def _log_cf(u: np.ndarray, S: float, T: np.ndarray, r: float, q: float,
            v0: float, kappa: float, theta: float, sigma: float, rho: float,
            lam: float, mu_j: float, sigma_j: float) -> np.ndarray:
    iu = 1j * u
    xi = kappa - rho * sigma * iu
    d = np.sqrt(xi ** 2 + sigma ** 2 * (u ** 2 + iu))
    g2 = (xi - d) / (xi + d)
    e = np.exp(-d * T)
    C = (kappa * theta / sigma ** 2) * ((xi - d) * T - 2.0 * np.log((1.0 - g2 * e) / (1.0 - g2)))
    D = (xi - d) / sigma ** 2 * (1.0 - e) / (1.0 - g2 * e)
    log_phi = iu * (math.log(S) + (r - q) * T) + C + D * v0
    if lam > 0.0:
        k = math.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
        log_phi = log_phi + lam * T * (np.exp(iu * mu_j - 0.5 * sigma_j ** 2 * u ** 2) - 1.0 - iu * k)
    return np.exp(log_phi)


def _gil_pelaez_probs(S: float, K: np.ndarray, T: np.ndarray, r: float, q: float,
                      params: tuple) -> tuple:
    v0, kappa, theta, sigma, rho, lam, mu_j, sigma_j = params
    K = np.asarray(K, dtype=float)[:, None]
    T = np.asarray(T, dtype=float)[:, None]
    # Per-contract rescaling u = x / sqrt(v̄ T): the integrand decays on the
    # scale of the total log-price standard deviation.
    v_bar = max(v0, theta, 1e-4) + lam * (mu_j ** 2 + sigma_j ** 2)
    scale = 1.0 / np.sqrt(v_bar * T)
    u = _CF_X[None, :] * scale
    w = _CF_W[None, :] * scale
    lnK = np.log(K)
    cf_args = (S, T, r, q, v0, kappa, theta, sigma, rho, lam, mu_j, sigma_j)
    phi = _log_cf(u, *cf_args)
    phi_shift = _log_cf(u - 1j, *cf_args)
    fwd = S * np.exp((r - q) * T)
    kernel = np.exp(-1j * u * lnK) / (1j * u)
    P2 = 0.5 + np.sum(w * np.real(kernel * phi), axis=1) / math.pi
    P1 = 0.5 + np.sum(w * np.real(kernel * phi_shift / fwd), axis=1) / math.pi
    return np.clip(P1, 0.0, 1.0), np.clip(P2, 0.0, 1.0)


def bates_price_vec(S: float, K: np.ndarray, T: np.ndarray, r: float, q: float,
                    v0: float, kappa: float, theta: float, sigma: float, rho: float,
                    lam: float, mu_j: float, sigma_j: float,
                    is_call: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    P1, P2 = _gil_pelaez_probs(S, K, T, r, q,
                               (v0, kappa, theta, sigma, rho, lam, mu_j, sigma_j))
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    call = np.maximum(disc_S * P1 - disc_K * P2, np.maximum(disc_S - disc_K, 0.0))
    put = call - disc_S + disc_K
    return np.where(is_call, call, np.maximum(put, 0.0))


def heston_price_vec(S: float, K: np.ndarray, T: np.ndarray, r: float, q: float,
                     v0: float, kappa: float, theta: float, sigma: float, rho: float,
                     is_call: np.ndarray) -> np.ndarray:
    return bates_price_vec(S, K, T, r, q, v0, kappa, theta, sigma, rho,
                           0.0, 0.0, 0.0, is_call)


def bates_call_analytical(S: float, K: float, r: float, T: float,
                           v0: float, kappa: float, theta: float,
                           sigma: float, rho: float,
                           lam: float, mu_j: float, sigma_j: float) -> tuple:
    """Scalar convenience wrapper around bates_price_vec (q = 0)."""
    call = float(bates_price_vec(S, np.array([K]), np.array([T]), r, 0.0, v0, kappa, theta, sigma, rho,
                                 lam, mu_j, sigma_j, np.array([True]))[0])
    put = max(call - S + K * math.exp(-r * T), 0.0)
    return call, float(put)


def _heston_x0(init_params: dict | None) -> list:
    return ([init_params['v0'], init_params['kappa'],
             init_params['theta'], init_params['sigma'], init_params['rho']]
            if init_params else [0.04, 1.5, 0.04, 0.30, -0.50])


_HESTON_BOUNDS = [(1e-4, 1.0), (0.1, 10.0), (1e-4, 1.0), (0.01, 2.0), (-0.99, -0.01)]


def _fit_heston_family(rows: pd.DataFrame, spot: float, r: float, q: float,
                       init_params: dict | None, jumps: tuple,
                       log: Optional[Callable[[str], None]] = None,
                       name: str = 'Heston NLS') -> dict | None:
    K_arr, T_arr, m_arr, is_call = _chain_arrays(rows)
    lam, mu_j, sigma_j = jumps
    progress = _ProgressLogger(log, name, ('v0', 'kappa', 'theta', 'xi', 'rho'), len(rows))

    def objective(params: np.ndarray) -> float:
        v0, kappa, theta, sigma, rho = params
        model = bates_price_vec(spot, K_arr, T_arr, r, q, v0, kappa, theta, sigma, rho,
                                lam, mu_j, sigma_j, is_call)
        if not np.all(np.isfinite(model)):
            return 1e8
        return float(np.sum((model - m_arr) ** 2))

    res = minimize(progress.wrap(objective), _heston_x0(init_params), method='L-BFGS-B',
                   bounds=_HESTON_BOUNDS, options={'maxiter': 200, 'ftol': 1e-6})
    progress.done(res)
    if not res.success:
        return None
    v0, kappa, theta, sigma, rho = res.x
    return {'v0': float(v0), 'kappa': float(kappa), 'theta': float(theta),
            'sigma': float(sigma), 'rho': float(rho)}


def _calib_rows(chain_df: pd.DataFrame) -> pd.DataFrame:
    rows = chain_df.dropna(subset=['strike', 'ttm', 'mid_price', 'option_type'])
    return rows[(rows['ttm'] > 1 / 365) & (rows['mid_price'] > 0.05)]


def calibrate_heston_nls(chain_df: pd.DataFrame, spot: float, r: float,
                          init_params: dict | None = None, q: float = 0.0,
                          log: Optional[Callable[[str], None]] = None) -> dict | None:
    rows = _calib_rows(chain_df)
    if len(rows) < 5:
        return None
    return _fit_heston_family(rows, spot, r, q, init_params, (0.0, 0.0, 0.0), log, 'Heston NLS')


# ---------------------------------------------------------------------------
# Merton jump-diffusion: closed form, MLE on returns, NLS on quotes
# ---------------------------------------------------------------------------

MERTON_N_TERMS = 40


def merton_price_vec(S: float, K: np.ndarray, T: np.ndarray, r: float, q: float,
                     sigma: float, lam: float, mu_j: float, sigma_j: float,
                     is_call: np.ndarray, n_terms: int = MERTON_N_TERMS) -> np.ndarray:
    """Merton (1976) price as a Poisson-weighted sum of Black-Scholes prices."""
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    k = math.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    price = np.zeros(np.broadcast(K, T).shape)
    for n in range(n_terms):
        # Conditional on n ~ Poisson(λT) jumps, ln S_T is normal with forward
        # S e^{(r-q-λk)T}(1+k)^n and variance σ²T + nσ_J²; discount at r.
        var_n = sigma ** 2 * T + n * sigma_j ** 2
        drift_shift = n * math.log1p(k) - lam * k * T
        log_w = (-lam * T + n * np.log(np.maximum(lam * T, 1e-300))
                 - math.lgamma(n + 1))
        # Price with spot adjusted so the conditional forward carries the jump shift.
        S_n = S * np.exp(drift_shift)
        price = price + np.exp(log_w) * bs_price_vec(1.0, K / S_n, T, r, q,
                                                     np.sqrt(var_n), is_call) * S_n
    return price


def calibrate_merton_mle(log_returns: np.ndarray,
                          dt: float = 1 / 252,
                          n_terms: int = 20,
                          log: Optional[Callable[[str], None]] = None) -> dict | None:

    if len(log_returns) < 30:
        return None

    ns = np.arange(n_terms, dtype=float)
    log_factorials = np.array([math.lgamma(n + 1) for n in ns])
    r_t = np.asarray(log_returns, dtype=float)[:, None]

    def neg_log_likelihood(params: np.ndarray) -> float:
        sigma, lam, mu_j, sigma_j = params
        if sigma <= 0 or lam < 0 or sigma_j <= 0:
            return 1e10

        lam_dt = max(lam * dt, 1e-300)
        # log P(N=n) = n*log(lambda*dt) - lambda*dt - log(n!)
        log_pn = ns * math.log(lam_dt) - lam * dt - log_factorials

        drift = (-0.5 * sigma ** 2 * dt
                 - lam * (math.exp(mu_j + 0.5 * sigma_j ** 2) - 1) * dt)
        mu_ns = drift + ns * mu_j
        var_ns = np.maximum(sigma ** 2 * dt + ns * sigma_j ** 2, 1e-10)

        log_phi = -0.5 * np.log(2 * math.pi * var_ns) - 0.5 * (r_t - mu_ns) ** 2 / var_ns
        return -float(np.sum(np.logaddexp.reduce(log_pn + log_phi, axis=1)))

    x0 = np.array([0.15, 0.10, -0.05, 0.08])
    bounds = [(0.01, 2.0), (0.0, 20.0), (-2.0, 0.5), (0.01, 1.5)]
    progress = _ProgressLogger(log, 'Merton MLE', ('sigma', 'lambda', 'mu_j', 'sigma_j'), len(log_returns))
    res = minimize(progress.wrap(neg_log_likelihood), x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 300, 'ftol': 1e-7})
    progress.done(res, note=f"(on {len(log_returns)} daily returns; obj = −log-likelihood)")
    if not res.success:
        return None
    sigma, lam, mu_j, sigma_j = res.x
    return {
        'lambda_jump':     float(lam),
        'mu_jump':         float(mu_j),
        'sigma_jump':      float(sigma_j),
        'diffusion_sigma': float(sigma),
    }


def calibrate_merton_nls(chain_df: pd.DataFrame, spot: float, r: float, q: float = 0.0,
                          init_params: dict | None = None,
                          log: Optional[Callable[[str], None]] = None) -> dict | None:
    """
    Robustness specification: Merton's (σ, λ, μ_J, σ_J) fitted to the t-1
    option cross-section by the same squared-price-deviation objective used
    for the other quote-calibrated models.
    """
    rows = _calib_rows(chain_df)
    if len(rows) < 5:
        return None
    K_arr, T_arr, m_arr, is_call = _chain_arrays(rows)

    def objective(params: np.ndarray) -> float:
        sigma, lam, mu_j, sigma_j = params
        model = merton_price_vec(spot, K_arr, T_arr, r, q, sigma, lam, mu_j, sigma_j, is_call)
        if not np.all(np.isfinite(model)):
            return 1e8
        return float(np.sum((model - m_arr) ** 2))

    x0 = ([init_params['diffusion_sigma'], init_params['lambda_jump'],
           init_params['mu_jump'], init_params['sigma_jump']]
          if init_params else [0.12, 1.0, -0.05, 0.08])
    bounds = [(0.01, 2.0), (0.0, 20.0), (-0.5, 0.2), (0.005, 0.5)]
    x0 = [float(np.clip(x, lo, hi)) for x, (lo, hi) in zip(x0, bounds)]
    progress = _ProgressLogger(log, 'Merton-Q NLS', ('sigma', 'lambda', 'mu_j', 'sigma_j'), len(rows))
    res = minimize(progress.wrap(objective), x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 200, 'ftol': 1e-7})
    progress.done(res)
    if not res.success:
        return None
    sigma, lam, mu_j, sigma_j = res.x
    return {'diffusion_sigma': float(sigma), 'lambda_jump': float(lam),
            'mu_jump': float(mu_j), 'sigma_jump': float(sigma_j)}


# ---------------------------------------------------------------------------
# Bates calibration (Heston part by NLS, jump part fixed from Merton MLE)
# ---------------------------------------------------------------------------

def calibrate_bates_nls(chain_df: pd.DataFrame, spot: float, r: float,
                         merton_params: dict,
                         init_heston: dict | None = None, q: float = 0.0,
                         log: Optional[Callable[[str], None]] = None) -> dict | None:

    jumps = (merton_params.get('lambda_jump', 0.1),
             merton_params.get('mu_jump', -0.05),
             merton_params.get('sigma_jump', 0.1))
    rows = _calib_rows(chain_df)
    if len(rows) < 5:
        return None
    return _fit_heston_family(rows, spot, r, q, init_heston, jumps, log, 'Bates NLS')


# ---------------------------------------------------------------------------
# Cross-section filtering and stratified sampling
# ---------------------------------------------------------------------------

def filter_chain_for_calibration(chain_df: pd.DataFrame, spot: float,
                                  moneyness_lo: float = 0.80,
                                  moneyness_hi: float = 1.20,
                                  min_ttm_days: int = 7,
                                  max_ttm_days: int = 90) -> pd.DataFrame:

    df = chain_df.copy()
    bid_col = 'bid_price' if 'bid_price' in df.columns else 'mid_price'
    df = df[(df['mid_price'] > 0) & (df[bid_col] > 0)]
    df = df[(df['ttm'] >= min_ttm_days / 365.25)
            & (df['ttm'] <= max_ttm_days / 365.25)]
    moneyness = df['strike'] / spot
    df = df[(moneyness >= moneyness_lo) & (moneyness <= moneyness_hi)]
    return df.reset_index(drop=True)


def stratified_sample(chain_df: pd.DataFrame, spot: float, n_per_cell: int, seed: int,
                      moneyness_edges: tuple = EVAL_MONEYNESS_EDGES,
                      ttm_col: str = 'ttm') -> pd.DataFrame:
    """
    Deterministic stratified sample: up to `n_per_cell` contracts drawn
    uniformly at random (seeded) from each (K/S band × maturity bucket) cell.
    `chain_df[ttm_col]` is in years.
    """
    df = chain_df.copy()
    ks = df['strike'].values / spot
    days = np.rint(df[ttm_col].values * 365.25)
    df['_band'] = pd.cut(ks, bins=list(moneyness_edges), labels=False, include_lowest=True)
    df['_mat'] = np.select([(days >= lo) & (days <= hi) for lo, hi in MATURITY_BUCKETS_DAYS],
                           list(range(len(MATURITY_BUCKETS_DAYS))), default=-1)
    df = df[df['_band'].notna() & (df['_mat'] >= 0)]
    rng = np.random.default_rng(seed)
    picks = []
    for _, cell in df.groupby(['_band', '_mat'], sort=True):
        take = min(n_per_cell, len(cell))
        picks.append(cell.iloc[np.sort(rng.choice(len(cell), size=take, replace=False))])
    if not picks:
        return df.iloc[0:0].drop(columns=['_band', '_mat'])
    return pd.concat(picks).drop(columns=['_band', '_mat']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# MMAR cross-sectional NLS calibration
# ---------------------------------------------------------------------------

# cascade_sigma upper bound widened from 2.0 to 4.0 after the static-H fit sat
# on the 2.0 cap on 22 of 45 dates in the 2026-09-24 run.
MMAR_BOUNDS = {'sigma': (0.01, 3.0), 'hurst': (0.05, 0.99), 'cascade_sigma': (0.0, 4.0)}


def calibrate_mmar_nls(chain_df: pd.DataFrame, spot: float, r: float,
                        cascade_z: np.ndarray,
                        q: float = 0.0,
                        init_params: dict | None = None,
                        hurst_starts: tuple = (0.65, 0.5),
                        bs_sigma: float = 0.15,
                        fixed_hurst: float | None = None,
                        max_iter: int = 400,
                        log: Optional[Callable[[str], None]] = None) -> dict | None:
    """
    Weekly cross-sectional NLS calibration of the MMAR (σ, H, s) against the
    t-1 option mid-quotes, using the conditional-Monte-Carlo pricer
    `models.mmar_price_vec` with a fixed matrix of cascade normals
    `cascade_z` (common random numbers). With the draws held fixed the
    objective is a deterministic, smooth function of the parameters, so
    Nelder-Mead's convergence test compares signal rather than noise.

    `fixed_hurst` gives the static-H specification: H is held at the given
    value (the returns-based estimate) and only (σ, s) are fitted.

    Multi-start over the H values in `hurst_starts` (plus the previous week's
    fit when `init_params` is supplied); only converged restarts are accepted
    and None is returned if none converge.
    """
    from models import mmar_price_vec

    rows = _calib_rows(chain_df)
    if len(rows) < 5:
        return None
    K_arr, T_arr, m_arr, is_call = _chain_arrays(rows)
    T_med = float(np.median(T_arr))

    def _unpack(x: np.ndarray) -> tuple:
        if fixed_hurst is not None:
            return x[0], fixed_hurst, x[1]
        return x[0], x[1], x[2]

    def objective(x: np.ndarray) -> float:
        sigma, H, s = _unpack(x)
        if not (MMAR_BOUNDS['sigma'][0] < sigma < MMAR_BOUNDS['sigma'][1]
                and MMAR_BOUNDS['hurst'][0] < H < MMAR_BOUNDS['hurst'][1]
                and MMAR_BOUNDS['cascade_sigma'][0] <= s < MMAR_BOUNDS['cascade_sigma'][1]):
            return 1e8
        model = mmar_price_vec(spot, K_arr, T_arr, r, q, sigma, H, s, is_call, cascade_z)
        return float(np.sum((model - m_arr) ** 2))

    starts = []
    if init_params is not None:
        starts.append((init_params['sigma'], init_params['hurst'], init_params['cascade_sigma']))
    for H0 in ([fixed_hurst] if fixed_hurst is not None else hurst_starts):
        # Match the BS total variance at the median maturity: σ² T^{2H} = σ_BS² T.
        starts.append((bs_sigma * T_med ** (0.5 - H0), H0, 0.3))

    label = 'MMAR static-H NLS' if fixed_hurst is not None else 'MMAR NLS'
    pnames = ('sigma', 's') if fixed_hurst is not None else ('sigma', 'H', 's')
    results = []
    for i, (sigma0, H0, s0) in enumerate(starts, 1):
        x0 = np.array([sigma0, s0] if fixed_hurst is not None else [sigma0, H0, s0])
        progress = _ProgressLogger(log, f"{label} start {i}/{len(starts)}", pnames, len(rows))
        f0 = objective(x0)
        rel_fatol = max(1e-3, 1e-5 * abs(f0))
        res = minimize(progress.wrap(objective), x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'maxfev': max_iter * 2,
                                'xatol': 1e-4, 'fatol': rel_fatol})
        progress.done(res, note=f"(from {progress._fmt(x0)}, cascade draws={cascade_z.shape[0]})")
        results.append(res)

    converged = [res for res in results if res.success and res.fun < 1e7]
    if not converged:
        return None
    best = min(converged, key=lambda res: res.fun)
    sigma, H, s = _unpack(best.x)
    return {
        'sigma':              float(sigma),
        'hurst':              float(H),
        'cascade_sigma':      float(s),
        'objective':          round(float(best.fun), 6),
        'converged':          True,
        'n_iter':             int(best.nit),
        'hurst_at_bound':     bool(fixed_hurst is None and min(abs(H - b) for b in MMAR_BOUNDS['hurst']) < 2e-3),
        'sigma_at_bound':     bool(min(abs(s - b) for b in MMAR_BOUNDS['cascade_sigma']) < 2e-3),
        'n_multistarts':      len(starts),
        'n_converged_starts': len(converged),
    }
