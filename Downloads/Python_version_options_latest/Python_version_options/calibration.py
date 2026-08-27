
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import roots_laguerre
from scipy.stats import norm

# Gauss-Laguerre quadrature nodes/weights precomputed once.
# 64 points give < 1e-6 relative error for typical Heston/Bates parameters.
# ∫₀^∞ f(u) du ≈ ∑_k (w_k · e^(x_k)) · f(x_k)
_GL_NODES, _GL_WEIGHTS = roots_laguerre(64)
_GL_WEIGHTS_EXP = _GL_WEIGHTS * np.exp(_GL_NODES)


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _bs_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sq
    d2 = d1 - sq
    return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))


def _bs_price(S: float, K: float, r: float, sigma: float,
              T: float, option_type: str) -> float:
    call = _bs_call(S, K, r, sigma, T)
    if option_type == 'call':
        return call
    return call - S + K * math.exp(-r * T)


def calibrate_bs_iv(chain_df: pd.DataFrame, spot: float, r: float) -> float:

    rows = chain_df.dropna(subset=['strike', 'ttm', 'mid_price', 'option_type'])
    rows = rows[(rows['ttm'] > 0) & (rows['mid_price'] > 0)]
    if rows.empty:
        return 0.20

    K_arr = rows['strike'].values.astype(float)
    T_arr = rows['ttm'].values.astype(float)
    m_arr = rows['mid_price'].values.astype(float)
    t_arr = rows['option_type'].values

    def objective(sigma: float) -> float:
        err = 0.0
        for K, T, mid, otype in zip(K_arr, T_arr, m_arr, t_arr):
            try:
                err += (_bs_price(spot, K, r, sigma, T, otype) - mid) ** 2
            except Exception:
                err += 1e4
        return err

    result = minimize_scalar(objective, bounds=(0.01, 2.0), method='bounded',
                             options={'xatol': 1e-4})
    return float(result.x) if result.success else 0.20


# ---------------------------------------------------------------------------
# Heston semi-closed-form pricing (characteristic function via Gil-Pelaez)
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


def calibrate_heston_nls(chain_df: pd.DataFrame, spot: float, r: float,
                          init_params: dict | None = None) -> dict | None:

    rows = chain_df.dropna(subset=['strike', 'ttm', 'mid_price', 'option_type'])
    rows = rows[(rows['ttm'] > 1 / 365) & (rows['mid_price'] > 0.05)]
    if len(rows) < 5:
        return None

    K_arr = rows['strike'].values.astype(float)
    T_arr = rows['ttm'].values.astype(float)
    m_arr = rows['mid_price'].values.astype(float)
    t_arr = rows['option_type'].values

    x0 = ([init_params['v0'], init_params['kappa'],
            init_params['theta'], init_params['sigma'], init_params['rho']]
           if init_params else [0.04, 1.5, 0.04, 0.30, -0.50])

    def objective(params: np.ndarray) -> float:
        v0, kappa, theta, sigma, rho = params
        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e8
        err = 0.0
        for K, T, mid, otype in zip(K_arr, T_arr, m_arr, t_arr):
            try:
                call_p, put_p = heston_call_analytical(
                    spot, K, r, T, v0, kappa, theta, sigma, rho)
                model = call_p if otype == 'call' else put_p
                err += (model - mid) ** 2
            except Exception:
                err += 1e4
        return err

    bounds = [(1e-4, 1.0), (0.1, 10.0), (1e-4, 1.0), (0.01, 2.0), (-0.99, -0.01)]
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 200, 'ftol': 1e-6})
    if not res.success:
        return None
    v0, kappa, theta, sigma, rho = res.x
    return {'v0': float(v0), 'kappa': float(kappa), 'theta': float(theta),
            'sigma': float(sigma), 'rho': float(rho)}


# ---------------------------------------------------------------------------
# Merton jump-diffusion MLE on historical log-returns
# ---------------------------------------------------------------------------

def calibrate_merton_mle(log_returns: np.ndarray,
                          dt: float = 1 / 252,
                          n_terms: int = 20) -> dict | None:

    if len(log_returns) < 30:
        return None

    ns = np.arange(n_terms, dtype=float)
    log_factorials = np.array([math.lgamma(n + 1) for n in ns])

    def neg_log_likelihood(params: np.ndarray) -> float:
        sigma, lam, mu_j, sigma_j = params
        if sigma <= 0 or lam < 0 or sigma_j <= 0:
            return 1e10

        lam_dt = max(lam * dt, 1e-300)
        log_lam_dt = math.log(lam_dt)
        # log P(N=n) = n*log(lambda*dt) - lambda*dt - log(n!)
        log_pn = ns * log_lam_dt - lam * dt - log_factorials

        drift = (-0.5 * sigma ** 2 * dt
                 - lam * (math.exp(mu_j + 0.5 * sigma_j ** 2) - 1) * dt)
        mu_ns = drift + ns * mu_j
        var_ns = np.maximum(sigma ** 2 * dt + ns * sigma_j ** 2, 1e-10)

        ll = 0.0
        for r_t in log_returns:
            log_phi = (-0.5 * np.log(2 * math.pi * var_ns)
                       - 0.5 * (r_t - mu_ns) ** 2 / var_ns)
            ll += float(np.logaddexp.reduce(log_pn + log_phi))
        return -ll

    x0 = np.array([0.15, 0.10, -0.05, 0.08])
    bounds = [(0.01, 2.0), (0.0, 20.0), (-2.0, 0.5), (0.01, 1.5)]
    res = minimize(neg_log_likelihood, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 300, 'ftol': 1e-7})
    if not res.success:
        return None
    sigma, lam, mu_j, sigma_j = res.x
    return {
        'lambda_jump':     float(lam),
        'mu_jump':         float(mu_j),
        'sigma_jump':      float(sigma_j),
        'diffusion_sigma': float(sigma),
    }


# ---------------------------------------------------------------------------
# Bates model (Heston + compound Poisson jumps) pricing and calibration
# ---------------------------------------------------------------------------

def _bates_cf(u: complex, S: float, K: float, r: float, T: float,
              v0: float, kappa: float, theta: float,
              sigma: float, rho: float,
              lam: float, mu_j: float, sigma_j: float, j: int) -> complex:

    heston = _heston_cf(u, S, K, r, T, v0, kappa, theta, sigma, rho, j)
    jump = np.exp(lam * T * (np.exp(1j * u * mu_j - 0.5 * u ** 2 * sigma_j ** 2) - 1.0))
    return heston * jump


def bates_call_analytical(S: float, K: float, r: float, T: float,
                           v0: float, kappa: float, theta: float,
                           sigma: float, rho: float,
                           lam: float, mu_j: float, sigma_j: float) -> tuple:

    def _P(j: int) -> float:
        cf_vals = np.array([
            _bates_cf(complex(u), S, K, r, T, v0, kappa, theta,
                      sigma, rho, lam, mu_j, sigma_j, j)
            for u in _GL_NODES
        ])
        integrand_vals = np.real(cf_vals / (1j * _GL_NODES))
        return 0.5 + float(np.dot(_GL_WEIGHTS_EXP, integrand_vals)) / math.pi

    P1, P2 = _P(1), _P(2)
    call = max(S * P1 - K * math.exp(-r * T) * P2, 0.0)
    put = max(call - S + K * math.exp(-r * T), 0.0)
    return float(call), float(put)


def calibrate_bates_nls(chain_df: pd.DataFrame, spot: float, r: float,
                         merton_params: dict,
                         init_heston: dict | None = None) -> dict | None:

    lam    = merton_params.get('lambda_jump', 0.1)
    mu_j   = merton_params.get('mu_jump', -0.05)
    sigma_j = merton_params.get('sigma_jump', 0.1)

    rows = chain_df.dropna(subset=['strike', 'ttm', 'mid_price', 'option_type'])
    rows = rows[(rows['ttm'] > 1 / 365) & (rows['mid_price'] > 0.05)]
    if len(rows) < 5:
        return None

    K_arr = rows['strike'].values.astype(float)
    T_arr = rows['ttm'].values.astype(float)
    m_arr = rows['mid_price'].values.astype(float)
    t_arr = rows['option_type'].values

    x0 = ([init_heston['v0'], init_heston['kappa'],
            init_heston['theta'], init_heston['sigma'], init_heston['rho']]
           if init_heston else [0.04, 1.5, 0.04, 0.30, -0.50])

    def objective(params: np.ndarray) -> float:
        v0, kappa, theta, sigma, rho = params
        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e8
        err = 0.0
        for K, T, mid, otype in zip(K_arr, T_arr, m_arr, t_arr):
            try:
                call_p, put_p = bates_call_analytical(
                    spot, K, r, T, v0, kappa, theta, sigma, rho, lam, mu_j, sigma_j)
                model = call_p if otype == 'call' else put_p
                err += (model - mid) ** 2
            except Exception:
                err += 1e4
        return err

    bounds = [(1e-4, 1.0), (0.1, 10.0), (1e-4, 1.0), (0.01, 2.0), (-0.99, -0.01)]
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 200, 'ftol': 1e-6})
    if not res.success:
        return None
    v0, kappa, theta, sigma, rho = res.x
    return {'v0': float(v0), 'kappa': float(kappa), 'theta': float(theta),
            'sigma': float(sigma), 'rho': float(rho)}


# ---------------------------------------------------------------------------
# Cross-section filtering (applied before any calibration routine)
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
