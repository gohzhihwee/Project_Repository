from AlgorithmImports import *          # resolved via local stub
import numpy as np
import pandas as pd
import nolds
import math
from scipy.stats import norm
from scipy.special import ndtr

# Base seed for the deterministic RNG streams used in Monte Carlo pricing:
# every (model, date) simulation spawns its own SeedSequence from this.
RNG_BASE_SEED = 42

# Monte Carlo pricing configuration (OOS prices for BS/Merton/Heston/Bates).
# N_MC_PATHS: total paths per (model, date), as 5,000 antithetic pairs.
# MC_STEPS_PER_YEAR: time-step density of the simulation grid (4 steps per
#   trading day); the grid additionally contains every contract maturity
#   exactly. BS/Merton increments are exact at any step; for Heston/Bates at
#   the fitted vol-of-vol (ξ ≈ 1.7–2, Feller violated) daily Euler steps left a
#   +$0.12 mean bias vs the closed form, which 4 steps/day removes (residual
#   gap = MC noise, median SE ≈ $0.09 at 10k paths).
N_MC_PATHS = 10_000
MC_STEPS_PER_YEAR = 252 * 4

# MMAR trading-time cascade: 2^8 = 256 dyadic cells of one trading day each,
# i.e. a ~1-year base horizon that exceeds the longest (90-day) maturity, so
# θ(T) is random for every priced contract.
CASCADE_LEVELS = 8
CASCADE_CELLS = 2 ** CASCADE_LEVELS
TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Legacy notebook helpers (Hurst by R/S segments, partition-function
# utilities). Not used by the pricing pipeline except calculate_hurst_for_segments,
# which feeds the strategy selector's regime features.
# ---------------------------------------------------------------------------

def segment_data(data, num_segments):

    len_segment = len(data) // num_segments
    return [data[i:i+len_segment] for i in range(0, len(data), len_segment) if len(data[i:i+len_segment]) == len_segment]

def calculate_hurst_for_segments(data, num_segments):

    segments = segment_data(data, num_segments)
    hurst_values = [nolds.hurst_rs(seg) for seg in segments]
    return hurst_values

def define_time_window(min_window:int, max_window:int, base:float=10, interval:float=0.25):

    window_sizes = list(map(
        lambda x: int(base**x),
        np.arange(math.log10(min_window), math.log10(max_window), interval)))

    return window_sizes

def calculate_scaling_exponent(delta, x_t, q):

    Fq = [[0 for x in range(len(delta))] for y in range(len(q))]

    for k in range (0, len(q)):
        if k%30==0:
            print("calculating q=" + str(k) + ' out of ' + str(len(q)-1))

        for j in range (0,len(delta)):

            for i in range (0,len(x_t)-1):
                if i < int((len(x_t)-1)/delta[j]):
                    Fq[k][j]=Fq[k][j] + abs(x_t[i*delta[j]+delta[j]]-x_t[i*delta[j]])**q[k]

    Fq=pd.DataFrame(Fq)

    for i in range(0,len(q)):
        Fq.rename(index={Fq.index[i]:q[i]}, inplace=True)
    for i in range(len(delta)-1,-1,-1):
        Fq.rename(columns={Fq.columns[i]:delta[i]}, inplace=True)

    tau_q_list = []
    for i,row in Fq.iterrows():
        # Add a small epsilon to avoid log of zero or negative numbers and explicitly convert to float
        Fq_matrix = np.vstack([np.log10((row.values + 1e-9).astype(float)), np.ones(len(row))]).T
        tau_q, c = np.linalg.lstsq(Fq_matrix, np.log10(delta), rcond=-1)[0]
        tau_q_list.append(tau_q)

    return Fq, tau_q_list


def estimate_multifractal_spectrum(tau_q_list:list, q:list, start_of_list:int, end_of_list:int):

    tau_q_estimated = np.polyfit(q[start_of_list:end_of_list], tau_q_list[start_of_list:end_of_list], 2)

    F_A = [0 for x in range(len(q)-10)]
    p = [0 for x in range(len(q)-10)]

    a = tau_q_estimated[0]
    b = tau_q_estimated[1]
    c = tau_q_estimated[2]

    for i in range(0, len(q)-10):
        p[i] = 2*a*q[i]+b
        F_A[i] = ((p[i]-b)/(2*a))*p[i] - (a*((p[i]-b)/(2*a))**2 + b*((p[i]-b)/(2*a)) + c)

    F_A = pd.DataFrame(F_A)
    F_A.rename(columns={F_A.columns[0]:"f(a)"}, inplace=True)
    F_A['p'] = p

    width_of_spectrum = 1/(4*a)
    holder_exponent = (-2*b)/(4*a)
    asymmetry_of_spectrum = (-4*a*c+b**2)/(4*a)


    return F_A, (width_of_spectrum, holder_exponent, asymmetry_of_spectrum)

def calculate_lognormal_cascade(layers:int, v:float, ln_lambda:float, ln_sigma:float):

    layers = layers - 1

    m0 = np.random.lognormal(ln_lambda,ln_sigma)
    m1 = np.random.lognormal(ln_lambda,ln_sigma)
    m0 = m0/(m0+m1)
    m1 = m1/(m0+m1)

    M = [m0, m1]

    if (layers >= 0):
        d=[0 for x in range(0,2)]
        for i in range(0,2):
            d[i] = calculate_lognormal_cascade(layers, (M[i]*v), ln_lambda, ln_sigma)

        v = d

    return v

def calculate_trading_time(layers:int, lognormal_cascade:list):

    trading_time = 2**layers*np.cumsum(lognormal_cascade)/sum(lognormal_cascade)
    return trading_time


# ---------------------------------------------------------------------------
# Returns-based Hurst exponent (MMAR partition-function estimator)
# ---------------------------------------------------------------------------

def estimate_hurst_partition(log_returns: np.ndarray,
                             q_grid: np.ndarray | None = None,
                             n_deltas: int = 12) -> dict:
    """
    MMAR partition-function estimator (Calvet & Fisher 2002). For block
    length Δt, S_q(Δt) = Σ_i |X(iΔt+Δt) − X(iΔt)|^q over non-overlapping
    blocks scales as Δt^{τ(q)} (the block count contributes Δt^{-1} to
    E|X(Δt)|^q ∝ Δt^{τ(q)+1}). τ(q) is the OLS slope of log S_q on log Δt;
    H = 1/q* where τ(q*) = 0, consistent with τ(q)+1 = qH at the root.
    Brownian motion gives τ(q) = q/2 − 1, q* = 2, H = ½.

    Returns {'hurst', 'q_star', 'q', 'tau'}; 'hurst' is NaN if τ has no root
    on the q grid.
    """
    r = np.asarray(log_returns, dtype=float)
    r = r[np.isfinite(r)]
    if q_grid is None:
        q_grid = np.arange(0.5, 6.0001, 0.05)
    n = len(r)
    deltas = np.unique(np.logspace(0, math.log10(max(n // 10, 2)), n_deltas).astype(int))
    log_S = np.empty((len(q_grid), len(deltas)))
    for j, d in enumerate(deltas):
        nb = n // d
        blocks = np.abs(r[:nb * d].reshape(nb, d).sum(axis=1))
        log_S[:, j] = np.log(np.sum(blocks[None, :] ** q_grid[:, None], axis=1))
    log_d = np.log(deltas)
    x = log_d - log_d.mean()
    tau = (log_S - log_S.mean(axis=1, keepdims=True)) @ x / (x @ x)
    crossing = np.where((tau[:-1] < 0) & (tau[1:] >= 0))[0]
    if len(crossing) == 0:
        return {'hurst': float('nan'), 'q_star': float('nan'), 'q': q_grid, 'tau': tau}
    i = crossing[0]
    q_star = q_grid[i] - tau[i] * (q_grid[i + 1] - q_grid[i]) / (tau[i + 1] - tau[i])
    return {'hurst': float(1.0 / q_star), 'q_star': float(q_star), 'q': q_grid, 'tau': tau}


# ---------------------------------------------------------------------------
# Monte Carlo pricing: one vectorised simulation per (model, date), shared by
# every contract on that date (common random numbers across contracts).
# ---------------------------------------------------------------------------

def _simulation_grid(T: np.ndarray) -> np.ndarray:
    """Daily steps up to max(T), plus every contract maturity exactly."""
    T_max = float(np.max(T))
    daily = np.arange(1, int(math.ceil(T_max * MC_STEPS_PER_YEAR)) + 1) / MC_STEPS_PER_YEAR
    return np.unique(np.concatenate([daily[daily < T_max], np.asarray(T, dtype=float)]))


def _antithetic(rng, half: int) -> np.ndarray:
    z = rng.standard_normal(half)
    return np.concatenate([z, -z])


def simulate_log_paths(model: str, params: dict, S0: float, r: float, q: float,
                       grid: np.ndarray, n_paths: int, rng) -> np.ndarray:
    """
    Simulate ln S on `grid` (years, increasing, excluding 0) under the
    risk-neutral measure with dividend yield q. Returns (len(grid), n_paths).

      BS     : exact lognormal increments.
      Merton : exact increments — diffusion plus compound-Poisson log-normal
               jumps, drift compensated by λk.
      Heston : log-Euler for S, full-truncation Euler for v (v⁺ = max(v,0)
               in drift and diffusion), corr(dW_S, dW_v) = ρ.
      Bates  : Heston + Merton jumps.
    Antithetic pairs: normals are mirrored; Poisson counts are shared.
    """
    half = n_paths // 2
    n = 2 * half
    log_s = np.full(n, math.log(S0))
    out = np.empty((len(grid), n))
    has_jumps = model in ('merton', 'bates')
    stoch_vol = model in ('heston', 'bates')

    if has_jumps:
        lam, mu_j, sig_j = params['lambda_jump'], params['mu_jump'], params['sigma_jump']
        jump_comp = lam * (math.exp(mu_j + 0.5 * sig_j ** 2) - 1.0)
    else:
        jump_comp = 0.0
    if stoch_vol:
        v = np.full(n, params['v0'])
        kappa, theta, xi, rho = params['kappa'], params['theta'], params['sigma'], params['rho']
        rho_c = math.sqrt(1.0 - rho ** 2)
    else:
        sigma = params['sigma']

    t_prev = 0.0
    for k, t in enumerate(grid):
        dt = t - t_prev
        t_prev = t
        z1 = _antithetic(rng, half)
        if stoch_vol:
            vp = np.maximum(v, 0.0)
            z2 = _antithetic(rng, half)
            log_s += (r - q - jump_comp - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
            v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * (rho * z1 + rho_c * z2)
        else:
            log_s += (r - q - jump_comp - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z1
        if has_jumps:
            n_j = rng.poisson(lam * dt, half)
            n_j = np.concatenate([n_j, n_j])
            zj = _antithetic(rng, half)
            log_s += n_j * mu_j + np.sqrt(n_j) * sig_j * zj
        out[k] = log_s
    return out


def mc_price_contracts(model: str, params: dict, S0: float, r: float, q: float,
                       K: np.ndarray, T: np.ndarray, is_call: np.ndarray,
                       n_paths: int, seed) -> tuple:
    """
    Price every contract of one date from a single path set.
    Returns (prices, standard_errors); the SE uses antithetic-pair averages.
    """
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    rng = np.random.default_rng(seed)
    grid = _simulation_grid(T)
    log_paths = simulate_log_paths(model, params, S0, r, q, grid, n_paths, rng)
    idx = np.searchsorted(grid, T)
    S_T = np.exp(log_paths[idx])                                  # (C, n)
    payoff = np.where(is_call[:, None], np.maximum(S_T - K[:, None], 0.0),
                      np.maximum(K[:, None] - S_T, 0.0))
    half = payoff.shape[1] // 2
    pair = 0.5 * (payoff[:, :half] + payoff[:, half:])
    disc = np.exp(-r * T)
    prices = disc * pair.mean(axis=1)
    se = disc * pair.std(axis=1, ddof=1) / math.sqrt(half)
    return prices, se


# ---------------------------------------------------------------------------
# Multifractal Model of Asset Returns: X(t) = B_H(θ(t)), conditional MC
# ---------------------------------------------------------------------------

def draw_cascade_normals(n_draws: int, rng) -> np.ndarray:
    """
    Standard normals driving the cascade multipliers: 2 per node over
    CASCADE_LEVELS levels. Drawn as antithetic pairs (rows z and −z).
    """
    half = n_draws // 2
    z = rng.standard_normal((half, 2 * (CASCADE_CELLS - 1)))
    return np.concatenate([z, -z])


def cascade_cell_masses(z: np.ndarray, cascade_sigma: float) -> np.ndarray:
    """
    Microcanonical lognormal cascade: at every node the mass splits between
    its two children in proportions M_i / (M_0 + M_1), ln M_i = s·Z_i. Returns
    (n_draws, CASCADE_CELLS) leaf masses, each row summing to 1; by symmetry
    each leaf has expected mass 1/CASCADE_CELLS.
    """
    n = z.shape[0]
    masses = np.ones((n, 1))
    offset = 0
    for level in range(CASCADE_LEVELS):
        m = 2 ** level
        w = np.exp(cascade_sigma * z[:, offset:offset + 2 * m].reshape(n, m, 2))
        offset += 2 * m
        w = w / w.sum(axis=2, keepdims=True)
        masses = (masses[:, :, None] * w).reshape(n, 2 * m)
    return masses


def mmar_trading_time(T: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    θ(T) in years for each maturity in T (years): cumulative cascade mass up
    to calendar position T·252 trading days, with linear interpolation within
    a cell, scaled so E[θ(T)] = T. Returns (len(T), n_draws).
    """
    cum = np.concatenate([np.zeros((masses.shape[0], 1)), np.cumsum(masses, axis=1)], axis=1)
    cum *= CASCADE_CELLS / TRADING_DAYS_PER_YEAR
    pos = np.clip(np.asarray(T, dtype=float) * TRADING_DAYS_PER_YEAR, 0.0, CASCADE_CELLS)
    j = np.minimum(np.floor(pos).astype(int), CASCADE_CELLS - 1)
    frac = pos - j
    return (cum[:, j] + frac[None, :] * (cum[:, j + 1] - cum[:, j])).T


def mmar_price_vec(S: float, K: np.ndarray, T: np.ndarray, r: float, q: float,
                   sigma: float, hurst: float, cascade_sigma: float,
                   is_call: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    MMAR European option price by conditional Monte Carlo.

    X(t) = ln S_t − ln S_0 − drift = σ·B_H(θ(t)), with B_H a fractional
    Brownian motion independent of the cascade trading time θ. Because a
    European payoff depends only on S_T and B_H is H-self-similar, conditional
    on θ(T) = u the log-price is exactly Gaussian with variance σ² u^{2H}; no
    path of B_H needs to be simulated. The drift is set so that, conditionally
    on u, E[S_T | u] = S e^{(r−q)T} (forward/martingale normalisation), so each
    conditional price is a Black-Scholes price with total standard deviation
    σ u^H, and the MMAR price is its average over the cascade draws `z`.

    Reduces to Black-Scholes(σ) when H = ½ and cascade_sigma = 0.
    """
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    T_u, inv = np.unique(T, return_inverse=True)
    masses = cascade_cell_masses(z, cascade_sigma)
    theta = mmar_trading_time(T_u, masses)                        # (U, n)
    total_sd = sigma * np.power(np.maximum(theta, 1e-12), hurst)  # (U, n)
    prices = np.empty(len(K))
    for u_idx in range(len(T_u)):
        sel = inv == u_idx
        p = _bs_price_matrix(S, K[sel], T_u[u_idx], r, q, total_sd[u_idx], is_call[sel])
        prices[sel] = p.mean(axis=1)
    return prices


def _bs_price_matrix(S: float, K: np.ndarray, T: float, r: float, q: float,
                     total_sd: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """BS prices for contracts K (C,) at one maturity across draws total_sd (n,) → (C, n)."""
    fwd = S * math.exp((r - q) * T)
    sd = total_sd[None, :]
    lnFK = np.log(fwd / K)[:, None]
    d1 = lnFK / sd + 0.5 * sd
    d2 = d1 - sd
    disc = math.exp(-r * T)
    call = disc * (fwd * ndtr(d1) - K[:, None] * ndtr(d2))
    put = call - disc * (fwd - K[:, None])
    return np.where(is_call[:, None], call, put)


# ---------------------------------------------------------------------------
# Black-Scholes Greeks (hedging and the strategy-selector target)
# ---------------------------------------------------------------------------

def bs_delta(S, K, r, sigma, T, option_type='call', q=0.0):
    """
    Black-Scholes delta. Used as the universal hedge ratio across all pricing models.

    Args:
        S: Spot price
        K: Strike price
        r: Risk-free rate
        sigma: Annualized volatility
        T: Time to maturity in years
        option_type: 'call' or 'put'
        q: Continuous dividend yield

    Returns:
        Delta in [-1, 1]. Positive for calls, negative for puts.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if option_type == 'call':
            return 1.0 if S >= K else 0.0
        return -1.0 if S <= K else 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    disc_q = math.exp(-q * T)
    if option_type == 'call':
        return float(disc_q * norm.cdf(d1))
    return float(disc_q * (norm.cdf(d1) - 1.0))


def bs_theta(S, K, r, sigma, T, option_type='call', q=0.0):
    """Black-Scholes theta per year (∂V/∂t, negative for a decaying long option)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    disc_q, disc_r = math.exp(-q * T), math.exp(-r * T)
    decay = -S * disc_q * norm.pdf(d1) * sigma / (2.0 * sqrt_T)
    if option_type == 'call':
        return float(decay - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1))
    return float(decay + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1))
