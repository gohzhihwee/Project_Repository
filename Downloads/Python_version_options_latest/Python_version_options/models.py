from AlgorithmImports import *          # resolved via local stub
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import nolds
from tqdm import tqdm
import math
from scipy.stats import norm

def bs_call_mc(S, K, r, sigma, T, t, Ite):
  z = np.random.normal(0, 1, Ite) # Generate z as a 1D array
  # Ensure S is treated as a scalar
  ST = S * np.exp((T - t)*(r - 0.5 * sigma**2) + sigma * np.sqrt(T - t) * z)
  # Calculate payoffs directly as a 1D array
  payoffs = np.maximum(0, ST - K)

  average_payoff = np.sum(payoffs) / float(Ite) # Calculate average of payoffs

  return np.exp(-r * (T - t)) * average_payoff

def bs_put_mc(S, K, r, sigma, T, t, Ite):
  z = np.random.normal(0, 1, Ite) # Generate z as a 1D array
  # Ensure S is treated as a scalar
  ST = S * np.exp((T - t)*(r - 0.5 * sigma**2) + sigma * np.sqrt(T - t) * z)
  # Calculate payoffs directly as a 1D array
  payoffs = np.maximum(0, K - ST)

  average_payoff = np.sum(payoffs) / float(Ite) # Calculate average of payoffs

  return np.exp(-r * (T - t)) * average_payoff

def SDE_vol(v0, kappa, theta, sigma, T, M, Ite, rand, row, cho_matrix):
  dt = T/M
  v = np.zeros((M + 1, Ite), dtype=float)
  v[0] = v0
  sdt = np.sqrt(dt)
  for t in range(1, M + 1):
    ran = np.dot(cho_matrix, rand[:, t])
    # Ensure non-negativity of volatility
    v[t] = np.maximum(0, v[t - 1] + kappa * (theta - v[t - 1]) * dt + sigma * sdt * ran[row])
  return v

def Heston_paths(S0, r, v_paths, T, M, Ite, rand, row, cho_matrix):
  dt = T/M
  S = np.zeros((M + 1, Ite), dtype=float)
  S[0] = S0
  sdt = np.sqrt(dt)
  for t in range(1, M + 1, 1):
    ran = np.dot(cho_matrix, rand[:, t])
    # Use the simulated volatility path
    S[t] = S[t - 1] * np.exp((r - 0.5 * v_paths[t]) * dt + np.sqrt(v_paths[t]) * sdt * ran[row])
  return S

def random_number_gen(M, Ite):
  # Generate 2 sets of random numbers for correlated simulation
  rand = np.random.standard_normal((2, M + 1, Ite))
  return rand

# The following simplified MC functions will be replaced later with functions
# that use the simulated paths. Keeping them for now to avoid breaking subsequent cells.
def heston_call_mc(S, K, r, T, t):
  # Placeholder: This should ideally use simulated paths
  payoff = np.maximum(0, S - K)
  average = payoff
  return np.exp(-r * (T - t)) * average

def heston_put_mc(S, K ,r, T, t):
  # Placeholder: This should ideally use simulated paths
  payoff = np.maximum(0, K - S)
  average = payoff
  return np.exp(-r * (T - t)) * average

def merton_call_mc(S, K, r, T, t):
  # Placeholder: This should ideally use simulated paths with jumps
  payoff = np.maximum(0, S - K)
  average = payoff
  return np.exp(-r * (T - t)) * average

def merton_put_mc(S, K ,r, T, t):
  # Placeholder: This should ideally use simulated paths with jumps
  payoff = np.maximum(0, K - S)
  average = payoff
  return np.exp(-r * (T - t)) * average

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

def calculate_magnitude_parameter(initial_value:float, eps:float, steps:float, number_of_path:int, real_std:float, layers:int, hurst_exponent:float):

    diff = np.inf
    magnitude_parameter = initial_value

    while abs(diff) > eps:
        std_list = []
        for nb in range(number_of_path): # excluding tqdm for a less verbose output
            n_steps = 10*2**layers+1
            fbm_simulation = generate_fbm_path(n_steps, hurst_exponent, dt=1, s0=0)
            fbm_simulation = fbm_simulation * magnitude_parameter  # Scale by magnitude_parameter
            std_list.append(np.std(fbm_simulation))
        diff = real_std - np.median(std_list)
        print('Diff: ', diff)
        if abs(diff) > eps:
            magnitude_parameter += diff * steps
            print('new magnitude_parameter:', magnitude_parameter)

    return  magnitude_parameter

def calculate_mmar_returns(S0:float, number_of_path:int, layers:int, hurst_exponent:float, trading_time:list, magnitude_parameter:float, time_window_base:float=10):

    mmar_returns = []
    mmar_prices = []

    for nb in tqdm(range(number_of_path)):
        n_steps = 10*2**layers+1
        fbm_simulation = generate_fbm_path(n_steps, hurst_exponent, dt=1, s0=0)
        fbm_simulation = fbm_simulation * magnitude_parameter  # Scale by magnitude_parameter
        fbm_simulation = fbm_simulation[1:]

        simulated_xt_array = [0 for x in range(0, len(trading_time))]
        for i in range(0, len(trading_time)):
            idx = int(min(trading_time[i]*10, len(fbm_simulation)-1))
            simulated_xt_array[i] = fbm_simulation[idx]
        mmar_returns.append(simulated_xt_array)

        simulated_prices_array = S0 * np.exp(simulated_xt_array)
        mmar_prices.append(simulated_prices_array)

    return mmar_returns, mmar_prices

def option_pricer(paths, strike, r, T, option_type='call'):

    if isinstance(paths, list):
        paths = np.array(paths)

    S_T = paths[:, -1]

    if option_type == 'call':
        payoffs = np.maximum(S_T - strike, 0)
    elif option_type == 'put':
        payoffs = np.maximum(strike - S_T, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price # Return the calculated option price

def generate_fbm_path(n, hurst, dt=1, s0=1):

    dW = np.random.randn(n)

    increments = dW * (dt**(hurst))

    fbm_path = np.cumsum(increments)

    fbm_path = fbm_path - fbm_path[0] + s0

    return fbm_path

def generate_multiple_paths(num_paths, n, hurst, dt=1, s0=1):

    paths = []

    for _ in range(num_paths):
        prices = generate_fbm_path(n, hurst, dt, s0)
        prices = np.where(prices > 0, prices, 0)
        paths.append(prices)

    return paths

def price_options_for_strikes(paths, center=80, step=5, num_strikes=5, r=0.05, T=1):

    option_prices = {}

    for i in range(1, num_strikes + 1):
        strike = center - i * step
        option_prices[strike] = option_pricer(paths, strike, r, T, option_type='put')

    for i in range(num_strikes + 1):
        strike = center + i * step
        option_prices[strike] = option_pricer(paths, strike, r, T, option_type='call')

    return option_prices

# Helper function to calculate Black-Scholes price for a single option
def calculate_bs_price(S, K, r, sigma, T, num_paths):
    t = 0 # Calculation starts at time t=0
    return bs_call_mc(S, K, r, sigma, T, t, num_paths), bs_put_mc(S, K, r, sigma, T, t, num_paths)

def _sample_cascade(layers: int, v: float, ln_lambda: float, ln_sigma: float) -> np.ndarray:
    """Recursively sample a lognormal cascade, returning a flat array of 2^layers leaf values."""
    if layers == 0:
        return np.array([v])
    m0 = np.random.lognormal(ln_lambda, ln_sigma)
    m1 = np.random.lognormal(ln_lambda, ln_sigma)
    total = m0 + m1
    m0, m1 = (m0 / total) * v, (m1 / total) * v
    return np.concatenate([
        _sample_cascade(layers - 1, m0, ln_lambda, ln_sigma),
        _sample_cascade(layers - 1, m1, ln_lambda, ln_sigma),
    ])


def calculate_mmar_price(S0: float, K: float, r: float, T: float,
                         hurst: float, num_paths: int,
                         n_steps_per_year: int = 252) -> tuple:
    """
    MMAR (Multifractal Model of Asset Returns) option pricing.
    Uses a lognormal cascade to generate multifractal trading time, then
    subordinates an fBM path to that time-change before discounting payoffs.
    """
    n_steps = max(int(n_steps_per_year * T), 2)
    # Cascade depth 5 gives 32 time buckets — fast enough for per-contract intraday calls
    CASCADE_LAYERS = 5
    # Normalised cascade parameters: E[M_i]=1, typical multifractal width for equity
    ln_lambda = -0.5 * np.log(2)
    ln_sigma  = np.sqrt(np.log(2) / 2.0)

    mmar_paths = []
    for _ in range(num_paths):
        # 1. Generate multifractal trading time on [0,1]
        flat_cascade  = _sample_cascade(CASCADE_LAYERS, 1.0, ln_lambda, ln_sigma)
        trading_time  = np.cumsum(flat_cascade) / flat_cascade.sum()  # shape (2^CASCADE_LAYERS,)
        n_tt          = len(trading_time)

        # 2. Generate an fBM path at T-scaled increments
        fbm = generate_fbm_path(n_steps + 1, hurst, dt=T / n_steps, s0=0.0)

        # 3. Subordinate: at each real-time step i/n_steps, find the multifractal
        #    trading time mt and index into the fBM path
        path = np.empty(n_steps + 1)
        path[0] = S0
        drift = r - 0.5 * (hurst * 0.2) ** 2  # approximate drift term
        for i in range(1, n_steps + 1):
            real_t  = i / n_steps                              # in [0, 1]
            tt_idx  = min(int(real_t * n_tt), n_tt - 1)
            mt      = trading_time[tt_idx]                     # multifractal time in [0, 1]
            fbm_idx = min(int(mt * n_steps), n_steps)
            log_ret = fbm[fbm_idx] + drift * T * real_t
            path[i] = S0 * np.exp(log_ret)

        mmar_paths.append(np.maximum(path, 0.0))

    call_price = option_pricer(mmar_paths, K, r, T, option_type='call')
    put_price  = option_pricer(mmar_paths, K, r, T, option_type='put')
    return call_price, put_price

# Helper function to calculate Heston price for a single option (using improved MC simulation)
def calculate_heston_price(S0, K, r, T, heston_params, num_paths, n_steps_per_year=252):
    M = int(n_steps_per_year * T) # Number of steps
    if M < 1: # Ensure at least one step for simulation
        M = 1
    Ite = num_paths # Number of iterations

    # Generate random numbers and Cholesky matrix for correlated simulation
    # Generate 2 sets of random numbers for correlated simulation
    rand = np.random.standard_normal((2, M + 1, Ite))
    # Calculate Cholesky decomposition of the covariance matrix for price and volatility
    # Assuming a 2x2 covariance matrix where the diagonal elements are 1 and rho is the off-diagonal
    cov_matrix = np.array([[1, heston_params['rho']], [heston_params['rho'], 1]])
    # Ensure the covariance matrix is positive semi-definite before Cholesky decomposition
    try:
        cho_matrix = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        print("Warning: Covariance matrix is not positive semi-definite. Using diagonal matrix.")
        cho_matrix = np.eye(2) # Use identity matrix if decomposition fails


    # Simulate volatility paths
    v_paths = SDE_vol(heston_params['v0'], heston_params['kappa'], heston_params['theta'], heston_params['sigma'], T, M, Ite, rand, 1, cho_matrix) # row 1 for volatility

    # Simulate asset price paths using the simulated volatility
    S_paths = Heston_paths(S0, r, v_paths, T, M, Ite, rand, 0, cho_matrix) # row 0 for price


    # Calculate option prices from the simulated paths
    call_price = option_pricer(S_paths, K, r, T, option_type='call')
    put_price = option_pricer(S_paths, K, r, T, option_type='put')

    return call_price, put_price


# Helper function to calculate Merton price for a single option (needs proper MC implementation)
def calculate_merton_price(S0, K, r, T, merton_params, num_paths, annualized_volatility, n_steps_per_year=252):
    M = int(n_steps_per_year * T) # Number of steps
    if M < 1: # Ensure at least one step for simulation
        M = 1
    Ite = num_paths # Number of iterations
    dt = T / M

    lambda_jump = merton_params['lambda_jump']
    mu_jump = merton_params['mu_jump']
    sigma_jump = merton_params['sigma_jump']

    # Calculate compensated drift for the continuous part
    # This ensures the expected return is r
    compensated_drift = r - 0.5 * annualized_volatility**2 - lambda_jump * (np.exp(mu_jump + 0.5 * sigma_jump**2) - 1)


    S_paths = np.zeros((M + 1, Ite), dtype=float)
    S_paths[0] = S0

    for i in range(Ite):
        for t in range(M):
            # Continuous part (Brownian Motion)
            dW = np.random.normal(0, np.sqrt(dt))
            continuous_return = compensated_drift * dt + annualized_volatility * dW # Using annualized_volatility as the volatility for the continuous part

            # Jump part (Poisson process + jump size)
            num_jumps = np.random.poisson(lambda_jump * dt) # Number of jumps in this small time step
            jump_return = 0
            if num_jumps > 0:
                # Sum of jump sizes (log-normally distributed)
                jump_sizes = np.random.normal(mu_jump, sigma_jump, num_jumps)
                jump_return = np.sum(jump_sizes)


            # Total return for the time step
            total_return = continuous_return + jump_return

            # Update price
            S_paths[t + 1, i] = S_paths[t, i] * np.exp(total_return)

    # Ensure prices are non-negative
    S_paths = np.maximum(0, S_paths)


    # Calculate option prices from the simulated paths
    call_price = option_pricer(S_paths, K, r, T, option_type='call')
    put_price = option_pricer(S_paths, K, r, T, option_type='put')

    return call_price, put_price

# Helper function to calculate Bates price for a single option (combining Heston and Merton)
def calculate_bates_price(S0, K, r, T, heston_params, merton_params, num_paths, n_steps_per_year=252):
    M = int(n_steps_per_year * T) # Number of steps
    if M < 1:
        M = 1
    Ite = num_paths
    dt = T / M

    # Heston parameters
    v0 = heston_params['v0']
    kappa = heston_params['kappa']
    theta = heston_params['theta']
    sigma_heston = heston_params['sigma'] # Renamed to avoid conflict
    rho = heston_params['rho']

    # Merton parameters
    lambda_jump = merton_params['lambda_jump']
    mu_jump = merton_params['mu_jump']
    sigma_jump = merton_params['sigma_jump']

    # Calculate Cholesky decomposition for correlated price and volatility Brownian motions
    cov_matrix = np.array([[1, rho], [rho, 1]])
    try:
        cho_matrix = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        print("Warning: Covariance matrix is not positive semi-definite for Bates. Using diagonal matrix.")
        cho_matrix = np.eye(2)


    S_paths = np.zeros((M + 1, Ite), dtype=float)
    v_paths = np.zeros((M + 1, Ite), dtype=float)
    S_paths[0] = S0
    v_paths[0] = v0

    # Compensated drift for the continuous price process under Bates
    # The drift depends on the current volatility and the jump component
    # The total expected return should be r
    # The continuous part's drift is r - 0.5*v[t] - lambda_jump * (exp(mu_jump + 0.5*sigma_jump^2) - 1)
    jump_compensation = lambda_jump * (np.exp(mu_jump + 0.5 * sigma_jump**2) - 1)


    for i in range(Ite):
        # Generate all random numbers for this path at once for efficiency
        rand_vol = np.random.normal(0, np.sqrt(dt), M)
        rand_price = np.random.normal(0, np.sqrt(dt), M)
        jump_poisson = np.random.poisson(lambda_jump * dt, M)
        # Generate jump sizes only if jumps occur (max number of jumps is M in this simplified approach)
        max_possible_jumps = np.sum(jump_poisson)
        if max_possible_jumps > 0:
             jump_sizes_rv = np.random.normal(mu_jump, sigma_jump, max_possible_jumps)
        else:
             jump_sizes_rv = np.array([])

        jump_size_idx = 0

        # Corrected loop range: iterate from 0 to M-1 to update indices 1 to M
        for t in range(M):
            # Correlated Brownian Motions
            correlated_rand = np.dot(cho_matrix, np.array([rand_price[t], rand_vol[t]]))
            dW1 = correlated_rand[0] # For price
            dW2 = correlated_rand[1] # For volatility

            # Simulate Volatility (Heston part)
            v_paths[t + 1, i] = np.maximum(0, v_paths[t, i] + kappa * (theta - v_paths[t, i]) * dt + sigma_heston * dW2 * np.sqrt(v_paths[t, i]))

            # Simulate Price (Heston + Merton parts)
            continuous_drift = r - 0.5 * v_paths[t + 1, i] - jump_compensation # Use updated volatility
            continuous_return = continuous_drift * dt + np.sqrt(v_paths[t + 1, i]) * dW1 # Use updated volatility


            # Jump part
            num_jumps_step = jump_poisson[t]
            jump_return = 0
            if num_jumps_step > 0:
                # Sum jump sizes for this step
                current_jump_sizes = jump_sizes_rv[jump_size_idx : jump_size_idx + num_jumps_step]
                jump_return = np.sum(current_jump_sizes)
                jump_size_idx += num_jumps_step


            # Total return
            total_return = continuous_return + jump_return

            # Update price
            S_paths[t + 1, i] = S_paths[t, i] * np.exp(total_return)


    # Ensure prices are non-negative
    S_paths = np.maximum(0, S_paths)


    # Calculate option prices from the simulated paths
    call_price = option_pricer(S_paths, K, r, T, option_type='call')
    put_price = option_pricer(S_paths, K, r, T, option_type='put')

    return call_price, put_price

def _calculate_bs_price_worker(S, K, r, sigma, T, num_paths):
    """Thread-safe wrapper for BS pricing"""
    t = 0
    return ('BS', calculate_bs_price(S, K, r, sigma, T, num_paths))

def _calculate_mmar_price_worker(S, K, r, T, hurst, num_paths):
    """Thread-safe wrapper for MMAR pricing"""
    return ('MMAR', calculate_mmar_price(S, K, r, T, hurst, num_paths))

def _calculate_heston_price_worker(S, K, r, T, heston_params, num_paths):
    """Thread-safe wrapper for Heston pricing"""
    return ('Heston', calculate_heston_price(S, K, r, T, heston_params, num_paths))

def _calculate_merton_price_worker(S, K, r, T, merton_params, num_paths, volatility):
    """Thread-safe wrapper for Merton pricing"""
    return ('Merton', calculate_merton_price(S, K, r, T, merton_params, num_paths, volatility))

def _calculate_bates_price_worker(S, K, r, T, heston_params, merton_params, num_paths):
    """Thread-safe wrapper for Bates pricing"""
    return ('Bates', calculate_bates_price(S, K, r, T, heston_params, merton_params, num_paths))

def calculate_all_model_prices_concurrent(S0, K, r, T, volatility, hurst, heston_params,
                                         merton_params, num_paths=20, max_workers=5, timeout=30):
    """
    Calculate all 5 model prices concurrently using ThreadPoolExecutor.

    Args:
        S0: Current spot price
        K: Strike price
        r: Risk-free rate
        T: Time to maturity (years)
        volatility: Annualized volatility
        hurst: Hurst exponent for MMAR
        heston_params: Dict with Heston parameters
        merton_params: Dict with Merton parameters
        num_paths: Number of MC paths
        max_workers: Maximum concurrent threads
        timeout: Timeout in seconds for completion

    Returns:
        Dictionary with model names as keys and (call_price, put_price) tuples as values
    """
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all pricing tasks
        future_to_model = {
            executor.submit(_calculate_bs_price_worker, S0, K, r, volatility, T, num_paths): 'BS',
            executor.submit(_calculate_mmar_price_worker, S0, K, r, T, hurst, num_paths): 'MMAR',
            executor.submit(_calculate_heston_price_worker, S0, K, r, T, heston_params, num_paths): 'Heston',
            executor.submit(_calculate_merton_price_worker, S0, K, r, T, merton_params, num_paths, volatility): 'Merton',
            executor.submit(_calculate_bates_price_worker, S0, K, r, T, heston_params, merton_params, num_paths): 'Bates',
        }

        # Collect results as they complete
        for future in as_completed(future_to_model, timeout=timeout):
            model_name, (call_price, put_price) = future.result()
            results[model_name] = {
                'call': float(call_price) if not np.isnan(call_price) else None,
                'put': float(put_price) if not np.isnan(put_price) else None
            }

    return results


def bs_delta(S, K, r, sigma, T, option_type='call'):
    """
    Black-Scholes delta. Used as the universal hedge ratio across all pricing models.

    Args:
        S: Spot price
        K: Strike price
        r: Risk-free rate
        sigma: Annualized volatility
        T: Time to maturity in years
        option_type: 'call' or 'put'

    Returns:
        Delta in [-1, 1]. Positive for calls, negative for puts.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if option_type == 'call':
            return 1.0 if S >= K else 0.0
        return -1.0 if S <= K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)