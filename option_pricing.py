"""
Utility functions for European option pricing with Monte Carlo.

This file is designed to be imported from a Jupyter notebook. It keeps the
code simple and readable while still covering the main parts of the project:

- Black-Scholes pricing for European calls and puts
- Monte Carlo pricing under risk-neutral GBM
- Confidence intervals and convergence analysis
- Volatility sensitivity analysis
- Greeks using finite differences
- Optional closed-form Black-Scholes Greeks for comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import exp, log, sqrt
from scipy.stats import norm



# -----------------------------------------------------------------------------
# Option payoffs and Black-Scholes price
# -----------------------------------------------------------------------------


def payoff(ST, params):
    """
    Payoff of a European call or put option at maturity.

    Parameters
    ----------
    ST : float or np.array
        Asset price at maturity.
    K : float
        Strike price.
    option_type : str
        "call" or "put".
    """

    K = params['K']
    option_type = params['option type']

    if option_type == "call":
        return np.maximum(ST - K, 0)
    elif option_type == "put":
        return np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")


def black_scholes_price(params):
    """
    Black-Scholes price for a European call or put option.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualized volatility.
    T : float
        Time to maturity in years.
    option_type : str
        "call" or "put".
    """

    S0 = params['S0']
    K = params['K']
    r = params['r']
    sigma = params['sigma']
    T = params['T']
    option_type = params['option type']

    if T <= 0:
        return float(payoff(S0, params))

    if sigma <= 0:
        forward_price = S0 * exp(r * T)
        return float(exp(-r * T) * payoff(forward_price, params))

    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "call":
        price = S0 * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    return float(price)


# -----------------------------------------------------------------------------
# Monte Carlo simulation
# -----------------------------------------------------------------------------


def simulate_terminal_prices(
    params,
    n_sim=100_000,
    seed=42,
    antithetic=False,
):
    """
    Simulate terminal stock prices under risk-neutral geometric Brownian motion.

    Since the option is European, pricing only requires the terminal price S_T.
    """
    rng = np.random.default_rng(seed)

    S0 = params['S0']
    r = params['r']
    sigma = params['sigma']
    T = params['T']

    if T <= 0:
        return np.full(n_sim, S0)

    if antithetic:
        half = n_sim // 2
        z_half = rng.standard_normal(half)
        z = np.concatenate([z_half, -z_half])

        if n_sim % 2 == 1:
            z = np.concatenate([z, rng.standard_normal(1)])
    else:
        z = rng.standard_normal(n_sim)

    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * sqrt(T) * z)
    return ST


def simulate_gbm_paths(params, n_paths=1_000, n_steps=252, seed=42):
    """
    Simulate full GBM paths under the risk-neutral measure.

    This is mainly useful for visualization. For pricing a European option,
    simulate_terminal_prices is more direct.
    """
    rng = np.random.default_rng(seed)

    S0 = params['S0']
    r = params['r']
    sigma = params['sigma']
    T = params['T']

    if T <= 0:
        return np.full((n_paths, 1), S0)

    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))

    log_returns = (r - 0.5 * sigma**2) * dt + sigma * sqrt(dt) * z
    log_paths = np.cumsum(log_returns, axis=1)
    log_paths = np.column_stack([np.zeros(n_paths), log_paths])

    paths = S0 * np.exp(log_paths)
    return paths


def monte_carlo_price(
    params,
    n_sim=100_000,
    seed=42,
    antithetic=False,
):
    """
    Estimate the price of a European option using Monte Carlo simulation.

    Returns a dictionary with the price, standard error and 95% confidence
    interval.
    """

    r = params['r']
    T = params['T']

    ST = simulate_terminal_prices(
        params = params,
        n_sim=n_sim,
        seed=seed,
        antithetic=antithetic,
    )

    discounted_payoff = exp(-r * T) * payoff(ST, params)

    price = np.mean(discounted_payoff)
    standard_error = np.std(discounted_payoff, ddof=1) / sqrt(len(discounted_payoff))

    ci_low = price - 1.96 * standard_error
    ci_high = price + 1.96 * standard_error

    bs_price = black_scholes_price(params)

    return {
        "mc_price": float(price),
        "bs_price": float(bs_price),
        "absolute_error": float(abs(price - bs_price)),
        "standard_error": float(standard_error),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "n_sim": int(len(discounted_payoff)),
        "antithetic": bool(antithetic),
    }


# -----------------------------------------------------------------------------
# Analysis helpers
# -----------------------------------------------------------------------------


def convergence_study(
    params,
    path_grid=None,
    seed=42,
    antithetic=False,
):
    """
    Study how the Monte Carlo estimate changes as the number of simulations grows.
    """
    if path_grid is None:
        path_grid = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000]

    rows = []

    for i, n_sim in enumerate(path_grid):
        result = monte_carlo_price(
            params = params,
            n_sim=n_sim,
            seed=seed + i,
            antithetic=antithetic,
        )
        rows.append(result)

    df = pd.DataFrame(rows)
    cols = ["n_sim"] + [col for col in df.columns if col != "n_sim"]
    return df[cols]


def volatility_analysis(
    params,
    vol_grid,
    n_sim=100_000,
    seed=42,
    antithetic=False,
):
    """
    Compare Black-Scholes and Monte Carlo prices across different volatilities.
    """
    rows = []

    for i, sigma in enumerate(vol_grid):
        params_vol = params.copy()
        params_vol['sigma'] = float(sigma)
        result = monte_carlo_price(
            params = params_vol,
            n_sim=n_sim,
            seed=seed + i,
            antithetic=antithetic,
        )
        result["volatility"] = float(sigma)
        rows.append(result)

    df = pd.DataFrame(rows)
    cols = ["volatility"] + [col for col in df.columns if col != "volatility"]
    return df[cols]


# -----------------------------------------------------------------------------
# Greeks
# -----------------------------------------------------------------------------


def price_option(
    params,
    method="black_scholes",
    n_sim=100_000,
    seed=42,
    antithetic=False,
):
    """
    Small wrapper to price an option either with Black-Scholes or Monte Carlo.
    """

    if method == "black_scholes":
        return black_scholes_price(params)

    if method == "monte_carlo":
        result = monte_carlo_price(
            params = params,
            n_sim=n_sim,
            seed=seed,
            antithetic=antithetic,
        )
        return result["mc_price"]

    raise ValueError("method must be 'black_scholes' or 'monte_carlo'.")


def finite_difference_greeks(
    params,
    method="black_scholes",
    n_sim=100_000,
    seed=42,
    antithetic=True,
):
    """
    Estimate option Greeks using finite differences.

    The same seed is reused in bumped Monte Carlo prices. This helps reduce
    simulation noise when comparing slightly different scenarios.
    """

    S0 = params['S0']
    r = params['r']
    sigma = params['sigma']
    T = params['T']
    option_type = params['option type']

    dS = 0.01 * S0
    d_sigma = 0.01
    dr = 0.0001
    dT = 1 / 365

    if T <= dT:
        raise ValueError("T must be larger than one day for the theta calculation.")


    # Function to create bumped parameter sets for finite differences
    def bumped_params(**updates):
        new_params = params.copy()
        new_params.update(updates)
        return new_params

    price_base = price_option(
        params=params,
        method=method,
        n_sim=n_sim,
        seed=seed,
        antithetic=antithetic,
    )

    price_S_up = price_option(
        params=bumped_params(S0=S0 + dS),
        method=method,
        n_sim=n_sim,
        seed=seed,
        antithetic=antithetic,
    )

    price_S_down = price_option(
        params=bumped_params(S0=S0 - dS),
        method=method,
        n_sim=n_sim,
        seed=seed,
        antithetic=antithetic,
    )

    delta = (price_S_up - price_S_down) / (2 * dS)
    gamma = (price_S_up - 2 * price_base + price_S_down) / (dS**2)

    price_vol_up = price_option(
        params=bumped_params(sigma=sigma + d_sigma),
        method=method,
        n_sim=n_sim,
        seed=seed,
        antithetic=antithetic,
    )

    price_vol_down = price_option(
        params=bumped_params(sigma=sigma - d_sigma),
        method=method,
        n_sim=n_sim,
        seed=seed,
        antithetic=antithetic,
    )

    vega = (price_vol_up - price_vol_down) / (2 * d_sigma)

    price_r_up = price_option(
        params=bumped_params(r=r + dr),
        method=method,
        n_sim=n_sim,
        seed=seed,
        antithetic=antithetic,
    )

    price_r_down = price_option(
        params=bumped_params(r=r - dr),
        method=method,
        n_sim=n_sim,
        seed=seed,
        antithetic=antithetic,
    )

    rho = (price_r_up - price_r_down) / (2 * dr)

    price_T_up = price_option(
        params=bumped_params(T=T + dT),
        method=method,
        n_sim=n_sim,
        seed=seed,
        antithetic=antithetic,
    )

    price_T_down = price_option(
        params=bumped_params(T=T - dT),
        method=method,
        n_sim=n_sim,
        seed=seed,
        antithetic=antithetic,
    )

    # Market convention: theta measures the change in option value as time passes.
    theta = -(price_T_up - price_T_down) / (2 * dT)

    return {
        "method": method,
        "option_type": option_type,
        "price": float(price_base),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
        "vega_per_1pct_vol": float(vega * 0.01),
        "theta_per_day": float(theta / 365),
        "rho_per_1pct_rate": float(rho * 0.01),
    }


def black_scholes_greeks(params):
    """
    Closed-form Black-Scholes Greeks.

    These are useful as a benchmark for the finite-difference estimates.
    """

    S0 = params['S0']
    K = params['K']
    r = params['r']
    sigma = params['sigma']
    T = params['T']
    option_type = params['option type']

    if T <= 0:
        raise ValueError("T must be positive.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (S0 * sigma * sqrt(T))
    vega = S0 * pdf_d1 * sqrt(T)

    if option_type == "call":
        price = black_scholes_price(params)
        delta = norm.cdf(d1)
        theta = -(S0 * pdf_d1 * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2)
        rho = K * T * exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = black_scholes_price(params)
        delta = norm.cdf(d1) - 1
        theta = -(S0 * pdf_d1 * sigma) / (2 * sqrt(T)) + r * K * exp(-r * T) * norm.cdf(-d2)
        rho = -K * T * exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    return {
        "method": "closed_form",
        "option_type": option_type,
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
        "vega_per_1pct_vol": float(vega * 0.01),
        "theta_per_day": float(theta / 365),
        "rho_per_1pct_rate": float(rho * 0.01),
    }


def greeks_by_volatility(
    params,
    vol_grid,
    method="black_scholes",
    n_sim=100_000,
    seed=42,
    antithetic=True,
):
    """
    Estimate Greeks across a range of volatility values.
    """
    rows = []

    for i, sigma in enumerate(vol_grid):
        params_vol = params.copy()
        params_vol['sigma'] = float(sigma)
        greeks = finite_difference_greeks(
            params = params_vol,
            method=method,
            n_sim=n_sim,
            seed=seed + i,
            antithetic=antithetic,
        )
        greeks["volatility"] = float(sigma)
        rows.append(greeks)

    df = pd.DataFrame(rows)
    cols = ["volatility"] + [col for col in df.columns if col != "volatility"]
    return df[cols]


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def plot_sample_paths(paths, max_paths, OUTPUT_DIR):
    """Plot a sample of simulated stock price paths."""
    plt.figure(figsize=(9, 5))

    n_paths = min(max_paths, paths.shape[0])

    for i in range(n_paths):
        plt.plot(paths[i], linewidth=0.8)

    plt.title("Simulated GBM paths")
    plt.xlabel("Time step")
    plt.ylabel("Stock price")
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "gbm_paths.png", dpi=150)
    plt.show()


def plot_convergence(df, OUTPUT_DIR):
    """Plot Monte Carlo convergence against the Black-Scholes benchmark."""
    plt.figure(figsize=(9, 5))
    plt.plot(df["n_sim"], df["mc_price"], marker="o", label="Monte Carlo")
    plt.axhline(df["bs_price"].iloc[0], linestyle="--", label="Black-Scholes")
    plt.fill_between(df["n_sim"], df["ci_95_low"], df["ci_95_high"], alpha=0.2)
    plt.xscale("log")
    plt.title("Monte Carlo convergence")
    plt.xlabel("Number of simulations")
    plt.ylabel("Option price")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "mc_convergence.png", dpi=150)
    plt.show()


def plot_volatility_analysis(df, OUTPUT_DIR):
    """Plot option prices as a function of volatility."""
    plt.figure(figsize=(9, 5))
    plt.plot(df["volatility"], df["bs_price"], marker="o", label="Black-Scholes")
    plt.plot(df["volatility"], df["mc_price"], marker="x", label="Monte Carlo")
    plt.title("Volatility impact on option price")
    plt.xlabel("Volatility")
    plt.ylabel("Option price")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "volatility_analysis.png", dpi=150)
    plt.show()


def plot_greek_by_volatility(df, greek, OUTPUT_DIR):
    """Plot one Greek as a function of volatility."""
    if greek not in df.columns:
        raise ValueError(f"'{greek}' is not a column in the DataFrame.")

    plt.figure(figsize=(9, 5))
    plt.plot(df["volatility"], df[greek], marker="o")
    plt.title(f"{greek.capitalize()} vs volatility")
    plt.xlabel("Volatility")
    plt.ylabel(greek)
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / f"{greek}_by_volatility.png", dpi=150)
    plt.show()
