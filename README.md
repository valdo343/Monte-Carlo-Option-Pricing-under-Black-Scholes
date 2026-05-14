# Monte-Carlo-Option-Pricing-under-Black-Scholes
Implemented a European option pricing engine in Python using Black-Scholes and Monte Carlo simulation under risk-neutral geometric Brownian motion. Added convergence analysis, confidence intervals, antithetic variates, volatility sensitivity, and Greeks estimation via finite differences validated against closed-form formulas.

## Features

- Black-Scholes pricing for European calls and puts
- Risk-neutral geometric Brownian motion simulation
- Monte Carlo pricing with confidence intervals
- Convergence analysis
- Volatility sensitivity analysis
- Greeks via finite differences
- Closed-form Black-Scholes Greeks for validation

## Methods

The asset follows a risk-neutral geometric Brownian motion:

$$
dS_t = r S_t dt + \sigma S_t dW_t
$$

Terminal prices are simulated as:

$$
S_T = S_0 \exp((r - 0.5\sigma^2)T + \sigma \sqrt{T}Z)
$$

## Example Results

The notebook compares Monte Carlo prices against the Black-Scholes benchmark
and visualizes convergence as the number of simulations increases.

## Technologies

Python, NumPy, Pandas, SciPy, Matplotlib, Jupyter.
