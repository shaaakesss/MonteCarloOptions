# Imports
import sys
import numpy as np
import yfinance as yf
import time
import pandas as pd
import matplotlib.pyplot as plt 

# Data Fetching and Volatility Calculation
def get_stock_data(ticker):
    # Fetches current price & calculates historical volatility
    try:
        stock = yf.Ticker(ticker)
        
        # Gets today's closing price, goes to the 'Close' column and then selects the last cell
        current_price = stock.history(period="1d")['Close'].iloc[-1]

        # Get historical data for volatility calculation 
        hist = stock.history(period="1y") 
        if hist.empty or len(hist) < 5:
            return current_price, 0.20 # If data is insufficent, automatically provides 0.20 as a value for volatility, so script doesnt break

        # Calculate daily log returns; log returns are necessary for Geometric Brownian Motion model 
        # when calculating first day, there is no 'yesterday', so dropna gets rid of that result.
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()

        """
        prices = hist['Close'].values
        log_returns = []
        for i in range(1, len(prices)):
            ret = np.log(prices[i] / prices[i-1])
            log_returns.appen
        """ # Old method

        # Calculate annualized volatility; takes standard deviation of daily log returns and sqrt of time (formula)
        annual_volatility = log_returns.std() * np.sqrt(252)
        
        return current_price, annual_volatility
        
    except Exception as e:
        print(f"\nError fetching data for {ticker}: {e}")
        return -1

# Monte Carlo Simulation Function
def monte_carlo_option_pricer(S0, K, T, r, sigma, N_simulations=500000, N_steps=252):

    dt = T / N_steps # Size of one step of time

    if sigma <= 0: # Implies zero volatitly, i dont think this is possible but edge case nonetheless
        return np.maximum(S0 - K * np.exp(-r * T), 0), None

    # Drift and Diffusion; Change in stock price is a sum of these 2
    drift = (r - 0.5 * sigma**2) * dt # Average rate of change in price for risk-neutral pricing, predictable movement
    diffusion = sigma * np.sqrt(dt) # Randomness caused by market unpredictability

    # Generates Random Numbers
    Z = np.random.standard_normal((N_steps, N_simulations))
    R = drift + diffusion * Z # This is the daily log return
    
    # Takes log of starting price
    log_S0 = np.log(S0)
    
    # Calculate log prices using cumulative sum
    log_prices = log_S0 + np.cumsum(R, axis=0)
    
    # Convert log_prices back to actual dollar prices
    S_paths = np.exp(log_prices)
    
    # Calculate the Option Price 
    final_prices = S_paths[-1] # Gets price on last day
    
    payoffs = np.maximum(final_prices - K, 0) 
    # Calculates potential profit and compares with zero (since you cannot lose money on payoff),then picks the larger value
    
    option_price = np.exp(-r * T) * np.mean(payoffs) 
    # Multiplies expected future value and discount factor; # discount factor converts value of dollar to "current day value"

    return option_price, S_paths

# Graphing
def plot_paths(S_paths, K, ticker, N_plot=100): # Plots a subset of the simulated stock price paths
    
    # Time 
    N_steps = S_paths.shape[0]
    time_steps = np.arange(0, N_steps + 1)
    
    # Gets the starting price 
    S0 = S_paths[0, 0] 
    
    # Create a matrix with S0 as the starting point for all paths
    S0_matrix = np.full((1, S_paths.shape[1]), S0)
    full_paths = np.vstack([S0_matrix, S_paths])

    plt.figure(figsize=(10, 6))
    
    # Plot only (N_plot) no. of graphs to ensure the chart is readable lol, do not plot 500000 lines on a graph
    for i in range(min(N_plot, full_paths.shape[1])):
        plt.plot(time_steps, full_paths[:, i], alpha=0.5, linewidth=0.5)
    
    # Adding starting price & strike price + styling
    plt.axhline(S0, color='blue', linestyle='--', linewidth=2, label=f'Current Price (S0: ${S0:.2f})')
    plt.axhline(K, color='red', linestyle='-', linewidth=2, label=f'Strike Price (K: ${K:.2f})')
    
    plt.title(f'Monte Carlo Simulation: {ticker} Stock Price Paths')
    plt.xlabel('Trading Days')
    plt.ylabel('Stock Price ($)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.show()
    


# Main
if __name__ == "__main__":
    
    # Risk free rate
    ANNUAL_RISK_FREE_RATE = 0.04 
    
    # User Input
    print("Monte Carlo Options Pricer (US Stocks ONLY)")
    ticker = input("Enter the US Stock Ticker (e.g., AAPL): ").strip().upper()
    try:
        K = float(input("Enter the Option Strike Price (K): "))
        days = int(input("Enter Time to Maturity in Days (e.g., 90): "))
        T = days / 365.0 # T in years
    except ValueError:
        print("\nInvalid number input.")
        sys.exit(1)
    

    # Data Retrieval; uses the get_stock_data func from above
    S0, sigma = get_stock_data(ticker)

    # Define MC parameters
    N_simulations = 500000 
    N_steps = 252 # No. of trading days in a year :)

    # Run Simulation
    print(f"\nRunning Monte Carlo Simulation ({N_simulations} paths)...")
    start_time = time.time()
    
    price_mc, S_paths = monte_carlo_option_pricer(S0, K, T, ANNUAL_RISK_FREE_RATE, sigma, N_simulations, N_steps)
    
    end_time = time.time()
    
    # Analysis for Buy/Not Buy Suggestion
    intrinsic_value = np.maximum(S0 - K, 0) # Cash value if exercised today
    time_value = price_mc - intrinsic_value # Calculate premium (diff between model price & intrinsic value)
    
    # Output Results 
    print("\nSIM COMPLETE ")
    print(f"Ticker: {ticker}")
    print(f"Stock Price (S0): {S0:.2f}")
    print(f"Strike Price (K): {K:.2f}")
    print(f"Time to Expiry (T): {days} days ({T:.2f} years)")
    print(f"Risk-Free Rate (r): {ANNUAL_RISK_FREE_RATE*100:.2f}%")
    print(f"Volatility (sigma): {sigma:.4f}")
    print(f"Paths Simulated: {N_simulations}")
    print(f"Calculated Price (by MonteCarlo): {price_mc:.4f}")
    print(f"Computation Time: {end_time - start_time:.4f} seconds")
    
    print("\nANALYSIS")
    print(f"Intrinsic Value: {intrinsic_value:.4f}")
    print(f"Time Value: {time_value:.4f}")
    
    # Final Buy/Not Buy Suggestion
    if S0 > K:
        suggestion = f"The option is In-The-Money by ${intrinsic_value:.2f}. \nYour calculated fair price is ${price_mc:.4f}. \nYou are paying a time premium (Time Value) of ${time_value:.4f} for the chance of further gains. \nThis suggests the option is theoretically worth at least ${intrinsic_value:.4f}."
    else:
        suggestion = f"The option is Out-of-The-Money (OTM). \nIt has zero intrinsic value and is composed entirely of Time Value (${time_value:.4f}). \nYou should buy only if you are confident the stock will rise significantly above the strike price (${K:.2f}) before expiry."

    print("\n")
    print(suggestion)
    print("\n")

    # Visualization
    if S_paths is not None:
        plot_paths(S_paths, K, ticker)
