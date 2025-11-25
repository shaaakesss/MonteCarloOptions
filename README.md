# Quantitative Finance Project: Implementing and Validating the Geometric Brownian Motion Model for Option Pricing

## Project Summary and Motivation
This project implements a Monte Carlo simulation method, utilizing the Geometric Brownian Motion (GBM) model, to calculate the theoretical fair value of European-style Call Options. 
As a student with a deep interest in trading and financial investment, I undertook this project to bridge my knowledge of market mechanics with computational finance. The project demonstrates an ability to implement complex mathematical models from first principles using efficient programming techniques.

## Technical Deep Dive: Model Implementation
The core objective was to translate the continuous mathematical framework of the GBM stochastic process into a discrete, high-performance simulation.

### Model Parameters
The simulation makes use of industry-standard inputs:
* The number of simulated paths is set to 500,000 to reduce variance in the final price estimate.
* The time interval is set to 1/252, reflecting the number of annual trading days, which allows the simulation to accurately model daily price fluctuations.
* Annual volatility is derived from the historical standard deviation of the stock's daily log returns over the past 252 trading days, calculated using data fetched via the yfinance library.

### Simulation Methodology
The process involves generating 500,000 independent future price paths. The final option price is determined by:
1.  Calculating the payoff for each path at maturity.
2.  Averaging these payoffs.
3.  Discounting the average back to the present value using the risk-free rate ($r$), consistent with the principle of risk-neutral valuation. 

## Development and Learning Process
The project was highly educational in several areas:

1. Computational Challenges
A primary difficulty definitely came in optimizing the code. I had to learn and implement NumPy library to execute the 500,000-path simulation using vectorized operations. This was absolutely necessary as Python's native methods were simply too slow for larger scaled financial modeling.

2. Theoretical Translation
Since I had no formal education in these topics, a significant chunk of time was spent in both understanding and accurately translating complex concepts into code. Segmented code which was both effective and conscice was incredibly difficult for me, it seemed that for every solution i came up with, there were 10 better, and it took me a long time to understand and implement these better solutions as well. As expected, this was where I faced the greatest difficulties.

3. Tool Utilization and Transparency
This project was aided by AI as a coding tool. Utilising AI as an helper allowed me to focus effort on the mathematical, code-related and financial aspects of the project, while leveraging AI for more trivial tasks like matplotlib visualisation styling and debugging.

## Future Development
I think a good way to build off of this project would be to implement the Black-Scholes formula to serve as a direct, comparative analysis between the two valuation solutions.  Furthermore I could even improve upon the current model by introducing and using Implied Volatility instead of historical volatility and even implement stochastic volatility. Though these concepts would require much more reseach and hardwork.

Overall, I really enjoyed the learning process and am incredibly interested in continuing down this rabbit-hole of merging computer science and finance.

