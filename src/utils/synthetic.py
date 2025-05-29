import numpy as np
import pandas as pd
from typing import List, Optional

def generate_gaussian_random_walk(length: int, mean: float, std: float, seed: int | None = None) -> pd.Series:
    """
    Generates a Gaussian random walk time series of specified length.
    
    Parameters:
    - length (int): The number of points in the time series.
    - std (float): The standard deviation of the increments.
    - seed (int, optional): Seed for reproducibility.
    
    Returns:
    - pd.Series: A time series where each increment is normally distributed with mean 0 and given std.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate i.i.d. normal increments
    increments = np.random.normal(loc=0, scale=std, size=length - 1)
    
    # Integrate to obtain the random walk
    x = np.concatenate([[mean], mean + np.cumsum(increments)])
    
    return pd.Series(x)

def synthetic_data(length: int, n_tickers: int, mean: List[float], std: List[float]):
    assert len(mean) == n_tickers, f"number of means ({len(mean)}) must equal number of tickers ({n_tickers})"
    assert len(std) == n_tickers, f"number of deviations ({len(std)}) must equal number of tickers ({n_tickers})"

    data = {}
    for i, avg, deviation in zip(range(n_tickers), mean, std):
        data[f'ticker_{i}'] = generate_gaussian_random_walk(length, avg, deviation)
    
    return data