import numpy as np


def zscore(features: np.ndarray, mean: np.ndarray, stdev: np.ndarray) -> np.ndarray:
    return (features - mean) / stdev
