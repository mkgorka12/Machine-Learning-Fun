import numpy as np


def mean_squared_error(predicted_labels: np.ndarray, 
                       true_labels: np.ndarray) -> float:
    if predicted_labels.shape[0] != true_labels.shape[0]:
        raise ValueError("Labels size and predicted labels size must be the same")

    return np.mean(np.square(predicted_labels - true_labels))
