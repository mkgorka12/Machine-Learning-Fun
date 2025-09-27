from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

import util.validation
import util.metrics
import util.normalization


class LinearModel(ABC):
    def __init__(self, 
                 normalization_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                 loss_func: Callable[[np.ndarray, np.ndarray], float]):
        self.fitted = False
        self.normalization_func = normalization_func
        self.loss_func = loss_func

    def _validate(self, features: np.ndarray, labels: np.ndarray, 
                        learning_rate: float=0.01, batch_size: int=None, 
                        number_epochs: int=None, eps: float=1e-6) -> None:
        util.validation.validate_data(features, labels)
        util.validation.validate_hyperparameters(learning_rate, batch_size, number_epochs, eps)

    def _prepare_feature_stats(self, features: np.ndarray) -> None:
        self.feature_means = np.mean(features, axis=0)

        self.feature_stdevs = np.std(features, axis=0, ddof=1)
        self.feature_stdevs = np.where(self.feature_stdevs == 0, 1, self.feature_stdevs)

    def _prepare_data(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = self.normalization_func(features, self.feature_means, self.feature_stdevs)
        self.labels = labels

    def _prepare_hyperparameters(self, learning_rate:float=0.01, batch_size: int=None, 
                                 number_epochs: int=None, eps: float=1e-6) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size if batch_size is not None else self.features.shape[0]
        self.number_epochs = number_epochs
        self.eps = eps

    def _prepare_parameters(self) -> None:
        self.bias = 0.0
        self.weights = np.zeros((self.features.shape[1], 1))

    def _prepare_training_variables(self) -> None:
        self.fitted = False
        self.epoch_losses = []
        self.epoch_idx = 0

    def _prepare_to_fit(self, features: np.ndarray, labels: np.ndarray, 
                        learning_rate: float=0.01, batch_size: int=None, 
                        number_epochs: int=None, eps: float=1e-6) -> None:
        self._validate(features, labels, learning_rate, batch_size, number_epochs, eps)
        
        self._prepare_feature_stats(features)
        self._prepare_data(features, labels)
        self._prepare_hyperparameters(learning_rate, batch_size, number_epochs, eps)
        self._prepare_parameters()
        self._prepare_training_variables()

    def _should_continue_approximation(self) -> bool:
        return ((self.number_epochs is not None and self.epoch_idx < self.number_epochs)
            or (not self.fitted))
    
    def _shuffle_dataset(self) -> None:
        indices = np.arange(len(self.labels))
        np.random.shuffle(indices)

        self.labels = self.labels[indices]
        self.features = self.features[indices]

    def _is_avg_loss_within_tolerance(self, avg_losses: float) -> bool:
        return (len(self.epoch_losses) 
                and abs(avg_losses - self.epoch_losses[-1]) < self.eps)

    def _train_one_epoch(self) -> None:
        self._shuffle_dataset()

        batch_losses = []
        number_features = self.features.shape[0]

        for batch_idx in range(0, number_features, self.batch_size):
            features_batch = self.features[batch_idx : batch_idx + self.batch_size, :]

            predicted_labels = self._predict(features_batch)
            true_labels = self.labels[batch_idx : batch_idx + self.batch_size, :]

            curr_loss = self.loss_func(predicted_labels, true_labels)
            batch_losses.append(curr_loss)

            self._update_parameters(features_batch, predicted_labels, true_labels) 

        batch_losses_avg = np.mean(batch_losses)
            
        if self._is_avg_loss_within_tolerance(batch_losses_avg):
            self.fitted = True
            return
            
        self.epoch_losses.append(batch_losses_avg)
        self.epoch_idx += 1

    def fit(self, features: np.ndarray, labels: np.ndarray, learning_rate: float=0.01, 
            batch_size: int=None, number_epochs: int=None, eps: float=1e-6) -> None:
        self._prepare_to_fit(features, labels, learning_rate, batch_size, number_epochs, eps)

        while self._should_continue_approximation():
            self._train_one_epoch()

        self.fitted = True
        return self

    @abstractmethod
    def _update_parameters(self, features:np.ndarray, 
            predicted_labels:np.ndarray, true_labels:np.ndarray) -> None:
        pass

    @abstractmethod
    def _predict(self, features: np.ndarray) -> np.ndarray:
        pass

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        elif features.shape[1] != self.weights.shape[0]:
            raise ValueError("Feature dimensions are different from feature dimensions in the training set")

        features = self.normalization_func(features, self.feature_means, self.feature_stdevs)
        return self._predict(features)
    
    def plot_losses(self):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before plotting losses per epoch")
        
        fig, ax = plt.subplots()
        ax.plot(range(len(self.epoch_losses)), self.epoch_losses)
        ax.set_title("Training loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        plt.show()

class LinearRegression(LinearModel):
    def __init__(self, 
                 normalization_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = util.normalization.zscore,
                 loss_func: Callable[[np.ndarray, np.ndarray], float] = util.metrics.mean_squared_error):
        super().__init__(normalization_func, loss_func)

    def _update_parameters(self, features:np.ndarray, 
            predicted_labels:np.ndarray, true_labels:np.ndarray) -> None:
        error = predicted_labels - true_labels
        n = features.shape[0]

        bias_derivative = 2 * np.sum(error) / n
        weight_derivatives = 2 * (features.T @ error) / n

        self.bias -= bias_derivative * self.learning_rate
        self.weights -= weight_derivatives * self.learning_rate

    def _predict(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weights + self.bias
