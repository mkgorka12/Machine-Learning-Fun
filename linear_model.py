import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self):
        self.fitted = False

    def _zscore(self, features:np.ndarray) -> np.ndarray:
        return (features - self.feature_mean) / self.feature_stdev

    def _validate_data(self, features, labels):
        if features.size == 0:
            raise ValueError("Features are empty")
        elif features.ndim != 2:
            raise ValueError("Features are expected to be a matrix")
        elif labels.size == 0:
            raise ValueError("Labels are empty")
        elif labels.ndim != 2:
            raise ValueError("Labels are expected to be a matrix of size (n, 1)")
        elif features.shape[0] != labels.shape[0]:
            raise ValueError("Feature row sizes and label row sizes must be the same")
        
        self.feature_mean = np.mean(features, axis=0)
        self.feature_stdev = np.std(features, axis=0, ddof=1)
        self.feature_stdev = np.where(self.feature_stdev == 0, 1, self.feature_stdev)

        self.features = self._zscore(features)
        self.labels = labels

    def _validate_hyperparameters(self, learning_rate, batch_size, number_epochs, eps):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be bigger than 0")
        elif batch_size is not None and batch_size <= 0:
            raise ValueError("Batch size must be bigger than 0")
        elif batch_size is not None and batch_size > self.features.shape[0]:
            raise ValueError("Batch size can't be bigger than label size")
        elif number_epochs is not None and number_epochs <= 0:
            raise ValueError("Number of epochs must be bigger than 0")
        elif eps <= 0:
            raise ValueError("Epsilon (tolerance for loss difference) must be positive")

        self.learning_rate = learning_rate
        self.batch_size = batch_size if batch_size is not None else self.features.shape[0]
        self.number_epochs = number_epochs
        self.eps = eps

    def _shuffle_dataset(self):
        indices = np.arange(len(self.labels))
        np.random.shuffle(indices)

        self.labels = self.labels[indices]
        self.features = self.features[indices]

    @staticmethod
    def mean_squared_error(predicted_labels:np.ndarray, true_labels:np.ndarray) -> float: 
        if predicted_labels.shape[0] != true_labels.shape[0]:
            raise ValueError("Labels size and predicted labels size must be the same")

        return np.mean(np.square(predicted_labels - true_labels))

    def _update_parameters(self, features:np.ndarray, predicted_labels:np.ndarray, true_labels:np.ndarray):
        error = predicted_labels - true_labels
        n = features.shape[0]

        bias_derivative = 2 * np.sum(error) / n
        weight_derivatives = 2 * (features.T @ error) / n

        self.bias -= bias_derivative * self.learning_rate
        self.weights -= weight_derivatives * self.learning_rate

    def fit(self, features:np.ndarray, labels:np.ndarray, 
            learning_rate:float=0.01, batch_size:int=None, number_epochs:int=None, eps:float=1e-6):
        self._validate_data(features, labels)
        self._validate_hyperparameters(learning_rate, batch_size, number_epochs, eps)
        
        self.fitted = True
        self.epoch_losses = []
        self.bias = 0.0
        self.weights = np.zeros((self.features.shape[1], 1))

        epoch_idx = 0
        while self.number_epochs is None or epoch_idx < self.number_epochs:
            batch_losses = []
            self._shuffle_dataset()

            for batch_idx in range(0, self.features.shape[0], self.batch_size):
                features_batch = self.features[batch_idx : batch_idx + self.batch_size, :]
                predicted_labels = (features_batch @ self.weights + self.bias).reshape((features_batch.shape[0], 1))
                true_labels = self.labels[batch_idx : batch_idx + self.batch_size, :]

                curr_loss = self.mean_squared_error(predicted_labels, true_labels)
                batch_losses.append(curr_loss)

                self._update_parameters(features_batch, predicted_labels, true_labels) 

            batch_losses_avg = np.mean(batch_losses)
            
            if len(self.epoch_losses) and abs(batch_losses_avg - self.epoch_losses[-1]) < self.eps:
                break
            
            self.epoch_losses.append(batch_losses_avg)
            epoch_idx += 1

        return self

    def plot_losses(self):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before plotting losses per epoch")
        
        fig, ax = plt.subplots()
        ax.plot(range(len(self.epoch_losses)), self.epoch_losses)
        ax.set_title("Training loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        plt.show()

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        elif features.shape[1] != self.weights.shape[0]:
            raise ValueError("Feature dimensions are different from feature dimensions in the training set")

        features = self._zscore(features)
        return features @ self.weights + self.bias
