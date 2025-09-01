from statistics import mean, stdev
from random import shuffle
import matplotlib.pyplot as plt


class Hyperparameters:
    def __init__(self, learning_rate=0.001, batch_size=50, number_epochs=20, eps=1e-6):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be higher than 0")
        elif batch_size <= 0:
            raise ValueError("Batch size must be higher than 0")
        elif number_epochs <= 0:
            raise ValueError("Number of epochs must be higher than 0")
        elif eps <= 0:
            raise ValueError("Epsilon (tolerance for loss difference) must be positive")

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.number_epochs = number_epochs
        self.eps = eps


class Model:
    class Parameters:
        def __init__(self):
            self.bias = 0.0
            self.weight = 0.0

    def _normalize_features_zscore(self):
        self.features_mean = mean(self.features)
        self.features_stdev = stdev(self.features)

        if self.features_stdev == 0:
            print("Standard deviation of features is zero, normalization impossible")
            return

        self.features = [(feature - self.features_mean) / self.features_stdev for feature in self.features]

    def __init__(self, labels: list[float], features: list[float], hyperparameters: Hyperparameters):
        if len(labels) != len(features):
            raise ValueError("Label and feature sizes are different")
        elif len(labels) == 0:
            raise ValueError("Label/feature sizes must be higher than 0")
        elif hyperparameters.batch_size > len(labels):
            raise ValueError("Batch size can't be higher than label/feature sizes")

        self.labels = labels
        self.features = features
        self._normalize_features_zscore()
        self.hyperparameters = hyperparameters
        self.parameters = Model.Parameters()

    def _mse_loss(self, predicted_labels: list[float], true_labels: list[float]) -> float: 
        if len(predicted_labels) != len(true_labels):
            raise ValueError("Lengths of true labels and predicted labels are different")

        N = len(true_labels) # nwm czy tutaj nie powinien byc inny len
        L2 = sum([(true_labels[i] - predicted_labels[i]) ** 2 for i in range(N)])

        return 1/N * L2

    def _update_parameters(self, features_slice: list[float], predicted_labels: list[float], true_labels: list[float]):
        # Supports only MSE loss methods

        if not (len(predicted_labels) == len(true_labels) == len(features_slice)):
            raise ValueError("Lengths of true labels and predicted labels are different")

        weight_derivative = 0.0
        bias_derivative = 0.0

        for i in range(len(predicted_labels)):
            prediction_diff = predicted_labels[i] - true_labels[i]
            weight_derivative += features_slice[i] * prediction_diff
            bias_derivative += prediction_diff

        derivative_constants = 2 / len(predicted_labels) 
        weight_derivative *= derivative_constants
        bias_derivative *= derivative_constants

        self.parameters.weight -= self.hyperparameters.learning_rate * weight_derivative
        self.parameters.bias -= self.hyperparameters.learning_rate * bias_derivative

    def train(self, plot_losses: bool = False):
        losses = []

        for _ in range(self.hyperparameters.number_epochs):
            temp = list(zip(self.labels, self.features))
            shuffle(temp)
            self.labels, self.features = zip(*temp)
            self.labels, self.features = list(self.labels), list(self.features)

            for i in range(0, len(self.features), self.hyperparameters.batch_size):
                batch = self.features[i : i + self.hyperparameters.batch_size]
                predicted_labels = [self.parameters.weight * feature + self.parameters.bias for feature in batch]
                true_labels = self.labels[i : i + self.hyperparameters.batch_size]
                features_slice = self.features[i : i + self.hyperparameters.batch_size]
                curr_loss = self._mse_loss(predicted_labels, true_labels)
                self._update_parameters(features_slice, predicted_labels, true_labels)

            print(f"Epoch {_ + 1}, loss={curr_loss}")

            if len(losses) and abs(curr_loss - losses[-1]) < self.hyperparameters.eps:
                break

            losses.append(curr_loss)
        print('')

        if plot_losses:
            fig, ax = plt.subplots()
            ax.plot([i for i in range(len(losses))], losses)
            ax.set_title("Training loss")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            plt.show()

    def predict(self, feature: float) -> float:
        if self.features_stdev == 0:
            print("Standard deviation of features is zero, normalization impossible")
            return self.parameters.weight * feature + self.parameters.bias

        normalized_feature = (feature - self.features_mean) / self.features_stdev
        return self.parameters.weight * normalized_feature + self.parameters.bias
