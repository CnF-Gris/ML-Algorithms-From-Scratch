from abc import ABC, abstractmethod
import random
from numpy import ndarray
import numpy
from sklearn.base import BaseEstimator, ClassifierMixin

class Layer(ABC):

    @abstractmethod
    def forward(X: ndarray) -> ndarray:
        pass

    @abstractmethod
    def backward(X: ndarray) -> ndarray:
        pass

    @abstractmethod
    def predict(self, X: ndarray) -> ndarray:
        pass

    def fit(self, X: ndarray, delta: float, learning_rate: float):
        return 0



class LinearLayer(Layer):

    def __init__(
            self,
            size: list[int],
            weights: ndarray = None,
            seed: int = random.randint(0, int(1E6))
        ) -> None:
        self.size = size
        self.weights = weights
        self.seed = seed

        if (self.weights is None) or (self.weights.size is not self.size):

            # Add a column for Bias
            self.size[0] += 1

            gen = numpy.random.default_rng(self.seed)
            self.weights = gen.random(self.size) * 1E-3

    def forward(self, X: ndarray):

        # Add Bias to input
        X = numpy.insert(X, 0, 1, axis=1)

        return X @ self.weights
    

    def backward(self, y: ndarray) -> ndarray:
        return self.weights[1:, :]
    

    def fit(self, X: ndarray, delta: float, learning_rate: float):
        X = numpy.insert(X, 0, 1, axis=1)
        gradient = X.T @ delta
        self.weights = self.weights - learning_rate * gradient

        return gradient
    
    def predict(self, X: ndarray) -> ndarray:
        return self.forward(X)



class ReLU(Layer):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X: ndarray) -> ndarray:

        X[X < 0] = 0
        return X
    
    def backward(self, X: ndarray) -> ndarray:
        X[X < 0] = 0
        X[X == 0] = 0.5
        X[X > 0] = 1
        return X
    
    def predict(self, X: ndarray) -> ndarray:
        return self.forward(X)



class Sigmoid(Layer):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X: ndarray) -> ndarray:
        return 1/(1 + numpy.exp(-X))
        
    
    def backward(self, X: ndarray) -> ndarray:
        return (1 - self.forward(X)) * self.forward(X)
    
    def predict(self, X: ndarray) -> ndarray:
        tmp = self.forward(X)
        tmp[tmp < 0.5] = 0
        tmp[tmp >= 0.5] = 1

        return tmp

        

class Neural_Network(BaseEstimator, ClassifierMixin):

    def __init__(
            self,
            layers: list[Layer],
            epochs: int = int(1E4),
            learning_rate: float = 0.05,
            debug_rate: int = 100,
            debug: bool = True
        ) -> None:
        super().__init__()
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.debug_rate = debug_rate
        self.debug = debug


    def fit(self, X: ndarray, y: ndarray):

        for current_epoch in range(0, self.epochs):
            intermediate_results = self.__forward(X)
            gradients = self.__backward(intermediate_results, y)

            if (current_epoch % self.debug_rate == 0) and self.debug:
                debug_string = (
                    "################################################################\n"
                    f"Neural Network - [DEBUG] - Epoch: {current_epoch + 1}\n\n"
                )

                for gradient, index in zip(gradients, range(0, len(self.layers))):
                    debug_string += f"Gradient {index + 1}:\n{gradient}\n"

                debug_string += "################################################################\n"
                print(debug_string)

        return self

    
    def __forward(self, X: ndarray):

        intermediate_results = [X]

        for layer in self.layers:
            X = layer.forward(X)
            intermediate_results.append(X)

        return intermediate_results
    

    def __backward(self, intermediate_results: list[ndarray], y: ndarray):

        delta = intermediate_results[-1] - y.reshape([-1, 1])
        gradients = []

        for index in range(len(self.layers) - 1, -1, -1):

            gradients.append(
                self.layers[index].fit(intermediate_results[index], delta, self.learning_rate)
            )

            if isinstance(self.layers[index], LinearLayer):
                delta = delta @ self.layers[index].backward(intermediate_results[index]).T
            elif index != 0:
                delta = delta * self.layers[index].backward(intermediate_results[index])

        gradients.reverse()

        return gradients

    
    def predict(self, X: ndarray):

        for layer in self.layers:
            X = layer.predict(X)

        return X
    

"""
a1 = X
b1 = X @ W1.T
c1 = NL(b1) = a2
b2 = a2 @ W2.T
a3 = NL(b2) = y

Loss = 1/N * (H(X) - Y).T @ (H(X) - Y)

dL/dW2 = dL/db2 * db2/dW2 = 1/N * sum((1 - y')* y' - y)
"""