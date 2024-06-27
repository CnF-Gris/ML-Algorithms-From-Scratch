import random
from numpy import ndarray
import numpy
from sklearn.base import BaseEstimator, ClassifierMixin


class Logistic_Regression(BaseEstimator, ClassifierMixin):

    def __init__(
            self,
            weights: ndarray = None,
            epochs: int = int(1E4),
            learning_rate: float = 0.05,
            debug: bool = True,
            seed: int = random.randint(0, 1E9)
        ) -> None:
        super().__init__()
        self.weights = weights
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.debug = debug
        self.seed = seed


    def fit(self, X: ndarray, y: ndarray):

        [number_of_samples, number_of_features] = X.shape

        if self.weights is None:
            randomGen = numpy.random.default_rng(self.seed)
            self.weights = randomGen.random(size=number_of_features)

        # compute each epoch in full batch
        for current_epoch in range(0, self.epochs):

            # compute the prediction
            predicted_y = self.predict(X)

            # compute the error
            errors : ndarray = X.T @ (predicted_y - y) 

            # compute the gradient
            gradient : ndarray = (1/number_of_samples) * errors

            # modify weights
            self.weights = self.weights - self.learning_rate * gradient

            if( current_epoch % 1000 == 0) and self.debug:

                debug_string = (
                    "######################################################\n"
                    f"[DEBUG] - Logistic Regression - Epoch: {current_epoch}\n"
                    "\n"
                    f"Epochs: {self.epochs}\n"
                    f"Learning_rate: {self.learning_rate}\n"
                    "\n"
                    f"Weights: \n {self.weights}\n"
                    f"Gradient: \n {gradient}\n"
                    f"Errors: \n {errors}\n"
                    "######################################################\n"
                )

                print(debug_string)

        return self
    

    def predict(self, X: ndarray) -> ndarray:

        return 1 / (1 + numpy.exp(- X @ self.weights.T))
    
