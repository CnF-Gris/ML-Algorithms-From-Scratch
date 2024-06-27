"""
For this class, let's implement it so that 
Scikit is able to perform a grid search automatically 
"""

import random
import numpy
from sklearn.base import BaseEstimator, RegressorMixin
from numpy import ndarray

class Linear_Regression(RegressorMixin, BaseEstimator):
    

    def __init__(
            self, 
            epochs : int = int(1E3),
            learning_rate : float = 0.05,
            debug: bool = True,
            weights : ndarray = None,
            seed: int = random.randint(0, 1E9)
        ) -> None:
        super.__init__()
        self.weights = weights  # Assuming that it is a row vector
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
                    f"[DEBUG] - Linear Regression - Epoch: {current_epoch}\n"
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

        return X @ self.weights.T
    
