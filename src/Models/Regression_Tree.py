from numpy import ndarray
from sklearn.base import BaseEstimator, RegressorMixin


class Node:

    def __init__(self, leaf_node: bool = True, splitter: float = None) -> None:
        pass

    def predict(self, X: ndarray):
        pass



class Decision_Tree:

    def __init__(self) -> None:
        pass

    def predict(self, X: ndarray):
        pass



class Regression_Tree(BaseEstimator, RegressorMixin):

    def __init__(
            self,
            tree: Decision_Tree = None,
            minimum_elements: int = 20     
        ) -> None:
        super().__init__()

    
    def fit(self, X: ndarray, y: ndarray):
        pass


    def predict(self, X: ndarray):
        pass