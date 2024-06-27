import numpy
from sklearn import datasets

from src.Models.Linear_Regression import Linear_Regression

def linearDebug():

    [diabetes_X, diabetes_y] = datasets.load_diabetes(return_X_y=True)

    diabetes_X = numpy.array(diabetes_X)
    diabetes_y = numpy.array(diabetes_y)

    model = Linear_Regression(int(1E4), 0.006)
    model = model.fit(diabetes_X, diabetes_y)