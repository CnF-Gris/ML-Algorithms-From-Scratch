import numpy
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import GridSearchCV

from src.Models.Neural_Network import Neural_Network, LinearLayer, ReLU, Sigmoid

def neural_debug():
    [breast_X, breast_y] = datasets.load_breast_cancer(return_X_y=True)

    breast_X = numpy.array(breast_X[:,1:])
    print(breast_X)
    #breast_X = numpy.insert(breast_X, 0, 1, axis=1)
    print(breast_X)
    print(breast_y)
    breast_y = numpy.array(breast_y)

    [number_X, feature_X] = breast_X.shape

    # model = Linear_Regression(int(1E6), 0.006)
    # model = model.fit(breast_X, breast_y)
    # 
    # predicted_diabetes_y = model.predict(breast_X)

    CV_model = Neural_Network(
        [
            LinearLayer([feature_X, 10]),
            ReLU(),
            LinearLayer([10,1]),
            Sigmoid()
        ],
        learning_rate=5E-6,
        debug_rate=int(1E4)
    )
    parameters = {
        "epochs": [int(1E5)],
        "learning_rate": [5E-6, 5E-10],
    }
    grid_search = GridSearchCV(CV_model, parameters, error_score="raise")
    grid_search.fit(breast_X, breast_y)

    print(grid_search.cv_results_)

    #CV_model.fit(breast_X, breast_y)
    #print(CV_model.predict(breast_X))