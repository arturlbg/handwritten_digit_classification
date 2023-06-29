import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):

        result = np.dot(X, self.w)
        '''
        The first element of the vector w is the bias, that is, the value that is added to the result of the multiplication
        of weights by attributes. Therefore, it is not multiplied by any attribute, just added to the result
        '''
        return np.sign(result)
    
    def get_w(self):
        return self.w