import numpy as np
from numpy import linalg as LA
from random import sample
class LogisticRegressionRegularized:

    def __init__(self, eta=0.1, tmax=1000, bs=1_000_000,lamb=0):
        self.eta = eta
        self.tmax = tmax
        self.batch_size = bs
        self.lamb = lamb
        self.w = None

    def fit(self, _X, _y):
        X = np.array(_X)
        y = np.array(_y)
        N = X.shape[0]
        d = X.shape[1]
        self.w = np.zeros(d)
        for t in range(self.tmax):
            if self.batch_size < N:
                indexes = sample(range(N), self.batch_size)
                X_batch = np.array([_X[i] for i in indexes])
                y_batch = np.array([_y[i] for i in indexes])
            else:
                X_batch = X
                y_batch = y
            gt = (-1/N) * np.dot(X_batch.T, y_batch/(1 + np.exp(y_batch * np.dot(X_batch, self.w))))


            #regularization
            if self.lamb !=0:
                gt += 2 * self.lamb * self.w


            if LA.norm( gt ) < 1e-3:
                break
            
            self.w = self.w - (self.eta * gt)

    def predict_prob(self, X):
        return 1/(1 + np.exp(-np.dot(X, self.w)))
    
    def predict(self, X):
        return np.sign(self.predict_prob(X) - 0.5)
    
    def get_w(self):
        return self.w