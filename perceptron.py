import random
import numpy as np


class PLA:

    def __init__(self, maxIter = 30_000):
        self.maxIter = maxIter
        self.w = None

    def get_w(self):
        return self.w
    
    def set_w(self, w):
        self.w = w

    def misclassified(self, X, y):
        """
        This function will receive the dataset and return the indexes of incorrectly classified points

        Args:
            X (list): list of training dataset points
            y (list): list of the label of each point in the training dataset

        returns:
            idx_incorretos (array): indexes of incorrectly sorted points
        """

        # Calculate the inner product of all points with the weight vector
        results = np.dot(X, self.w)

        # Apply the activation function (signal) to the results
        results = np.sign(results)

        # Compare the results with the true labels and return the indices of the incorrectly classified points
        incorrect_idx = np.where(results != y)[0]
        
        return incorrect_idx




    def fit(self, X, y):
        """
        Execution of the PLA algorithm

        Args:
            X (list): list of training dataset points
            y (list): list of the label of each point in the training dataset

        returns:
            w (array): weight vector
        """
        # Initialize weight vector = 0
        self.w = np.zeros(X.shape[1]) # [wo,w1,w2]

        iterator = 0

        while iterator < self.maxIter:
            # Calculate results for all points and compare with true labels
            incorrect_idx = self.misclassified(X, y)

            if len(incorrect_idx) == 0:
                break

            # Choose an incorrectly classified point at random
            i = random.choice(incorrect_idx)

            # Update the weight vector
            self.w = self.w + y[i] * X[i]

            iterator += 1

        print("number of iterations ->", iterator)
        return self.w

    def predict(self, X):
        """
        Sort a point

        Args:
            X (list): list of test dataset points

        returns:
            y (list): label list of each test dataset point
        """

        # Calculate the inner product of all points with the weight vector
        result = np.dot(X, self.w)

        # Apply the activation function (signal) to the results
        result = np.sign(result)

        return result
