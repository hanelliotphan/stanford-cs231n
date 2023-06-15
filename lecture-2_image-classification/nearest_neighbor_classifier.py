import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass
    
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        num_test = X.shape[0]
        y_pred = np.zeros(num_test, dtype=self.y_train.dtype)
        for i in range(num_test):
            distances = np.sum(np.abs(self.X_train - X[i,:]), axis=1)
            min_index = np.argmin(distances)
            y_pred[i] = self.y_train[min_index]
        return y_pred
