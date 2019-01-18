import numpy as np
from math import sqrt



class SGD:
    def __init__(self, learning_rate, epochs, init_weights=[0.,0.]):
        "docstring"

        self.curr_weights  = np.array(init_weights)
        self.learning_rate = learning_rate
        self.epochs        = epochs

    def fit(self, Y, X):
        N          = float(len(Y))
        w1_current = self.curr_weights[0]
        w2_current = self.curr_weights[1]

        for i in range(self.epochs):
            # Y_pred = (m_current * X) + b_current
            Y_pred = X.dot(self.curr_weights)
            cost = sum([data**2 for data in (Y-Y_pred)]) / N
            rmse = sqrt(cost)

            w1_gradient = (2/N) * (X[0] * (Y_pred - Y))
            w2_gradient = (2/N) * (X[1] * (Y_pred - Y))

            w1_current = w1_current - (self.learning_rate * w1_gradient)
            w2_current = w2_current - (self.learning_rate * w2_gradient)

            self.curr_weights = np.array([w1_current, w2_current])
            print("RMSE: {} \t Weights: {}".format(rmse, self.curr_weights))


            return self.curr_weights, rmse



sgd = SGD(.01, 1 )

for i in range(0, 100):
    X = np.random.randint(10, size=2)
    Y = np.array([np.dot(X, np.array([11, 2]))])
    print(Y)
    

    sgd.fit(Y,X)
