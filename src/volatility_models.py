import numpy as np
from scipy.optimize import minimize
import pandas as pd

class ARCH():

    name = 'ARCH'

    def __init__(self, p=1, theta=None, stationary=True, maxiter=500,
                 method='BFGS', bounds=None):

        self.p = p
        self.stationary = stationary
        self.maxiter = maxiter
        self.method = method
        self.bounds = bounds

        if theta is None:
            self.theta = np.ones((1, self.p+1)).flatten()
        else:
            # assert len(theta) == p
            self.theta = theta


        return None


    def get_lags(self, y):

        len_lag = len(y)-self.p
        lags = np.ones((len_lag, self.p))

        for i in range(self.p):
            if i != 0:
                lags[:,i] = y[i:-self.p+i]

        return lags


    def get_sigma(self, data):
        # sigma = omega + alpha*x(t)^2 + beta*sigma(t);

        print('1:', np.ones((len(data),1)).shape)
        print('2:', np.power(data,2).shape)
        print('3:', data)

        model_data = np.concatenate((np.ones((len(data),1)), np.power(data,2).reshape(len(data),1)), axis=1)
        sigma = np.sum(np.multiply(self.theta, model_data), axis=0)

        return sigma


    # def loglikelihood(self, X):
    #     # l = -(1 / 2) * log(2 * pi) - (1 / 2) * log(sig(1:T)) - (1 / 2) * (x').^2./sig(1:T);
    #
    #     # sigma = self.get_sigma(X)
    #     #
    #     # llik = -(1/2) * np.log(2 * np.pi) - (1/2) * np.log(sigma) - (1/2) * np.divide(np.power(X,2), sigma)
    #
    #
    #
    #     return llik

    def sigmoid(self, X, beta):
        z = np.dot(X, beta)
        return 1 / (1 + np.exp(-z))

    def loglikelihood(self, beta, X):

        y = 0.5
        m = len(X)
        h = self.sigmoid(X, beta)

        J = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

        llik = -(1 / 2) * np.log(2 * np.pi) - (1 / 2) * np.log(h) - (1 / 2) * np.divide(np.power(X, 2), h)

        return llik

    def optimizer(self, theta, X):

        opt = minimize(fun=self.loglikelihood,
                            x0=theta,
                            method=self.method,
                            args=(X,),
                            bounds=self.bounds,
                            options={'maxiter':self.maxiter}
                            )

        return opt

    def fit(self, y):

        X = self.get_lags(y)
        self.X = np.concatenate([np.ones((len(X),1)), X], axis=1)
        # print(X.shape)

        print(self.theta)
        self.opt = self.optimizer(self.theta, self.X)

        self.theta_hat = self.opt.x

        return self.theta_hat


df = pd.read_csv('../data/BTC-USD.csv')

close_diff = df['Close'].diff().values

model = ARCH()
model.fit(close_diff)

print(model.theta_hat)