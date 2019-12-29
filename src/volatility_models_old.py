import numpy as np
from scipy.optimize import minimize
import pandas as pd

class ARCH():

    name = 'ARCH'

    def __init__(self, p=1, theta=None, stationary=True, maxiter=500,
                 method='L-BFGS-B', bounds=((0,1),(0,1))):

        self.p = p
        self.stationary = stationary
        self.maxiter = maxiter
        self.method = method
        self.bounds = bounds


        if theta is None:
            self.theta = np.ones((1, 2)).flatten() * 0.1
        else:
            # assert len(theta) == p
            self.theta = theta


        return None


    # def get_lags(self, y):
    #
    #     len_lag = len(y)-self.p
    #     lags = np.ones((len_lag, self.p))
    #
    #     for i in range(self.p):
    #         if i != 0:
    #             lags[:,i] = y[i:-self.p+i]
    #
    #     return lags


    def get_sigma(self, theta, eps):
        # sigma = omega + alpha*x(t)^2 + beta*sigma(t);

        sigma = np.zeros((len(eps),1))

        for i in range(len(eps)):
            sigma[i] = theta[0] + theta[1]*np.power(eps[i],2)
            # print(i, sigma[i], theta)

        # print('1:', np.ones((len(data),1)).shape)
        # print('2:', np.power(data,2).shape)
        # print('3:', data)

        # model_data = np.concatenate((np.ones((len(data),1)), np.power(data,2).reshape(len(data),1)), axis=1)
        # sigma = np.sum(np.multiply(self.theta, model_data), axis=0)

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

    # def sigmoid(self, X, beta):
    #     z = np.dot(X, beta)
    #     return 1 / (1 + np.exp(-z))

    def loglikelihood(self, theta, eps):


        sigma = self.get_sigma(theta, eps)
        # print('sigma', sigma.shape)
        # print('eps', eps.shape)
        # J = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

        # a = -(1 / 2) * np.log(2 * np.pi)
        # b = - (1 / 2) * np.log(h)
        # c =  - (1 / 2) * np.divide(np.power(eps, 2), h)

        # l = -(1 / 2) * log(2 * pi) - (1 / 2) * log(sig(1:T)) - (1 / 2) * (x').^2./sig(1:T);
        # llik = -(1/2) * np.log(2*np.pi) - (1/2)*np.log(h) - (1/2)

        llik = -(1 / 2) * np.log(2 * np.pi) - (1 / 2) * np.log(sigma) - (1 / 2) * np.divide(np.power(eps, 2), sigma)

        return np.mean(llik)

    def optimizer(self, theta, eps):

        opt = minimize(fun=self.loglikelihood,
                            x0=theta,
                            method=self.method,
                            args=(eps,),
                            bounds=self.bounds,
                            options={'maxiter':self.maxiter},
                            )

        return opt

    def fit(self, y):

        # X = self.get_lags(y)
        # self.X = np.concatenate([np.ones((len(y),1)), y], axis=1)
        # print(X.shape)
        self.eps = y.reshape((len(y),1))
        print(self.theta)
        self.opt = self.optimizer(self.theta, self.eps)

        self.theta_hat = self.opt.x

        return self.theta_hat


import matplotlib.pyplot as plt

df = pd.read_csv('../data/BTC-USD.csv')

close_diff = np.log(df['Close']) - np.log(df['Close'].shift(1))
close_diff = close_diff.dropna().values

plt.plot(close_diff)


model = ARCH()
model.fit(close_diff)

print(model.theta_hat)

plt.show()