import numpy as np
from scipy.optimize import minimize
import pandas as pd

class ARCH():

    name = 'ARCH'

    def __init__(self, p=1, theta=None, stationary=True, maxiter=500, distribution='Normal',
                 method='L-BFGS-B', bounds=((0,1),(0,1))):

        self.p = p
        self.stationary = stationary
        self.maxiter = maxiter
        self.method = method
        self.bounds = bounds
        self.distribution = distribution
        self.params = 2

        if theta is None:
            self.theta = np.ones((self.params)) * 0.1
        else:
            self.theta = theta

        return None


    def get_sigma(self, theta, eps):

        sigma = np.zeros((len(eps), 1))
        sigma[0] = np.var(eps)

        for i in range(1, len(eps)):
            sigma[i] = theta[0] + theta[1] * np.power(eps[i-1], 2)

        return sigma


    def loglikelihood(self, theta, eps):

        sigma = self.get_sigma(theta, eps)

        if self.distribution == 'Normal':
            llik = -(1 / 2) * np.log(2 * np.pi) - (1 / 2) * np.log(sigma) - (1 / 2) * np.divide(np.power(eps, 2), sigma)
        # elif self.distribution == 'Student-t':
        #     llik = np.log(theta)

        return np.mean(-llik)

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

        self.n = len(y)
        self.m = len(self.theta)

        self.eps = y.reshape((len(y),1))
        self.opt = self.optimizer(self.theta, self.eps)

        print(self.opt)
        self.llik = self.opt.fun
        self.theta_hat = self.opt.x

        self.se = np.diagonal(self.opt.hess_inv.todense())
        self.z_values = np.divide(self.theta_hat, self.se)

        self.AIC = 2 * self.m - 2 * np.log(self.llik)
        self.BIC = np.log(self.n) * self.m - 2 * np.log(self.llik)

        return None

    def forecast(self, y):

        return self.get_sigma(self.theta_hat, y)



