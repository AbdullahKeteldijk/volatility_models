import numpy as np
from scipy.optimize import minimize
import pandas as pd
from math import gamma

class GARCH():
    '''
    Base class for the Autoregressive Conditional Heteroskedasticy model.
    '''

    name = 'GARCH'

    def __init__(self, p=1, theta=None, maxiter=500, distribution='Normal',
                 method='L-BFGS-B', bounds=((0,1),(0,1),(0,1)), dof=3):
        '''

        :param p: int - Number of lag periods
        :param theta: list - Parameters that have to be optimized
        :param maxiter: Maximum number of iterations of the optimizers
        :param distribution: string - Name of probability distribution
        :param method: string - Name of Scipy minimize optimization algorithm
        :param bounds: tuple - Tuple of tuples with lower and upper bound of the theta parameters
        '''
        
        self.p = p
        self.maxiter = maxiter
        self.method = method
        self.bounds = bounds
        self.distribution = distribution
        self.params = 2
        self.dof = dof

        if theta is None:
            self.theta = np.ones((self.params)) * 0.1
        else:
            self.theta = theta


        self.params = 3

        if theta is None:
            self.theta = np.ones((self.params))
            self.theta[:2] = self.theta[:2] * 0.1
            self.theta[2] = self.theta[2] * 0.8
        else:
            self.theta = theta

        self.bounds = bounds

    def get_sigma(self, theta, eps):
        '''
        Base sigma function
        :param theta: Parameters that have to be optimized
        :param eps: input vector with I(0) data
        :return: sigma: estimated volatility
        '''
        sigma = np.zeros((len(eps), 1))
        sigma[0] = np.var(eps)

        for i in range(1, len(eps)):
            sigma[i] = theta[0] + theta[1] * np.power(eps[i-1], 2) + theta[2]*sigma[i-1]

        return sigma


    def loglikelihood(self, theta, eps):
        '''
        The loglikelihood is the cost function for the algorithm. We can implement multiple probability distributions.
        :param theta: Parameters that have to be optimized
        :param eps: input vector with I(0) data
        :return: llik: loglikelihood value
        '''

        sigma = self.get_sigma(theta, eps)

        if self.distribution == 'Normal':
            llik = -(1 / 2) * np.log(2 * np.pi) - (1 / 2) * np.log(sigma) - (1 / 2) * np.divide(np.power(eps, 2), sigma)

        elif self.distribution == 'Student-t':

            nu = self.dof

            llik = np.log(gamma(nu+1)/2) - np.log(gamma(nu/2)) \
                   - 0.5 * np.log((nu-2)*np.pi*sigma) \
                   - ((nu+1)/2)*np.log(1+(np.divide(np.power(eps,2),(nu-2)*sigma)))

        return np.mean(-llik)

    def optimizer(self, theta, eps):
        '''
        Minimize function from Scipy.
        :param theta: Parameters that have to be optimized
        :param eps: input vector with I(0) data
        :return: opt: Scipy optimize object
        '''
        opt = minimize(fun=self.loglikelihood,
                            x0=theta,
                            method=self.method,
                            args=(eps,),
                            bounds=self.bounds,
                            options={'maxiter':self.maxiter},
                            )
        return opt

    def fit(self, y):
        '''
        Fitting the model to the data.
        :param y: input vector with I(0) data
        :return: None
        '''
        self.n = len(y)
        self.m = len(self.theta)

        self.eps = y.reshape((len(y),1))
        self.opt = self.optimizer(self.theta, self.eps)

        print(self.opt)
        self.llik = self.opt.fun
        self.theta_hat = self.opt.x

        self.se = np.sqrt(np.diagonal(self.opt.hess_inv.todense()))
        self.z_values = np.divide(self.theta_hat, self.se)

        self.AIC = 2 * self.m - 2 * np.log(self.llik)
        self.BIC = np.log(self.n) * self.m - 2 * np.log(self.llik)

        return None

    def forecast(self, y):
        '''
        Forecast function
        :param y: input vector with I(0) data
        :return: prediction based on trained model
        '''
        return self.get_sigma(self.theta_hat, y)



