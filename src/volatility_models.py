import numpy as np
from scipy.optimize import fmin_bfgs

class ARCH():

    name = 'ARCH'

    def __init__(self, p=1, theta=None, stationary=True, maxiter=500):

        self.p = p
        self.stationary = stationary
        self.maxiter = maxiter

        if theta is None:
            self.theta = np.ones((1, self.p+1))[0]
        else:
            # assert len(theta) == p
            self.theta = theta


        return None


    # def get_lags(self, y):
    #
    #     len_lag = len(y)-self.p
    #     self.lags = np.zeros((len_lag, self.p))
    #
    #     for i in range(self.p):
    #         self.lags[:,i] = y[i:-self.p+i]
    #
    #     return self.lags


    # def get_sigma(self, data):
    #     # sigma = omega + alpha*x(t)^2 + beta*sigma(t);
    #
    #     print('1:', np.ones((len(data),1)).shape)
    #     print('2:', np.power(data,2).shape)
    #     print('3:', data)
    #
    #     self.model_data = np.concatenate((np.ones((len(data),1)), np.power(data,2).reshape(len(data),1)), axis=1)
    #     self.sigma = np.sum(np.multiply(self.theta, self.model_data), axis=0)
    #
    #     return self.sigma


    def loglikelihood(self, X):
        # l = -(1 / 2) * log(2 * pi) - (1 / 2) * log(sig(1:T)) - (1 / 2) * (x').^2./sig(1:T);
        #
        # self.sigma = self.get_sigma(X)
        #
        # self.llik = -(1/2) * np.log(2 * np.pi) - (1/2) * np.log(self.sigma) - (1/2) * np.divide(np.power(X,2), self.sigma)

        return None # self.llik


    def fit(self, y):

        X = 0 #self.get_lags(y)
        # print(X.shape)

        print(self.theta)
        self.theta_hat = fmin_bfgs(f=self.loglikelihood,
                                  x0=self.theta,
                                  # maxiter=self.maxiter,
                                  args=(X,)
                                  )

        return self.theta_hat

