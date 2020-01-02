from src.volatility_models import ARCH
import numpy as np

class GARCH(ARCH):

    def __init__(self, theta=None, bounds=((0,1),(0,1),(0,1)), p=1, q=1):
        '''
        Inherits the init function of the base class.
        :param theta: Parameters that have to be optimized
        '''
        super().__init__(self)
        self.params = 3

        self.p = p
        self.q = q

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
        # print('test1', theta)
        # print('test2', self.theta)
        # print('test3', self.p)
        eps_mat = self.gen_eps_mat(eps)
        const = np.ones((len(eps_mat), 1))
        eps_mat_sum = np.sum(np.multiply(theta[1:self.p+1], eps_mat), axis=1)

        sigma = np.zeros((len(eps), 1))
        lags = max([self.p, self.q])
        sigma[:lags] = np.var(eps)

        #
        # print('eps', eps_mat_sum.shape)
        # print('sigma', sigma.shape)

        for i in range(lags, len(sigma)):
            sigma_lag = sigma[i-lags:i+lags]

            sigma_vec_sum = np.sum(np.multiply(theta[self.p+1:self.p+self.q+1], sigma_lag))

            # print(sigma[i].shape)
            # print(theta.shape)
            print(theta[0], eps_mat_sum[i-lags*2], sigma_vec_sum)
            # print(eps_mat_sum[i-lags*2])
            # print(sigma_vec_sum)

            sigma[i] = theta[0] + eps_mat_sum[i-lags*2] + sigma_vec_sum
            print('sigma', sigma)

        print(sigma[:5], sigma[-5:])
        return sigma

class ZD_GARCH(ARCH):

    def __init__(self, theta=None):
        '''
        Inherits the init function of the base class.
        :param theta: Parameters that have to be optimized
        '''
        super().__init__(self)
        self.params = 2
        if theta is None:
            self.theta = np.ones((self.params))
            self.theta[0] = self.theta[0] * 0.1
            self.theta[1] = self.theta[1] * 0.8
        else:
            self.theta = theta


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
            sigma[i] = theta[0] + theta[1]*sigma[i-1]

        return sigma

class EGARCH(ARCH):

    def __init__(self, theta=None, bounds=((0,1),(0,1),(0,1),(0,1))):
        '''
        Inherits the init function of the base class.
        :param theta: Parameters that have to be optimized
        '''
        super().__init__(self)
        self.params = 4
        if theta is None:
            self.theta = np.ones((self.params))
            self.theta[:2] = self.theta[:2] * 0.1
            self.theta[2] = self.theta[2] * 0.8
            self.theta[3] = self.theta[3] * 0.1
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
            sigma[i] = np.exp(theta[0] + theta[2]*np.log(sigma[i-1])
                              + theta[1] * np.divide(np.absolute(eps[i-1]), np.sqrt(sigma[i-1]))
                              - np.sqrt(2/np.pi) + theta[3] * np.divide(eps[i-1],np.sqrt(sigma[i-1]))
                              )

        return sigma

class SE_GARCH(ARCH):

    def __init__(self, theta=None, bounds=((0,1),(0,1),(0,1))):
        '''
        Inherits the init function of the base class.
        :param theta: Parameters that have to be optimized
        '''
        super().__init__(self)
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
            sigma[i] = theta[0] + theta[1] * np.power(eps[i-1], 2) *sigma[i-1] + theta[2]*sigma[i-1]

        return sigma

class NGARCH(ARCH):

    def __init__(self, theta=None, bounds=((0,1),(0,1),(0,1),(0,1))):
        '''
        Inherits the init function of the base class.
        :param theta: Parameters that have to be optimized
        '''
        super().__init__(self)
        self.params = 4

        if theta is None:
            self.theta = np.ones((self.params))
            self.theta[0] = self.theta[0] * 0.1
            self.theta[1] = self.theta[1] * 0.8
            self.theta[2] = self.theta[2] * 0.5
            self.theta[3] = self.theta[3] * 0.1
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
            eps_plus = eps[i-1] / np.sqrt(sigma[i-1])
            sigma[i] = theta[0] + theta[1] * sigma[i-1] + theta[2]*sigma[i-1] * np.power((eps_plus-theta[3]),2)

        return sigma


