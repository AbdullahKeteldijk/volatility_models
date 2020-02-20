from src.volatility_models import GARCH
import numpy as np

class ARCH(GARCH):

    def __init__(self, theta=None, bounds=((0,1),(0,1))):
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
            sigma[i] = theta[0] + theta[1] * np.power(eps[i-1], 2) + theta[2]*sigma[i-1]

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

class EGARCH(GARCH):

    def __init__(self, theta=None, bounds=((0,1),(0,1),(0,1),(0,1)), distribution='Normal'):
        '''
        Inherits the init function of the base class.
        :param theta: Parameters that have to be optimized
        '''
        super().__init__(self)
        self.params = 4
        self.distribution = distribution
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

class SE_GARCH(GARCH):

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

class NGARCH(GARCH):

    def __init__(self, theta=None, bounds=((0,1),(0,1),(0,1),(0,1)), distribution='Normal'):
        '''
        Inherits the init function of the base class.
        :param theta: Parameters that have to be optimized
        '''
        super().__init__(self)
        self.params = 4
        self.distribution = distribution

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


