from src.volatility_models import ARCH
import numpy as np

class GARCH(ARCH):

    def __init__(self, theta=None, bounds=((0,1),(0,1),(0,1))):

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

        sigma = np.zeros((len(eps), 1))
        sigma[0] = np.var(eps)

        for i in range(1, len(eps)):
            sigma[i] = theta[0] + theta[1] * np.power(eps[i-1], 2) + theta[2]*sigma[i-1]

        return sigma

class ZD_GARCH(ARCH):

    def __init__(self, theta=None):

        super().__init__(self)
        self.params = 2
        if theta is None:
            self.theta = np.ones((self.params))
            self.theta[0] = self.theta[0] * 0.1
            self.theta[1] = self.theta[1] * 0.8
        else:
            self.theta = theta


    def get_sigma(self, theta, eps):

        sigma = np.zeros((len(eps), 1))
        sigma[0] = np.var(eps)

        for i in range(1, len(eps)):
            sigma[i] = theta[0] + theta[1]*sigma[i-1]

        return sigma

class EGARCH(ARCH):

    def __init__(self, theta=None, bounds=((0,1),(0,1),(0,1),(0,1))):

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

        sigma = np.zeros((len(eps), 1))
        sigma[0] = np.var(eps)

        for i in range(1, len(eps)):
            sigma[i] = theta[0] + theta[1] * np.power(eps[i-1], 2) *sigma[i-1] + theta[2]*sigma[i-1]

        return sigma

class NGARCH(ARCH):

    def __init__(self, theta=None, bounds=((0,1),(0,1),(0,1),(0,1))):

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

        sigma = np.zeros((len(eps), 1))
        sigma[0] = np.var(eps)

        for i in range(1, len(eps)):
            eps_plus = eps[i-1] / np.sqrt(sigma[i-1])
            sigma[i] = theta[0] + theta[1] * sigma[i-1] + theta[2]*sigma[i-1] * np.power((eps_plus-theta[3]),2)

        return sigma


