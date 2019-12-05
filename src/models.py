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

class ZDGARCH(ARCH):

    def __init__(self, theta=None, bounds=((0,1),(0,1),(0,1))):

        super().__init__(self)
        self.params = 2
        if theta is None:
            self.theta = np.ones((self.params))
            self.theta[:2] = self.theta[0] * 0.1
            self.theta[2] = self.theta[1] * 0.8
        else:
            self.theta = theta

        self.bounds = bounds

    def get_sigma(self, theta, eps):

        sigma = np.zeros((len(eps), 1))
        sigma[0] = np.var(eps)

        for i in range(1, len(eps)):
            sigma[i] = theta[0] + theta[1]*sigma[i-1]

        return sigma

class EGARCH(ARCH):

    def __init__(self, theta=None, bounds=((0,1),(0,1),(0,1))):

        super().__init__(self)
        self.params = 2
        if theta is None:
            self.theta = np.ones((self.params))
            self.theta[:2] = self.theta[0] * 0.1
            self.theta[2] = self.theta[1] * 0.8
        else:
            self.theta = theta

        self.bounds = bounds

    def get_sigma(self, theta, eps):

        sigma = np.zeros((len(eps), 1))
        sigma[0] = np.var(eps)

        for i in range(1, len(eps)):
            sigma[i] = np.exp(theta[0] + theta[1] * np.power(eps[i-1], 2) + theta[2]*sigma[i-1])

        return sigma



import matplotlib.pyplot as plt
#
# df = pd.read_csv('../data/BTC-USD.csv')
#
# close_diff = np.log(df['Close']) - np.log(df['Close'].shift(1))
# close_diff = close_diff.dropna().values
#
# plt.plot(close_diff)

import datetime as dt
from arch import arch_model
import arch.data.sp500

st = dt.datetime(1988, 1, 1)
en = dt.datetime(2018, 1, 1)
data = arch.data.sp500.load()
market = data['Adj Close']
returns = 100 * market.pct_change().dropna()



am = arch_model(returns)
res = am.fit(update_freq=5)
print(res.summary())

model = GARCH()
model.fit(returns.values)
forecasts = model.forecast(returns)

plt.plot(returns.values)
plt.plot(forecasts)
plt.show()
# print(model.theta_hat)

# plt.show()