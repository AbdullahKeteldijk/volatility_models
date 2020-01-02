import unittest
from src.models import GARCH, SE_GARCH, NGARCH, EGARCH, ZD_GARCH
from src.volatility_models import ARCH

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime as dt
import arch.data.sp500

# from src.volatility_models import ARCH

class MyTestCase(unittest.TestCase):




    def test_ARCH(self):
        st = dt.datetime(1988, 1, 1)
        en = dt.datetime(2018, 1, 1)
        data = arch.data.sp500.load()
        market = data['Adj Close']
        returns = 100 * market.pct_change().dropna().values

        theta = np.array([0.1, 0.1, 0.02, 0.01])
        bounds = ((0,1), (0,1), (0,1), (0,1))

        model = ARCH(theta=theta, bounds=bounds, p=3)
        model.fit(returns)
        forecasts = model.forecast(returns)

        plt.plot(returns, label='returns')
        plt.plot(forecasts, label='ARCH')
        plt.title('Volatility Forecast S&P 500')
        plt.legend()
        plt.savefig('../data/output/ARCH.png')

        self.assertEqual(True, True)

    def test_GARCH(self):
        # st = dt.datetime(1988, 1, 1)
        # en = dt.datetime(2018, 1, 1)
        # data = arch.data.sp500.load()
        data = pd.read_csv('../data/input/AAPL.csv')
        market = data['Adj Close']
        returns = market.pct_change().dropna().values
        # plt.plot(returns)
        # plt.show()

        theta = np.array([0.05, 0.1, 0.9])
        bounds = ((0.00001, 10), (0.00001, 10), (0.00001, 10))

        model = GARCH(theta=theta, bounds=bounds, p=1, q=1)
        print(model.method)
        model.fit(returns)
        forecasts = model.forecast(returns)

        plt.plot(returns, label='returns')
        plt.plot(forecasts, label='GARCH')
        plt.title('Volatility Forecast S&P 500')
        plt.legend()
        plt.savefig('../data/output/GARCH.png')

        self.assertEqual(True, True)

    def test_SE_GARCH(self):
        st = dt.datetime(1988, 1, 1)
        en = dt.datetime(2018, 1, 1)
        data = arch.data.sp500.load()
        market = data['Adj Close']
        returns = 100 * market.pct_change().dropna().values

        theta = np.array([0.2, 0.3, 0.6])
        bounds = ((0.00001, 100), (0, 1), (0, 1))

        model = SE_GARCH(theta, bounds)
        model.fit(returns)
        forecasts = model.forecast(returns)

        plt.plot(returns, label='returns')
        plt.plot(forecasts, label='SE_GARCH')
        plt.title('Volatility Forecast S&P 500')
        plt.legend()
        plt.savefig('../data/output/SE_GARCH.png')

        self.assertEqual(True, True)

    def test_EGARCH(self):
        st = dt.datetime(1988, 1, 1)
        en = dt.datetime(2018, 1, 1)
        data = arch.data.sp500.load()
        market = data['Adj Close']
        returns = 100 * market.pct_change().dropna().values

        theta = np.array([0.1, 0.1, 0.8, 0.1])
        bounds = ((0, 1), (0, 1), (0, 1), (0, 1))

        model = EGARCH(theta, bounds)
        model.fit(returns)
        forecasts = model.forecast(returns)

        plt.plot(returns, label='returns')
        plt.plot(forecasts, label='EGARCH')
        plt.title('Volatility Forecast S&P 500')
        plt.legend()
        plt.savefig('../data/output/EGARCH.png')

        self.assertEqual(True, True)

    def test_NGARCH(self):
        st = dt.datetime(1988, 1, 1)
        en = dt.datetime(2018, 1, 1)
        data = arch.data.sp500.load()
        market = data['Adj Close']
        returns = 100 * market.pct_change().dropna().values

        theta = np.array([0.1, 0.1, 0.8, 0.1])
        bounds = ((0, 1), (0, 1), (0, 1), (0, 1))

        model = NGARCH(theta, bounds)
        model.fit(returns)
        forecasts = model.forecast(returns)

        plt.plot(returns, label='returns')
        plt.plot(forecasts, label='NGARCH')
        plt.title('Volatility Forecast S&P 500')
        plt.legend()
        plt.savefig('../data/output/NGARCH.png')

        self.assertEqual(True, True)

    def test_ZD_GARCH(self):
        st = dt.datetime(1988, 1, 1)
        en = dt.datetime(2018, 1, 1)
        data = arch.data.sp500.load()
        market = data['Adj Close']
        returns = 100 * market.pct_change().dropna().values

        theta = np.array([0.1, 0.8])
        bounds = ((0, 1), (0, 1))

        model = ZD_GARCH()
        model.fit(returns)
        forecasts = model.forecast(returns)

        plt.plot(returns, label='returns')
        plt.plot(forecasts, label='ZD_GARCH')
        plt.title('Volatility Forecast S&P 500')
        plt.legend()
        plt.savefig('../data/output/ZD_GARCH.png')

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
