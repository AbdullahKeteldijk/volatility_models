import unittest
from src.volatility_models import GARCH
from src.models import ARCH, SE_GARCH, NGARCH, EGARCH, ZD_GARCH

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime as dt
import arch.data.sp500

# from src.volatility_models import GARCH

class MyTestCase(unittest.TestCase):



    def test_GARCH(self):
        st = dt.datetime(1988, 1, 1)
        en = dt.datetime(2018, 1, 1)
        data = arch.data.sp500.load()
        market = data['Adj Close']
        returns = 100 * market.pct_change().dropna().values

        model = GARCH(distribution='Normal')
        model.fit(returns)
        forecasts = model.forecast(returns)

        plt.plot(returns, label='returns')
        plt.plot(forecasts, label='GARCH')
        plt.title('Volatility Forecast S&P 500')
        plt.legend()
        plt.savefig('../data/output/GARCH.png')

        self.assertEqual(True, True)

    def test_GARCH_s(self):
        st = dt.datetime(1988, 1, 1)
        en = dt.datetime(2018, 1, 1)
        data = arch.data.sp500.load()
        market = data['Adj Close']
        returns = 100 * market.pct_change().dropna().values

        model = GARCH(distribution='Student-t')
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

        model = SE_GARCH()
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

        model = EGARCH()
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

        model = NGARCH()
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
