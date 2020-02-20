import unittest
from src.volatility_models import GARCH
from src.models import ARCH, SE_GARCH, NGARCH, EGARCH, ZD_GARCH

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import datetime as dt
import arch.data.sp500

# from src.volatility_models import GARCH

class MyTestCase(unittest.TestCase):



    def testPlot(self):
        data = pd.read_csv('../data/input/AAPL.csv')
        market = data['Close']
        date = pd.to_datetime(data['Date'], yearfirst=True)
        # print(date)
        returns = 100 * market.pct_change().dropna().values

        plt.plot(date, market, label='price')
        plt.title('Apple Daily Closing Price')
        plt.legend()
        plt.savefig('../data/output/Apple_close.png')

    def testPlot2(self):
        data = pd.read_csv('../data/input/AAPL.csv')
        market = data['Close']
        date = pd.to_datetime(data['Date'], yearfirst=True)
        # print(date)
        returns = 100 * market.pct_change().dropna().values

        plt.plot(date.iloc[1:], returns, label='price')
        plt.title('Apple Daily Returns')
        plt.legend()
        plt.savefig('../data/output/Apple_returns.png')

    def testPlot3(self):
        data = pd.read_csv('../data/input/AAPL.csv')
        market = data['Close']
        date = pd.to_datetime(data['Date'], yearfirst=True)
        # print(date)
        returns = 100 * market.pct_change().dropna().values

        sns.distplot(returns, bins=100, label='price')
        plt.title('Apple Daily Returns Distribution')
        plt.legend()
        plt.savefig('../data/output/Apple_returns_histogram.png')



    def test_GARCH(self):

        data = pd.read_csv('../data/input/AAPL.csv')
        # st = dt.datetime(1988, 1, 1)
        # en = dt.datetime(2018, 1, 1)
        # data = arch.data.sp500.load()
        market = data['Close']
        returns = 100 * market.pct_change().dropna().values

        model = GARCH(distribution='Normal')
        model.fit(returns)
        forecasts = model.forecast(returns)

        plt.plot(returns, label='returns')
        plt.plot(forecasts, label='GARCH')
        plt.title('Apple Daily returns + GARCH')
        plt.legend()
        plt.savefig('../data/output/Gaussian_GARCH.png')

        self.assertEqual(True, True)

    def test_GARCH_s(self):
        data = pd.read_csv('../data/input/TSLA.csv')
        # st = dt.datetime(1988, 1, 1)
        # en = dt.datetime(2018, 1, 1)
        # data = arch.data.sp500.load()
        market = data['Close']
        returns = 100 * market.pct_change().dropna().values

        model = GARCH(distribution='Student-t')
        model.fit(returns)
        forecasts = model.forecast(returns)

        plt.plot(returns, label='returns')
        plt.plot(forecasts, label='GARCH')
        plt.title('Volatility Forecast S&P 500')
        plt.legend()
        plt.savefig('../data/output/Student_t_GARCH.png')

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
        data = pd.read_csv('../data/input/TSLA.csv')
        # st = dt.datetime(1988, 1, 1)
        # en = dt.datetime(2018, 1, 1)
        # data = arch.data.sp500.load()
        market = data['Close']
        returns = 100 * market.pct_change().dropna().values

        model = EGARCH(distribution='Normal')
        model.fit(returns)
        forecasts = model.forecast(returns)

        plt.plot(returns, label='returns')
        plt.plot(forecasts, label='EGARCH')
        plt.title('Volatility Forecast S&P 500')
        plt.legend()
        plt.savefig('../data/output/Gaussian_EGARCH.png')

        self.assertEqual(True, True)

    def test_NGARCH(self):
        data = pd.read_csv('../data/input/AAPL.csv')
        # st = dt.datetime(1988, 1, 1)
        # en = dt.datetime(2018, 1, 1)
        # data = arch.data.sp500.load()
        market = data['Close']
        returns = 100 * market.pct_change().dropna().values

        model = NGARCH(distribution='Normal')
        model.fit(returns)
        forecasts = model.forecast(returns)

        plt.plot(returns, label='returns')
        plt.plot(forecasts, label='NGARCH')
        plt.title('Volatility Forecast S&P 500')
        plt.legend()
        plt.savefig('../data/output/Gaussian_NGARCH.png')

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
        plt.title('Volatility Forecast AAPL')
        plt.legend()
        plt.savefig('../data/output/ZD_GARCH.png')

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
