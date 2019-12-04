import unittest
import sys
sys.path.append("../")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.volatility_models import ARCH

class MyTestCase(unittest.TestCase):
    def test_something(self):

        df = pd.read_csv('../data/BTC-USD.csv')

        close = df['Close']

        y = close.values

        model = ARCH(p=2)
        model.fit(y)

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
