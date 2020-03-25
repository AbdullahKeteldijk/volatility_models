# How to run file

1. Create new folder in the root directory called project

2. Import the models as follows

```
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from src.volatility_models import GARCH

# import log returns

returns = pd.read("../data/returns.csv")

in_sample = returns[:100]
out_sample = returns[100:]

garch = GARCH()

garch.fit(in_sample)

prediction = garch.predict(out_sample)

# statistics
AIC = garch.AIC
BIC = garch.BIC
llik = garch.llik
theta_hat = garch.theta_hat
se = garch.se
z_values = garch.z_values
MSE = mean_squared_error(out_sample, prediction)
MAE = mean_absolute_error(out_sample, prediction)
```
