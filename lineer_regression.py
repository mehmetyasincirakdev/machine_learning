# Sales prediction with lineer regression

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn

pandas.set_option("display.float_format", lambda x: "%.2f" % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

dataFrame = pandas.read_csv("Datasets/advertising.csv")
dataFrame.head()
dataFrame.shape

X = dataFrame[["TV"]]
y = dataFrame[["sales"]]

# Model

regression_model = LinearRegression().fit(X, y)

# y_hat=b+ w*x
regression_model.intercept_[0]

# tv nin katsayısı w1
regression_model.coef_[0][0]

# Tahmin
# 500 birimlik tv harcaması olsa ne kadar satış olur
regression_model.intercept_[0] + regression_model.coef_[0][0] * 500
dataFrame.describe().T

# Modelin görselleştirilmesi

g = seaborn.regplot(x=X, y=y, scatter_kws={"color": "b", "s": 9}, ci=False, color="r")
g.set_title(f"Model Denklemi: Sales ={round(regression_model.intercept_[0], 2)} + TV*{round(regression_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show(block=True)

# Tahmin başarısı
# MSE
y_prediction = regression_model.predict(X)
mean_squared_error(y, y_pred=y_prediction)
y.mean()
y.std()

# RMSE
numpy.sqrt(mean_squared_error(y, y_prediction))
# MAE
mean_absolute_error(y, y_prediction)

# R_KARE

regression_model.score(X, y)

# Multiple Linear Regression

X = dataFrame.drop("sales", axis=1)
y = dataFrame[["sales"]]

# Modelleme

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

multiRegression_model = LinearRegression().fit(X_train, y_train)

multiRegression_model.intercept_
multiRegression_model.coef_

#Train RMSE
y_prediction=multiRegression_model.predict(X_train)
numpy.sqrt(mean_squared_error(y_train,y_prediction))

#Train RKARE
multiRegression_model.score(X_train,y_train)

y_prediction=multiRegression_model.predict(X_test)
numpy.sqrt(mean_squared_error(y_test,y_prediction))