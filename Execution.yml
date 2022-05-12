from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('netflix.csv')
dataset.describe()
x = dataset[['High','Low','Open','Volume']].values
y = dataset[['Close']].values
x_train,x_test, y_train, y_test  = train_test_split(x,y,test_size= 0.2,random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print(regressor.coef_)
print(regressor.intercept_)
predicted = regressor.predict(x_test)
print(predicted)
df = pd.DataFrame({'Actual':y_test.flatten(), 'Predicted': predicted.flatten()})
print(df)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,predicted))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,predicted))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,predicted)))
import math
graph = df.head(20)
graph.plot(kind= 'bar')
np.exp(df[['Actual','Predicted']].cumsum()).plot
plt.title("Stock price")
plt.xlabel("days")
plt.ylabel("Close")
plt.plot(predicted, color="Red")

plt.show()
