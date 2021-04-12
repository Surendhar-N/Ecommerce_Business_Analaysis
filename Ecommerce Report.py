# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 18:57:13 2021

@author: Surendhar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

"""
*****************************************************************************
Load Data and get its detail
*****************************************************************************
"""

ecomm_data = pd.read_csv('D:\ML Projects\Ecommerce Store\\Ecommerce Customers.csv')
print(ecomm_data['Yearly Amount Spent'].describe())
print(ecomm_data.describe())
ecomm_data.info()

"""
****************************************************************************
Analyse the dataset through graphical representation
**************************************************************************
"""

web = sns.jointplot(x='Time on Website',y ='Yearly Amount Spent', data = ecomm_data)
web.fig.suptitle("Correlation between Time on website and amount spent",y=1.1,fontsize = 20)

app = sns.jointplot(x='Time on App',y ='Yearly Amount Spent', data = ecomm_data)
app.fig.suptitle("Correlation between Time on App and amount spent",y=1.1,fontsize = 20)

ES = sns.pairplot(ecomm_data)
ES.fig.suptitle("Correlation between various features and sale",y =1.1,fontsize = 20)
plt.show()

ax = sns.heatmap(ecomm_data.corr(),cmap = 'Blues', annot=True)
plt.title("HeatMap of various veatures",y=1.1,fontsize = 20)
plt.show()

"""
****************************************************************************
Create a ML model to predict the evaluate the correlation
****************************************************************************
"""
X = ecomm_data[['Avg. Session Length', 'Time on App','Time on Website','Length of Membership']]
y = ecomm_data['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = LinearRegression()
model.fit(X_train,y_train)

predict = model.predict(X_test)

plt.scatter(X_test['Time on App'],y_test)
plt.scatter(X_test['Time on App'],predict)
plt.xlabel('Time on App')
plt.ylabel('Sales')
plt.legend(['Given_val','Predicted_val'],loc ="lower right")
plt.title("Scatter plot represnting given data set sales and predicted sales for Time on App")
plt.show()

plt.scatter(X_test['Time on Website'],y_test)
plt.scatter(X_test['Time on Website'],predict)
plt.xlabel('Time on Website')
plt.ylabel('Sales')
plt.legend(['Given_val','Predicted_val'],loc ="lower right")
plt.title("Scatter plot represnting given data set sales and predicted sales for Time on Website")
plt.show()

plt.scatter(X_test['Length of Membership'],y_test)
plt.scatter(X_test['Length of Membership'],predict)
plt.xlabel('Length of membership')
plt.ylabel('Sales')
plt.legend(['Given_val','Predicted_val'],loc ="lower right")
plt.title("Scatter plot represnting given data set sales and predicted sales for Length of membership")
plt.show()

plt.scatter(y_test,predict)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.title("Scatter plot represnting given data set sales and predicted sales")
plt.show()

"""
****************************************************************************
Evaluate the model and get coefficents of various features wit respect to sale
****************************************************************************
"""

print('MAE: {}'.format(metrics.mean_absolute_error(y_test, predict)))
print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, predict))))

coeff = pd.DataFrame(model.coef_ , X.columns, columns=['Coeffecient'])
print(coeff)