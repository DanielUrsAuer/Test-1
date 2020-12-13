# -*- coding: utf-8 -*-
"""
Created on Mon May 28 2020

Test scikit learn with very simple data

@author: auda
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
# import pandas as pd

plt.close('all')


'''
#################### create random data ####################
'''
# xdata = np.array([0,1,2,3,4,5])
# ydata = np.array([0,0.8,0.9,0.1,-0.8,-1])

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

# # straight line
# def func(x, a, b):
#     return a * x + b

elementCount = 1000
xdata = np.linspace(0, 10, elementCount)
y = func(xdata, 1.5, 1.3, 0.5)
# y = func(xdata, 2.5, 1.3)
np.random.seed(1729)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')

'''
#################### classic polyfit ####################
'''
# linear fit
p1 = np.polyfit(xdata,ydata,1)
print('linear fit coefficients:', p1)
# quadratic fit
p2 = np.polyfit(xdata,ydata,2)
print('quadratic fit coefficients:', p2)
# cubic fit
p3 = np.polyfit(xdata,ydata,3)
print('cubic fit coefficients:', p3)

plt.figure()
plt.plot(xdata,ydata,'o')
xp = np.linspace(-1,12,100)
plt.plot(xp,np.polyval(p1,xp), label='linear')
plt.plot(xp,np.polyval(p2,xp), '--', label='quadratic')
plt.plot(xp,np.polyval(p3,xp), ':', label='cubic')
plt.grid()
plt.legend()
plt.title('Poly. fit Example')
plt.show()

# residual for linear fit
yfit = p1[0] * xdata +p1[1]
#print(yfit)
yresid = ydata - yfit
SSresid = sum(pow(yresid,2)) # Sum Squre
SStotal = len(ydata) * np.var(ydata)
rsq = 1 - SSresid/SStotal
print('rsq manual calc for linear fit:', rsq)
print('Mean squared error polyfit: %.3f'
      % mean_squared_error(ydata, yfit))

# with scipy
slope,intercept,r_value,p_value,std_err = stats.linregress(xdata,ydata)
print('rsq scipy calc for linear fit:',pow(r_value,2))


'''
#################### scikit learn approach ####################

https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html
https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-download-auto-examples-linear-model-plot-ols-py

'''
print('\n')
# Split and Randomize the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(xdata, ydata,
                                                 test_size=0.3,
                                                 random_state=0)

print('data before reshape:',X_train.ndim, X_train.shape)
# reshape data
X_train = X_train.reshape(-1,1) #https://www.youtube.com/watch?v=sGCuryS8zjc
y_train = y_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
y_test = y_test.reshape(-1,1)
print('data after reshape:',X_train.ndim, X_train.shape)

plt.figure()
plt.plot(X_test, y_test,'o', label='test')
plt.plot(X_train, y_train, 'x', label='train')
plt.grid()
plt.legend()
plt.title('scikit test vs train data split')
plt.show()

from sklearn import linear_model
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
plt.plot(X_test, y_pred, '^', label='pred')
plt.grid()
plt.legend()
plt.title('scikit LinearRegression with degree default')
plt.show()

# The coefficients
print('Coefficients:', regr.coef_)
# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and y.
print('Score:', regr.score(X_test, y_test))


# The mean squared error
print('Mean squared error scikit: %.3f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.3f'
      % r2_score(y_test, y_pred))


plt.figure()
plt.plot(xp,np.polyval(p1,xp), label='linear polyfit')
plt.plot(X_test, y_pred, '^', label='pred scikit ML')
plt.grid()
plt.legend()
plt.title('polyfit vs. scikit')
plt.show()


'''
# Create regression object with higher degree
# https://codefying.com/2016/08/18/two-ways-to-perform-linear-regression-in-python-with-numpy-ans-sk-learn/
'''
print('\n')
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

degree = 4
regr = Pipeline([('poly', PolynomialFeatures(degree=degree)),('linear', linear_model.LinearRegression())])
# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The mean squared error
print('Mean squared error scikit higher degree: %.3f'
      % mean_squared_error(y_test, y_pred))

plt.figure()
plt.plot(X_test, y_test,'o', label='test')
plt.plot(X_train, y_train, 'x', label='train')
plt.plot(X_test, y_pred, '^', label='pred')
plt.grid()
plt.legend()
plt.title('scikit LinearRegression with higher degree')
plt.show()
