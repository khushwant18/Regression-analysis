# -*- coding: utf-8 -*-
"""
Created on Tue Oct 1 15:28:56 2019

@author: Khushwant Rai
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import KFold 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

data = pd.read_csv('Boston.csv')
correlation=0
i=1

#This function defines the dependent and indepedent variables in dataset
#For linear model only one idependent feature is chosen and for others three are chosen
#the outliers are also removed in this function
def get_data(lin=0):
    global data
    if lin == 1:
        x = data.iloc[:,[12]].values
        y = data.iloc[:,13].values
        index = np.where(np.round(stats.zscore(y),2) < 2.7)
        y = y[index]
        x = x[index]
    else:
        x = data.iloc[:,[5,10,12]].values
        y = data.iloc[:,13].values
        index = np.where(np.round(stats.zscore(y),2) < 2.7)
        y = y[index]
        x = x[index]
    return (x,y)

#This function is only used for data exploration purpose
def explore_data():
    global data, correlation
    print('Dataset summary',np.round(data.describe()))
    print(data.isnull().sum())
    plt.figure(0)
    plt.hist(data.iloc[:,13],density=True,bins=30)
    plt.xlabel('medv')
    correlation = round(data.corr(), 2)
    print(correlation)
    scatter_matrix(data.iloc[:,[2,5,9,10,12,13]])
    plt.show()

#This fuction performs the cross validation on four regression models    
def regression_analysis():
    scores = {}
    #fetch data for linear & polynomial model
    (xlin,ylin) = get_data(1)
    #fetch data for other models
    (x,y) = get_data()
    #dictionary for triversing functions of all the models
    models = {'linear_reg': linear_reg, 
              'poly_reg': polynomial_reg, 
              'reg_tree': tree_reg, 
              'SVR': svr_reg,
              'neural_reg': neural_reg}
    cross_val = KFold(n_splits=10, shuffle=True, random_state=None)
    for (train, test), (ltrain, ltest) in zip(cross_val.split(x), cross_val.split(xlin)):
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]
        #Linear and polynomial models have only one feature therefore different train, test sets
        xlin_train, xlin_test = xlin[ltrain], xlin[ltest]
        ylin_train, ylin_test = ylin[ltrain], ylin[ltest]
        for key, reg_model in models.items():
            if key in ['linear_reg','poly_reg']:
                #get the y_predict by calling function of model
                ylin_pred = reg_model(xlin_train, ylin_train, xlin_test, ylin_test)
                #calcuate the errors by calling evaluation function
                (rmse,mape) = evaluation(ylin_test, ylin_pred)
            else:
                (y_pred,nx_train,nx_test) = reg_model(x_train, y_train, x_test, y_test)
                (rmse,mape) = evaluation(y_test, y_pred)
            #store errors for the models in a dictionary
            if key in scores:
               scores[key]['rmse'].append(rmse)
               scores[key]['mape'].append(mape)
            else:
               scores.update({key:{'rmse':[rmse],'mape':[mape]}})
               #plot the models by calling plot_model function
               if key in ['linear_reg', 'poly_reg']:
                  plot_model(key,xlin_train,ylin_train,xlin_test,None,ylin_pred,'lstat','medv')
               else:
                  plot_model(key,nx_train,y_train,nx_test,y_test,y_pred,None,None)
    return scores                  

#performs linear regression
def linear_reg(x_train, y_train, x_test, y_test):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    return lr.predict(x_test)

#performs polynomial regresssion
def polynomial_reg(x_train, y_train, x_test, y_test):
    pr = LinearRegression()
    #transform feature to get higher order terms
    xp = PolynomialFeatures(degree=2)
    px_train = xp.fit_transform(x_train)
    px_test = xp.fit_transform(x_test)
    pr.fit(px_train, y_train)
    return pr.predict(px_test)

#performs regression tree 
def tree_reg(x_train, y_train, x_test, y_test):
    #transform features to reduce disparity in variance
    (x_train, x_test) = feature_scale(x_train, x_test)
    tr = DecisionTreeRegressor(random_state=0)
    tr.fit(x_train,y_train)
    tr.fit(x_train, y_train)
    return (tr.predict(x_test),x_train,x_test)

#performs support vector regression
def svr_reg(x_train, y_train, x_test, y_test):
    (x_train, x_test) = feature_scale(x_train, x_test)
    svr = SVR(kernel='rbf', gamma=0.1, C=1e3)
    svr.fit(x_train,y_train)
    return (svr.predict(x_test),x_train,x_test)

#performs neural regression
def neural_reg(x_train, y_train, x_test, y_test):
    (x_train, x_test) = feature_scale(x_train, x_test)
    nr = MLPRegressor(hidden_layer_sizes=(64,32,),activation='relu',max_iter=1000)
    nr.fit(x_train,y_train)
    return (nr.predict(x_test),x_train,x_test)
 
#transforms ths features and reduce mean to 0 and standard deviation to 1
def feature_scale(x_train, x_test):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    return (x_train, x_test)
    
#calculates the RMSE, MAPE and RSQUARE errors
def evaluation(y_test, y_pred):
    rmse =  np.round(np.sqrt(mean_squared_error(y_test,y_pred)),2)
    mape = np.round((np.mean(np.abs((y_test - y_pred) / y_test)) * 100),2)
    return (rmse, mape)

#plot th models
def plot_model(model_name,x_train,y_train,x_test,y_test,y_pred,x_lable,y_lable):
    global i
    i+=1
    plt.figure(i)
    plt.title(model_name)
    if model_name == 'linear_reg':
        plt.scatter(x_train,y_train, color='blue')
        plt.plot(x_test, y_pred, color='red')
    elif model_name == 'poly_reg':
        newx, newy = zip(*sorted(zip(x_test, y_pred)))
        plt.scatter(x_train,y_train, color='blue')
        plt.plot(newx, newy, color='red')
    else:
        plt.scatter(x_test[:,[0]],y_test, color='blue', label='y_test')
        plt.scatter(x_test[:,[1]],y_test, color='blue')
        plt.scatter(x_test[:,[0]],y_pred, color='red',marker='x', label='y_pred')
        plt.scatter(x_test[:,[1]],y_pred, color='red',marker='x')
        plt.legend(loc='upper left')
    plt.xlabel(x_lable)
    plt.ylabel(y_lable)

#run data exploration 
explore_data()  
#get the errors  
errors = regression_analysis()
mean_errors = {}
#get the mean of RMSE, MAPE ad rsquare errors for evrry model and stores in a dictionary
for model in errors:
    mean_errors.update({model:{'rmse':np.round(np.mean(errors[model]['rmse']),2),
                               'mape':np.round(np.mean(errors[model]['mape']),2)}})
        
    