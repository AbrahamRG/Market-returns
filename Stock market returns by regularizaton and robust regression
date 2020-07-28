''' This code provides regularization models (Lasso, Ridge and ElasticNet) and Robust regression for modeling stock market quaterly returns with macroeconomic fundamentals'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import fredapi
import yfinance as yf
import yahoofinancials
import datetime as td
import seaborn as sns
import sklearn 
import math 
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from matplotlib.dates import DateFormatter

from yahoofinancials import YahooFinancials
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error


fred = Fred(api_key='Type your Fred API')
plt.style.use('seaborn') 

if __name__ == '__main__':
    GDP = fred.get_series('GDPC1', observation_start='1990-01-01', observation_end='2020')
    GDP = pd.DataFrame(GDP, columns=['GDPC1'])
    GDP['GDPGrowth'] = np.log(GDP.GDPC1).diff(periods = 1)
    GDP['ARY'] = GDP['GDPGrowth'].shift(1)
    GDP = GDP.iloc[1:]
    GDP.index = GDP.index.to_period("q")
    print(GDP)
    

    #Money Supply
    MS = fred.get_series('M2', observation_start='1990-01-01', observation_end='2020-03-31')
    MS = pd.DataFrame(MS, columns=['M2'])
    MS = MS.groupby(MS.index.to_period("q")).agg('mean')
    MS['GrowthMS'] = np.log(MS.M2).diff(periods = 1)
    MS = MS.iloc[1:]

    #CPI
    CPI = fred.get_series('CPIAUCSL', observation_start='1990-01-01', observation_end='2020-03-31')
    CPI = pd.DataFrame(CPI, columns=['CPIAUCSL'])
    CPI = CPI.groupby(CPI.index.to_period("q")).agg('mean')
    CPI['Prices'] = np.log(CPI.CPIAUCSL).diff(periods = 1)
    CPI['ARP'] = CPI['Prices'].shift(1)
    CPI = CPI.iloc[1:]
    
   
    tickers_list = YahooFinancials(['^GSPC','^VIX'])
    Index = tickers_list.get_historical_price_data("1990-01-01", "2020-03-31", "monthly")

    Stock = pd.DataFrame(Index['^GSPC']['prices'], columns=['date', 'close', 'formatted_date'])
    Volatility = pd.DataFrame(Index['^VIX']['prices'], columns=['date', 'close', 'formatted_date'])
  
        
    #Generate returns
    # Stocks
    Stock.date = pd.to_datetime(Stock.formatted_date)
    Stock = Stock.groupby(Stock['date'].dt.to_period('q'))['close'].agg('mean')
    Stock = pd.DataFrame(Stock, columns=['close'])
    Stock['Returns'] = np.log(Stock.close).diff(periods = 1)
    Stock['Clase'] = 'Stock'
    Stock1 = pd.DataFrame(Stock, columns=['Returns'])
    

    # Volatility
    Volatility.date = pd.to_datetime(Volatility.formatted_date)
    Volatility = Volatility.groupby(Volatility['date'].dt.to_period('q'))['close'].agg('mean')
    Volatility = pd.DataFrame(Volatility, columns=['close'])
    Volatility['Returns'] = np.log(Volatility.close).diff(periods = 1)
    Volatility['Clase'] = 'Volatility'
    Volatility1 = pd.DataFrame(Volatility, columns=['Returns'])

   
    #Rename returns 
    Stock1 = Stock1.rename(columns={'Returns': 'Stock'})
    Volatility1 = Volatility1.rename(columns={'Returns': 'Volatility'})
    df = pd.concat([GDP, Stock1, Volatility1, CPI], axis=1) 
    
   
    print(Stock.shape, 'Stock')
    print(Tech.shape, 'Tech')
    print(Gold.shape, 'Gold')
    print(CorpDebt.shape, 'CorpDebt')
    print(Debt.shape, 'Debt')
    print(Exchange.shape, 'Exchange')
    print(df.shape, 'Data' )
    print(df)


    #Correlations
    matriz = df.corr()
    print(matriz)

    #sns.heatmap(matriz)
    #plt.show()
    

  ###### Modeling
    print("=="*80)
    df.index = df.index.strftime('%m/%Y')
    
    #Take away Na
    df = df.iloc[2:]
    print(df)
    
    # Split in features and target
    x = df[['GDPGrowth', 'Volatility', 'Prices']] # 'ARS', 'ARV', 'ARG' , 'GrowthMS' , 'ReturnsTreasury'
    y = df[['Stock']]

    print(x.shape)
    print(y.shape)
    
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    train_data_len = math.ceil( len(x) * 0.8 )
    print(train_data_len)

    x_train = x.iloc[0:train_data_len, :]
    y_train = y.iloc[0:train_data_len, :]

    x_test = x[train_data_len : ]
    y_test = y[train_data_len : ]

    # Scale the data
    #scaler_x = MinMaxScaler(feature_range=(0,1)).fit_transform(x)
    #scaler_y = MinMaxScaler(feature_range=(0,1)).fit_transform(y)

    #x_train = scaler_x[0:train_data_len, :]
    #y_train = scaler_y[0:train_data_len, :]

    #Create teh data sets test
    #x_test = scaler_x[train_data_len : ]
    #y_test = y[train_data_len : ]

    print("="*80)
    print(x_test.shape)
   
   #Builts models
    modelLinear = LinearRegression().fit(x_train, y_train)
    y_predict_linear = modelLinear.predict(x_test)
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    linear_score = modelLinear.score(x_train, y_train) * 100

    modelLasso = LassoCV(alphas=[1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1], cv=10, random_state=0).fit(x_train, y_train)
    y_predict_lasso = modelLasso.predict(x_test)
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    lasso_score = modelLasso.score(x_train, y_train) * 100

    modelRidge = RidgeCV(alphas=[1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1], cv=7).fit(x_train, y_train)
    y_predict_ridge = modelRidge.predict(x_test)
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    ridge_score = modelRidge.score(x_train, y_train) * 100

    modelElastic = ElasticNetCV(cv=7).fit(x_train, y_train)
    y_predict_elastic = modelElastic.predict(x_test)
    elastic_loss = mean_squared_error(y_test, y_predict_elastic)
    elastic_score = modelElastic.score(x_train, y_train) * 100

    #Robust regression, it works very well with outliers
    modelHuber = HuberRegressor().fit(x_train, y_train)
    y_predict_huber = modelHuber.predict(x_test)
    huber_loss = mean_squared_error(y_test, y_predict_huber)
    huber_score = modelHuber.score(x_train, y_train) *100
    
    #Parameters
    print("="*80)
    print("Linear Regression  ||  MSE: {}  |  R^2: {}%".format(round(linear_loss, 6), round(linear_score, 2)))
    print("Lasso Regression   ||  MSE: {}  |  R^2: {}%".format(round(lasso_loss, 6), round(lasso_score, 2)))
    print("Ridge Regression   ||  MSE: {}  |  R^2: {}%".format(round(ridge_loss, 6), round(ridge_score, 2)))
    print("Elastic Regression ||  MSE: {}  |  R^2: {}%".format(round(elastic_loss, 6), round(elastic_loss, 2)))
    print("Huber Regression   ||  MSE: {}  |  R^2: {}%".format(round(huber_loss, 6), round(huber_loss, 2)))


    print("="*80)
    print("Linear Regression")
    print("Intercept   | ", modelLinear.intercept_)
    print("Coeficients | ", modelLinear.coef_)

    print("="*80)
    print("Lasso Regression")
    print("Intercept   | ", modelLasso.intercept_)
    print("Coeficients | ", modelLasso.coef_)

    print("-"*80)
    print("Ridge Regression")
    print("Intercept   | ", modelRidge.intercept_)
    print("Coeficients | ", modelRidge.coef_)

    print("-"*80)
    print("Elastic Regression")
    print("Intercept   | ", modelElastic.intercept_)
    print("Coeficients | ", modelElastic.coef_)

    print("-"*80)
    print("Huber Regression")
    print("Intercept   | ", modelHuber.intercept_)
    print("Coeficients | ", modelHuber.coef_)
    print("="*80)

    pd.plotting.register_matplotlib_converters()
    
    #Join forecast and valid data
    y_test['forecast_linear'] = y_predict_linear
    y_test['forecast_lasso'] = y_predict_lasso
    y_test['forecast_ridge'] = y_predict_ridge
    y_test['forecast_elastic'] = y_predict_elastic
    y_test['forecast_huber'] = y_predict_huber
    y_test = y_test[['forecast_linear','forecast_lasso','forecast_ridge','forecast_elastic','forecast_huber']]
    valid = pd.concat([y, y_test], axis=1).tail(50)

    valid.index = pd.to_datetime(valid.index)
    sns.set(style='ticks')
 
    #Plot Forecast and valid data
    fig, ax = plt.subplots()
    plt.title('Rendimientos (%) trimestrales promedio y pronóstico de regressión Lasso', size=12, x=0.001, horizontalalignment='left')
    plt.suptitle('S&P500: Rendimientos trimestrales', size=14, fontweight="bold", x=0.12, horizontalalignment='left')
    plt.xlabel("\n Fuente: FRED St Louis Fed | Elaboración propia", horizontalalignment='right', x=1.0, size=10, color='gray')
    plt.ylabel('Rendimientos %')
    ax.plot(valid['Stock'], color='blue', label='Rendimientos')
    #plt.plot(valid['forecast_linear'], color='red', label='Pronóstico Linear')
    #ax.plot(valid['forecast_lasso'], color='red', label='Pronóstico Lasso')
    plt.plot(valid['forecast_ridge'], color='g', label='Pronóstico Ridge')
    #plt.plot(valid['forecast_elastic'], color='orange', label='Pronóstico Elastic')
    #plt.plot(valid['forecast_huber'], color='black', label='Pronóstico Huber')
    plt.grid(True, which='both')
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    sns.despine(ax=ax, offset=0, right=True,  left=False) 
    plt.legend(loc=0)
    plt.show()

