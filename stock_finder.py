import yfinance as yf
from datetime import timedelta, datetime
import mplfinance as mpf
import pandas as pd
from numpy import trapz,nan
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.metrics import accuracy_score

# for modeling
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.utils import to_categorical 


ticker = yf.Ticker('nvda')
df_stock = ticker.history('2y')

df_stock['Date'] = df_stock.index.values

window_check_interval = 30
start_index = 0
df_stock['signal'] = 0
while(start_index < len(df_stock)):
    df_stock_filtered_interval = df_stock.iloc[start_index:start_index + window_check_interval]
    minimum_price = min(df_stock_filtered_interval['Close'])
    maximum_price = max(df_stock_filtered_interval['Close'])
    if maximum_price >= 1.1 * minimum_price:
        maximum_date = df_stock_filtered_interval.loc[df_stock_filtered_interval['Close'] == maximum_price, 'Date'].values[0]
        minimum_date = df_stock_filtered_interval.loc[df_stock_filtered_interval['Close'] == minimum_price, 'Date'].values[0]
        

        if maximum_date > minimum_date:
            df_stock.loc[(df_stock['Date'] >= minimum_date) & (df_stock['Date'] < maximum_date), 'signal'] = 1
        

    start_index += window_check_interval
plot_buy = []
x_values = []
for index, entry in df_stock.iterrows():
    if entry['signal']:
        plot_buy.append(entry['Close'])
    else:
        plot_buy.append(None)
    x_values.append(index)
plt.plot(df_stock['Close'])
plt.plot(x_values,plot_buy)
plt.show()