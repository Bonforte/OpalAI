from keras.models import load_model

import yfinance as yf
from datetime import timedelta, datetime
import mplfinance as mpf
import pandas as pd
import time
from numpy import trapz,nan
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.metrics import accuracy_score

# for modeling
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical 

print("Libraries Imported ...")


def compute_ichimoku(df_stock):
    #Tenkan Sen
    tenkan_max = df_stock['High'].rolling(window = 9, min_periods = 0).max()
    tenkan_min = df_stock['Low'].rolling(window = 9, min_periods = 0).min()
    df_stock['tenkan_avg'] = (tenkan_max + tenkan_min) / 2

    #Kijun Sen
    kijun_max = df_stock['High'].rolling(window = 26, min_periods = 0).max()
    kijun_min = df_stock['Low'].rolling(window = 26, min_periods = 0).min()
    df_stock['kijun_avg'] = (kijun_max + kijun_min) / 2

    df_stock['senkou_a'] = ((df_stock['kijun_avg'] + df_stock['tenkan_avg']) / 2).shift(26)

    #Senkou Span B
    #52 period High + Low / 2
    senkou_b_max = df_stock['High'].rolling(window = 52, min_periods = 0).max()
    senkou_b_min = df_stock['Low'].rolling(window = 52, min_periods = 0).min()
    df_stock['senkou_b'] = ((senkou_b_max + senkou_b_min) / 2).shift(52)

    #Chikou Span
    #Current close shifted -26
    df_stock['chikou'] = (df_stock['Close']).shift(-26)

    return df_stock

def get_adx(high, low, close, lookback):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.rolling(lookback).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
    adx_smooth = adx.ewm(alpha = 1/lookback).mean()
    return plus_di, minus_di, adx_smooth

stocks = ['QCOM', 'NKE']
#stocks = ['NFLX', 'RACE', 'MPC', 'SLB']
df_stocks = dict()

for stock in stocks:
    df_stocks[stock] = yf.Ticker(stock).history('2y')

    df_stocks[stock]['Date'] = df_stocks[stock].index.values
    df_stocks[stock] = compute_ichimoku(df_stocks[stock])
    df_stocks[stock] = df_stocks[stock].drop(['chikou', 'Dividends', 'Stock Splits'], axis=1).dropna()
    df_stocks[stock]['cloud_avg_dist'] = (df_stocks[stock]['senkou_a'] + df_stocks[stock]['senkou_b']) / 2 - df_stocks[stock]['Close']
    df_stocks[stock]['base_lines_avg_dist'] = (df_stocks[stock]['tenkan_avg'] + df_stocks[stock]['kijun_avg']) / 2 - df_stocks[stock]['Close']
    df_stocks[stock]['kijun_dist'] = df_stocks[stock]['kijun_avg'] - df_stocks[stock]['Close']
    df_stocks[stock]['tenkan_dist'] = df_stocks[stock]['tenkan_avg'] - df_stocks[stock]['Close']
    df_stocks[stock]['senkou_a_dist'] = df_stocks[stock]['senkou_a'] - df_stocks[stock]['Close']
    df_stocks[stock]['senkou_b_dist'] = df_stocks[stock]['senkou_b'] - df_stocks[stock]['Close']
    df_stocks[stock]['senkou_range'] = df_stocks[stock]['senkou_a'] - df_stocks[stock]['senkou_b']
    df_stocks[stock]['baseline_range'] = df_stocks[stock]['tenkan_avg'] - df_stocks[stock]['kijun_avg']
    df_stocks[stock]['high_low_difference'] = df_stocks[stock]['High'] - df_stocks[stock]['Low']
    df_stocks[stock]['adx'] = pd.DataFrame(get_adx(df_stocks[stock]['High'], df_stocks[stock]['Low'], df_stocks[stock]['Close'], 14)[2]).rename(columns = {0:'adx'})
    df_stocks[stock]['surface'] = nan
    df_stocks[stock]['slope_tenkan'] = nan
    df_stocks[stock]['slope_kijun'] = nan

print("Stocks fetched ...")

date_intervals = dict()
for stock in stocks:
    first_entry = df_stocks[stock].head(1)
    last_order = first_entry['tenkan_avg'] < first_entry['kijun_avg']
    border_dates = []
    for index, row in df_stocks[stock].iterrows():
        new_order = row['tenkan_avg'] < row['kijun_avg']
        if new_order is not last_order:
            border_dates.append(row['Date'])

        last_sell_line = row['tenkan_avg']
        last_buy_line = row['kijun_avg']
        last_order = last_sell_line < last_buy_line

    border_dates.append(df_stocks[stock]['Date'].iloc[-1])
    border_dates.pop(0)


    date_intervals[stock] = []
    for index in range(len(border_dates)-1):
        date_intervals[stock].append((border_dates[index], border_dates[index+1]))

for stock in stocks: 
    for interval in date_intervals[stock]:
        interval_df_stock = df_stocks[stock][(df_stocks[stock]['Date'] >= interval[0]) & (df_stocks[stock]['Date'] <= interval[1])]
        for index, entry in interval_df_stock.iterrows():
            entry_interval_df_stock = interval_df_stock[interval_df_stock['Date'] <= entry['Date']]
            interval_buy_line = entry_interval_df_stock['tenkan_avg']
            interval_sell_line = entry_interval_df_stock['kijun_avg']

            buy_area_interval = trapz(interval_buy_line, dx=1)
            sell_area_interval = trapz(interval_sell_line, dx=1)

            area_between_lines_interval = buy_area_interval - sell_area_interval
            df_stocks[stock].at[index,'surface'] = area_between_lines_interval

for stock in stocks:    
    for index, entry in df_stocks[stock].iterrows():
        current_date = entry['Date']
        intermediary_df_stock = df_stocks[stock][df_stocks[stock]['Date']<=current_date]
        slope_sell = None
        slope_buy = None
        interval_buy_line = intermediary_df_stock['tenkan_avg']
        interval_sell_line = intermediary_df_stock['kijun_avg']
        if len(interval_sell_line) > 2 and len(interval_buy_line) > 2:
            x1_sell,y1_sell = 1, interval_sell_line[-2]
            x2_sell,y2_sell = 2, interval_sell_line[-1]
            slope_sell = ((y2_sell-y1_sell)/(x2_sell-x1_sell))
            x1_buy,y1_buy = 1, interval_buy_line[-2]
            x2_buy,y2_buy = 2, interval_buy_line[-1]
            slope_buy = ((y2_buy-y1_buy)/(x2_buy-x1_buy))

        df_stocks[stock].at[index,'slope_tenkan'] = slope_buy
        df_stocks[stock].at[index,'slope_kijun'] = slope_sell
    df_stocks[stock] = df_stocks[stock].dropna()

print("Structures created ...")
window_check_interval = 30

for stock in stocks:
    start_index = 0
    df_stocks[stock]['signal'] = 0
    while(start_index < len(df_stocks[stock])):
        df_stock_filtered_interval = df_stocks[stock].iloc[start_index:start_index + window_check_interval]
        minimum_price = min(df_stock_filtered_interval['Close'])
        maximum_price = max(df_stock_filtered_interval['Close'])
        if maximum_price >= 1.1 * minimum_price:
            maximum_date = df_stock_filtered_interval.loc[df_stock_filtered_interval['Close'] == maximum_price, 'Date'].values[0]
            minimum_date = df_stock_filtered_interval.loc[df_stock_filtered_interval['Close'] == minimum_price, 'Date'].values[0]
            

            if maximum_date > minimum_date:
                df_stocks[stock].loc[(df_stocks[stock]['Date'] >= minimum_date) & (df_stocks[stock]['Date'] < maximum_date), 'signal'] = 1
            

        start_index += window_check_interval

print("Classes recorded ...")
for stock in stocks:
    buy_counter = 0
    for index, row in df_stocks[stock].iterrows():
        
        if row['signal'] == 1:
            buy_counter += 1

        if row['signal'] == 0 and buy_counter != 0:
            if buy_counter < 5:
                df_stocks[stock].loc[index-pd.DateOffset(days=buy_counter):index, 'signal'] = 0
            buy_counter = 0


df_stocks_filtered = dict()
ncn = ['Volume', 'cloud_avg_dist', 'adx', 'base_lines_avg_dist', 'kijun_dist', 'senkou_range', 'baseline_range', 'high_low_difference', 'tenkan_dist', 'senkou_a_dist', 'senkou_b_dist', 'surface', 'slope_tenkan', 'slope_kijun']

for stock in stocks:
    df_stocks_filtered[stock] = df_stocks[stock][['Date', 'Volume', 'adx', 'cloud_avg_dist', 'base_lines_avg_dist', 'senkou_range', 'baseline_range', 'high_low_difference', 'kijun_dist', 'tenkan_dist', 'senkou_a_dist', 'senkou_b_dist', 'surface', 'slope_tenkan', 'slope_kijun', 'signal']]

    df_stocks_filtered[stock][ncn] = (df_stocks_filtered[stock][ncn] - df_stocks_filtered[stock][ncn].min())/(df_stocks_filtered[stock][ncn].max()-df_stocks_filtered[stock][ncn].min())


print("Columns filtered ...")

stock_names = []
for stock in stocks:
    for i in range(6):
        model=load_model(f'stock_models/{stock}/model{i}.h5')
        X = df_stocks_filtered[stock].drop(['signal', 'Date'], axis=1)
        Y = df_stocks_filtered[stock]['signal']

        X_test = X[-180:]
        Y_test = Y[-180:]


        Y_pred = model.predict(X_test)

        if model.evaluate(X_test, Y_pred, verbose=0)[0] <= 0.2 and stock not in stock_names:
            stock_names.append(stock) 

    plot_buy = []
    x_values = []
    df_stocks[stock][-180:]['signal'] = Y_pred
    for index, entry in df_stocks[stock][-180:].iterrows():
        if entry['signal']:
            plot_buy.append(entry['Close'])
        else:
            plot_buy.append(None)
        x_values.append(index)
    plt.plot(df_stocks[stock][-180:]['Close'])
    plt.plot(x_values,plot_buy)
    plt.show()

print(stock_names)
# print(Y_pred)


