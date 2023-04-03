import pandas as pd
import yfinance as yf
from helpers import compute_ichimoku
from numpy import trapz,nan
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

rf_parameters = {
    "n_estimators":[5,10,50,100,250],
    "max_depth":[2,4,8,16,32,None]
}

ticker = yf.Ticker('AAPL')
df_stock = ticker.history('5y')
df_stock = df_stock
df_stock = df_stock
df_stock['Date'] = df_stock.index.values
df_stock = compute_ichimoku(df_stock)
df_stock = df_stock[(df_stock['senkou_a']>0) & (df_stock['senkou_b']>0)]

df_stock['cloud_avg_dist'] = (df_stock['senkou_a'] + df_stock['senkou_b']) / 2 - df_stock['Close']
df_stock['base_lines_avg_dist'] = (df_stock['tenkan_avg'] + df_stock['kijun_avg']) / 2 - df_stock['Close']
df_stock['kijun_dist'] = df_stock['kijun_avg'] - df_stock['Close']
df_stock['tenkan_dist'] = df_stock['tenkan_avg'] - df_stock['Close']
df_stock['senkou_a_dist'] = df_stock['senkou_a'] - df_stock['Close']
df_stock['senkou_b_dist'] = df_stock['senkou_b'] - df_stock['Close']
df_stock['surface'] = nan
df_stock['slope_tenkan'] = nan
df_stock['slope_kijun'] = nan

first_entry = df_stock.head(1)
last_order = first_entry['tenkan_avg'] < first_entry['kijun_avg']
border_dates = []
for index, row in df_stock.iterrows():
    new_order = row['tenkan_avg'] < row['kijun_avg']
    if new_order is not last_order:
        border_dates.append(row['Date'])

    last_sell_line = row['tenkan_avg']
    last_buy_line = row['kijun_avg']
    last_order = last_sell_line < last_buy_line

border_dates.append(df_stock['Date'].iloc[-1])
border_dates.pop(0)


date_intervals = []
for index in range(len(border_dates)-1):
    date_intervals.append((border_dates[index], border_dates[index+1]))

for interval in date_intervals:
    interval_df_stock = df_stock[(df_stock['Date'] >= interval[0]) & (df_stock['Date'] <= interval[1])]
    for index, entry in interval_df_stock.iterrows():
        entry_interval_df_stock = interval_df_stock[interval_df_stock['Date'] <= entry['Date']]
        interval_buy_line = entry_interval_df_stock['tenkan_avg']
        interval_sell_line = entry_interval_df_stock['kijun_avg']

        buy_area_interval = trapz(interval_buy_line, dx=1)
        sell_area_interval = trapz(interval_sell_line, dx=1)

        area_between_lines_interval = buy_area_interval - sell_area_interval
        df_stock.at[index,'surface'] = area_between_lines_interval



for index, entry in df_stock.iterrows():
    current_date = entry['Date']
    intermediary_df_stock = df_stock[df_stock['Date']<=current_date]
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

    df_stock.at[index,'slope_tenkan'] = slope_buy
    df_stock.at[index,'slope_kijun'] = slope_sell


df_stock = df_stock.dropna()

df_stock['signal'] = nan
df_stock_ai = df_stock[['Date', 'Volume', 'cloud_avg_dist', 'base_lines_avg_dist', 'kijun_dist', 'tenkan_dist', 'senkou_a_dist', 'senkou_b_dist', 'surface', 'slope_tenkan', 'slope_kijun', 'signal']]

normalize_column_names = ['Volume', 'cloud_avg_dist', 'base_lines_avg_dist', 'kijun_dist', 'tenkan_dist', 'senkou_a_dist', 'senkou_b_dist', 'surface', 'slope_tenkan', 'slope_kijun']
df_stock_ai[normalize_column_names]=(df_stock_ai[normalize_column_names]-df_stock_ai[normalize_column_names].min())/(df_stock_ai[normalize_column_names].max()-df_stock_ai[normalize_column_names].min())

counter1 = 0
counter2=0
for interval in date_intervals:
    interval_df_stock = df_stock[(df_stock['Date'] >= interval[0]) & (df_stock['Date'] <= interval[1])]
    interval_buy_line = interval_df_stock['tenkan_avg']
    interval_sell_line = interval_df_stock['kijun_avg']

    buy_area_interval = trapz(interval_buy_line, dx=1)
    sell_area_interval = trapz(interval_sell_line, dx=1)

    area_between_lines_interval = buy_area_interval - sell_area_interval
    counter1+=1
    if abs(area_between_lines_interval) >= 4:
        counter2+=1
        minimum_price = min(interval_df_stock['Close'])
        maximum_price = max(interval_df_stock['Close'])
        no_interval_entries = len(interval_df_stock['Close'])
        sell_index = None
        stop_sell_index = None
        buy_index = None
        stop_buy_index = None
        for index, entry in interval_df_stock.iterrows():
            
            if area_between_lines_interval >= 0:
                if entry['Close'] == minimum_price:
                    buy_index = index
                if entry['Close'] == maximum_price:
                    stop_buy_index = index
                
                buy_date = df_stock.at[buy_index, 'Date'] if buy_index else None
                stop_buy_date = df_stock.at[stop_buy_index, 'Date'] if stop_buy_index else None
                if stop_buy_date and buy_date:
                    df_stock_ai.loc[(df_stock_ai['Date'] >= buy_date) & (df_stock_ai['Date'] <= stop_buy_date), 'signal'] = 1
            elif area_between_lines_interval < 0:
                if entry['Close'] == minimum_price:
                    stop_sell_index = index
                if entry['Close'] == maximum_price:
                    sell_index = index

                sell_date = df_stock.at[sell_index, 'Date'] if sell_index else None
                stop_sell_date = df_stock.at[stop_sell_index, 'Date'] if stop_sell_index else None
                if sell_date and stop_sell_date:
                    df_stock_ai.loc[(df_stock_ai['Date'] >= sell_date) & (df_stock_ai['Date'] <= stop_sell_date), 'signal'] = 1

df_stock_ai.signal.fillna(value=0, inplace=True)
df_stock_ai['signal'] = df_stock_ai['signal'].astype(int)

#train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25)


rfc = RandomForestClassifier(max_depth=32, n_estimators=50)
rfc.fit(df_stock_ai.iloc[:-365].drop(['signal', 'Date'], axis=1), df_stock_ai.iloc[:-365]['signal'])

test_labels = df_stock_ai.tail(365)['signal']

pred_labels = rfc.predict(df_stock_ai.tail(365).drop(['signal', 'Date'], axis=1))
accuracy = accuracy_score(test_labels, pred_labels)
print("Accuracy:", accuracy)


test_profit_df = df_stock.tail(365)
test_profit_df['signal'] = pred_labels


start_date = test_profit_df['Date'].iloc[0]

first_entry = test_profit_df.head(1)
last_order = first_entry['tenkan_avg'] < first_entry['kijun_avg']
test_border_dates = []
for index, row in test_profit_df.iterrows():
    new_order = row['tenkan_avg'] < row['kijun_avg']
    if new_order is not last_order:
        test_border_dates.append(row['Date'])

    last_sell_line = row['tenkan_avg']
    last_buy_line = row['kijun_avg']
    last_order = last_sell_line < last_buy_line

test_border_dates.append(test_profit_df['Date'].iloc[-1])
test_border_dates.pop(0)


test_date_intervals = []
for index in range(len(test_border_dates)-1):
    test_date_intervals.append((test_border_dates[index], test_border_dates[index+1]))


transactions_profits = []
transaction_dates = []
transaction_total_values = []
for interval in test_date_intervals:
    filtered_test_profit_df = test_profit_df[(test_profit_df['Date']>=interval[0]) & (test_profit_df['Date']<=interval[1])]
    
    interval_buy_line = filtered_test_profit_df['tenkan_avg']
    interval_sell_line = filtered_test_profit_df['kijun_avg']

    buy_area_interval = trapz(interval_buy_line, dx=1)
    sell_area_interval = trapz(interval_sell_line, dx=1)

    area_between_lines_interval = buy_area_interval - sell_area_interval

    transaction_values =[]
    entry_price = None
    exit_price = None
    profit = None
    counter0=0
    counter1=0
    entry_flag = False
    buy_flag = None
    for index, entry in filtered_test_profit_df.iterrows():
        if entry['signal']:
            counter1 += 1
            counter0 = 0
        else:
            counter0 += 1

        if counter1 and entry_price is None:
            entry_price = entry['Close']
        if counter0>=3 and exit_price is None and entry_price:
            exit_price = entry['Close']

        if entry_price:
            transaction_values.append(entry['Close'])
        else:
            transaction_values.append(None)

        if entry_price and exit_price:
            if area_between_lines_interval>0:
                profit = ((exit_price/entry_price) * 100) - 100
                
            else:
                profit = ((entry_price/exit_price) * 100) - 100
            break

    transactions_profits.append(profit)
    transaction_total_values.extend(transaction_values)

        

print(transactions_profits)

initial_sum = 100
for profit in transactions_profits:
    initial_sum += initial_sum*(profit/100) if profit else 0

print(initial_sum-100)


signals = test_profit_df['signal']


# buy_transaction_plot = []
# sell_transaction_plot = []
# x_values =[]
# for index, entry in test_profit_df.iterrows():
#     if entry['signal']:
#         buy_transaction_plot.append(entry['Close']) if entry['tenkan_avg'] > entry['kijun_avg'] else buy_transaction_plot.append(None)
#         sell_transaction_plot.append(entry['Close']) if entry['tenkan_avg'] < entry['kijun_avg'] else sell_transaction_plot.append(None)
#     else:
#         buy_transaction_plot.append(None)
#         sell_transaction_plot.append(None)
#     x_values.append(index)

# fig, (ax1, ax2) = plt.subplots(2)

# ax1.plot(x_values, buy_transaction_plot, )
# ax1.plot(x_values, sell_transaction_plot, )
# ax2.plot(x_values, test_profit_df['Close'])
plt.plot(transaction_total_values)
plt.show()



    

        
