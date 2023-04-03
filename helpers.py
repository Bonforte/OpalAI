import yfinance as yf
from datetime import timedelta, datetime
import mplfinance as mpf
import pandas as pd
from numpy import trapz

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