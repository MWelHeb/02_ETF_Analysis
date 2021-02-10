# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:09:20 2021

@author: Marc Wellner
"""


import pandas as pd
import investpy
from pandas import DataFrame
from pandas.plotting import scatter_matrix
import numpy as np
import datetime
from datetime import datetime
from datetime import date
import investpy
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#from mpl_finance import candlestick_ohlc
from matplotlib.dates import DateFormatter, date2num, WeekdayLocator, DayLocator, MONDAY



locpath1 = "C:/Users/Marc Wellner/01_projects/streamlit/02_finance_app/01_data/"
#locpath1 = "/home/ubuntu/02_finance_app/01_data/"

############################################################
# (1) Time series of x selected ETF

#################(1) Generate ETF Universe

# (a) Selected ETF
etf_lst = ['iShares Core MSCI World UCITS',
           'iShares Core S&P 500 UCITS',
           'iShares NASDAQ-100 UCITS',
           'iShares EURO STOXX 50 UCITS',
           'iShares MSCI China A UCITS USD',
           'iShares Core DAX UCITS',
           'iShares MDAX UCITS DE',
           'iShares TecDAX UCITS']

etf_sc_lst = ['MSCIWorld', 
              'SP500',
              'NASDAQ100',
              'EURO50',
              'MSCIChina',
              'DAX',
              'MDAX',
              'TDAX']

etf_ctry_lst = ['united kingdom',
                'united kingdom',
                'germany',
                'germany',
                'united kingdom',
                'germany',
                'germany',
                'germany']

etf_num = ['etf1',
           'etf2',
           'etf3',
           'etf4',
           'etf5',
           'etf6',
           'etf7',
           'etf8']

etf_seli = {'etf_nm': etf_lst, 'etf_sc': etf_sc_lst, 'etf_ctry': etf_ctry_lst}
etf_univ = pd.DataFrame(etf_seli, columns = ['etf_nm','etf_sc', 'etf_ctry'], index=etf_num)
etf_univ.to_excel(locpath1+"etf_univ.xlsx", sheet_name='Tabelle1')


# (b) Selected Timeframe
ds1 = date.fromisoformat('2010-01-01').strftime("%d/%m/%Y")
ds2 = date.today().strftime("%d/%m/%Y")

rows, cols = etf_univ.shape
my_etf = pd.DataFrame({'A' : []})
for x in range(0, rows):
    print("We're on etf %d" % (x))
    print(etf_univ.iat[x,1])
    my_etf_c = investpy.etfs.get_etf_historical_data(etf = etf_univ.iat[x,0],country = etf_univ.iat[x,2], from_date=ds1, to_date=ds2, stock_exchange=None, as_json=False, order='ascending', interval='Daily')    
    my_etf_c = my_etf_c.loc[:,['Close']]            
    my_etf_c = my_etf_c.rename(columns={"Close": etf_univ.iat[x,1]})
    my_etf = pd.concat([my_etf, my_etf_c], axis = 1)


my_etf = my_etf.reset_index()
my_etf = my_etf.drop(columns=['A'])
my_etf = my_etf.rename(columns={"index": "datum"})
#my_etf
my_etf.to_excel(locpath1+"my_etf.xlsx", sheet_name='Tabelle1')






