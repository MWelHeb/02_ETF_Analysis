# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 20:51:04 2021

@author: Marc Wellner
"""

import streamlit as st
#import yfinance as yf
import pandas as pd
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

#################(1) Load Data
etf_univ = pd.read_excel(locpath1+"etf_univ.xlsx", keep_default_na=False)
my_etfu = pd.read_excel(locpath1+"my_etf.xlsx", keep_default_na=False)



#################(2) Select Boxes (a) ETF (b) Time  

st.title('ETF Analysis')

####### (a) ETF
etf_univ_lst = etf_univ.loc[:,['etf_nm']]
etf_univ_lst = etf_univ_lst['etf_nm'].astype(str).values.tolist()

lst = []
lst = etf_univ_lst
lst.append('Select all')

ulst = st.sidebar.multiselect('Choose an ETF?',lst)

if 'Select all' in ulst :
	ulst = etf_univ_lst

#st.write(ulst)


####### (b) ETF
d1 = date.fromisoformat('2010-01-01')
d2 = date.today()
#d2 = date.today()
#print(today)
#print(startdate)

du1 = st.sidebar.date_input('Start date', d1)
du2 = st.sidebar.date_input('End date', d2)

#if du1 < du2:
#    st.success('Start date: `%s`\n\nEnd date: `%s`' % (du1, du2))
#else:
#    st.error('Error: End date must fall after start date.')

#delta = du2 - du1

dus1 = du1.strftime("%d/%m/%Y")
dus2 = du2.strftime("%d/%m/%Y")

##w/o user input 
#delta = d2 - d1
#ds1 = d1.strftime("%d/%m/%Y")
#ds2 = d2.strftime("%d/%m/%Y")

#st.write(dus1)
#st.write(dus2)

#################(3) Load User selected Data

etf_univ_sel = etf_univ[etf_univ['etf_nm'].isin(ulst)]
etf_univ_sel = etf_univ[etf_univ['etf_nm'].isin(ulst)]
etf_univ_selsc = etf_univ_sel.loc[:,['etf_sc']]
etf_univ_selsclst = etf_univ_selsc['etf_sc'].astype(str).values.tolist()
etf_univ_selsclst.append("datum")
my_etf = my_etfu[my_etfu.columns.intersection(etf_univ_selsclst)]
#my_etf["dates"] = my_etf.datum
my_etf.loc[:,"dates"] = pd.to_datetime(my_etf.loc[:,'datum']).dt.date
my_etf = my_etf[(my_etf.dates >= du1) & (my_etf.dates <= du2)] 

my_etf.index = pd.DatetimeIndex(my_etf["datum"])
my_etf = my_etf.drop(columns=["datum", "dates"])
#my_etf
for c in my_etf.columns.values:
    my_etf[c] = pd.to_numeric(my_etf[c],errors='coerce')
#my_etf.dtypes

rows, cols = my_etf.shape
etf_amt = cols

st.subheader('Raw data price')
st.write(my_etf)

st.subheader('Raw data price as index')
my_etf_i = my_etf/my_etf.iloc[0]
st.write(my_etf_i)

st.subheader('Raw data returns')
my_etf_r = my_etf.pct_change()
st.write(my_etf_r)

#################(4) Plot Time series 
#### (a)price
st.subheader('Time series plots')

title = 'ETF Close Price History'
fig, ax = plt.subplots()
for c in my_etf.columns.values:
    my_etf[c].plot(label = c, figsize=(14,5) )

plt.title(title)
plt.xlabel('Date')
plt.ylabel('Price EUR (€)')
plt.legend(my_etf.columns.values, loc='upper left')
#    plt.show()

st.pyplot(fig)


#### (b) price as index number 
#my_etf_i = my_etf/my_etf.iloc[0]

fig, ax = plt.subplots()
title = 'ETF Close Index History'
for c in my_etf_i.columns.values:
    my_etf_i[c].plot(label = c, figsize=(14,5) )
plt.title(title)
plt.xlabel('Date')
plt.ylabel('Index')
plt.legend(my_etf_i.columns.values, loc='upper left')
plt.show()

st.pyplot(fig)

#### (c) returns
#my_etf_r = my_etf.pct_change()

fig, ax = plt.subplots()
title = 'ETF Returns History'
for c in my_etf_r.columns.values:
    my_etf_r[c].plot(label = c, figsize=(14,5) )
plt.title(title)
plt.xlabel('Date')
plt.ylabel('Returns (in pct)')
plt.legend(my_etf_i.columns.values, loc='upper left')
plt.show()

st.pyplot(fig)


###############(5) Caluclation of Annual Returns and Volatility 

#### Returns
returns_annual = my_etf_r.mean()*252
#returns_annual

#### Variance
var_annual = my_etf_r.var()*252
#var_annual

#### Volatility
std_annual = np.sqrt(var_annual)
#std_annual

#### KPI
my_etf_kpi = pd.concat([returns_annual,std_annual], axis = 1)
my_etf_kpi.columns = ['Return', 'Volatility']
#my_etf_kpi

#st.subheader('Annual returns vs. volatility')
#st.write(my_etf_kpi)


#################(6) Caluclation of equal weighted Portfolio 

###List of weights
weights = [1/etf_amt] * etf_amt
#print(weights)
#weights

####Retunr
port_return = np.sum(my_etf_r.mean()*weights)*252
#port_return

#### Portfolio Variance/Volatility
# Annual Covariance of returns
cov_matrix_annual = my_etf_r.cov()*252
##cov_matrix_annual

port_variance = np.dot(weights, np.dot(cov_matrix_annual, weights))
#port_variance
port_volatility = np.sqrt(port_variance)
#port_volatility


##### KPI
pf_data = {'Return': [port_return], 'Volatility': [port_volatility]}
my_etfpf_kpi = pd.DataFrame(pf_data, columns = ['Return','Volatility'], index=['PFEqualw'])
#my_etfpf_kpi
my_etf_kpi = my_etf_kpi.append(my_etfpf_kpi)

st.subheader('Annual returns vs. volatility for ETFs vs equally weighted Portfolio')
st.write(my_etf_kpi)



#### (7) Optimal Portfolio (according to "Efficient Frontier")


from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

#my_etf

# Calculation of expected returns and annualized sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(my_etf)
S  = risk_models.sample_cov(my_etf)

# Optimize for max sharpe ratio (expected returns/volatility)
ef = EfficientFrontier(mu,S)
wght = ef.max_sharpe()
#print(wght)
clean_wght = ef.clean_weights()
#print(clean_wght)

index = ['Expected annual return:', 'Annual volatility:', 'Sharpe Ratio:']
columns = ['KPI opt PF']
kpi_opt_pf = pd.DataFrame(ef.portfolio_performance(verbose = True),columns=columns,  index=index)


from collections import OrderedDict, Counter

ord_list = [OrderedDict(clean_wght)]
#ord_list
col = Counter()
for k in ord_list:
    col.update(k)
wght_opt_pf = pd.DataFrame([k.values() for k in ord_list], columns = col.keys())


#Get the dicrete allocation for each share per stock

from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices = get_latest_prices(my_etf)
wght = clean_wght
da = DiscreteAllocation(wght, latest_prices, total_portfolio_value = 100000)
allocation, leftover = da.lp_portfolio()

alloc_opt_pf = pd.DataFrame.from_dict(allocation,orient = 'index', columns=['Amount'])


st.subheader('Optimal Portfolio according to Efficient Frontier would lead to ... ')

st.subheader('... following results:')
kpi_opt_pf

st.subheader('... following weights:')
wght_opt_pf

st.subheader('With an investment of $100.000 following allocation of ETF amounts would be optimal:')
alloc_opt_pf


#st.write('Clean weights:', clean_wght)
#st.write('Discrete allocation:', allocation)
#st.write('Funds remaining: ${:.2f}'.format(leftover))
#print('Discrete allocation:', allocation)
#print('Funds remaining: ${:.2f}'.format(leftover))



