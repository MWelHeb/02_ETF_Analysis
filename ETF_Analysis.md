# <a name="id0"></a>Analyzing ETF with Python 

### CONTENT
### [1 - Starting point](#id1)
### [2 - Data analysis](#id3)
#### [2a - Data Preparation](#id3a)
#### [2b - Web Application](#id3b)

### -----------------------------------------------------------------------------------------------------------------------------
### <a name="id1"></a>1 - Starting point [(Back to the Top)](#id0)

Next data science area which I was interested in related to the topic of finance. When it comes to investing money many advices center around investments into ETF (i.e. exhange traded funds vs. e.g. single stocks). Therefore I wanted to get a better understanding on the varity of ETS and their historical developments. How did the prices of different ETF develop? Which ETF had a large (annual) return over a given time period? How volatile were these returns? What is the ratio between return and volatility for a given ETF. Is there a strong postive/negative correlation amongst the returns of different ETF. Again, like in any data science project the first step centers around data: Where do I get appropriate data - ideally in a very automated way and always up to date? Once an interface to the relevant data source is available much work has to be conducted around the topic of analyzing this data, i.e. data preparation, applying statistical/econometric methods, preparation/visualization of results, interpretation, etc..  

### <a name="id3"></a>3 - Data analysis [(Back to the Top)](#id0)

The following python scripts contain the various steps of the data analysis which have been conducted: 

- xxx
- xxx

A further and more detailed description of these python script is given below.

#### <a name="id3a"></a>3a - Data Preparation [(Back to the Top)](#id0)

As always the initial step is about getting data on the topic of interest, in this case ETF. When searching the internet you find a large variety of potential libraries which offer an interface to financial data. E.g. one source is the library [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/index.html) which offers accesss to various (financial) datasources. After some Google search I decided to use the package [investpy](https://investpy.readthedocs.io/index.html) due to its documentation which I found helpfull and the easy access to a wide range of ETS in this library. According to the documentation library investpy retrieves data from the finance portal [investing.com](https://www.investing.com/)
