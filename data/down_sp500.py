#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import yfinance as yf
import time
    
def down_load_sp_500():
    df_names = pd.read_csv('./SP500_ Shares_outstanding.csv')
    names = df_names['SYMBOL'].tolist()
    name_shares = df_names.set_index('SYMBOL').T.to_dict('SHARES')
    df1 = pd.read_csv('./SP500_price.csv')
    names1 = df1['code'].tolist()
    df2 = pd.read_csv('./SP500_price1.csv')
    names2 = df2['code'].tolist()
    names = list(set(names)-(set(names1)|set(names2)))
    print('there are still {} stocks to download'.format(len(names)))
    j = 0
    df = None
    for name_i in names:
        try:
            shares = name_shares[name_i]['SHARES']
            data = yf.download(name_i, 
                      start='1990-01-01', 
                      end='2021-2-10', 
                      progress=False)
            #data = web.DataReader(name_i, "av-daily-adjusted", start=datetime(1990, 1, 1),
            #end=datetime(2021, 2, 10), api_key='EB0U6I0ICWGMSYAT')
            data.insert(data.shape[1], 'dividend amount', 0)
            data.insert(data.shape[1], 'split coefficient', 1.0)
            data.insert(data.shape[1], 'shares', shares)
            data.insert(data.shape[1], 'code', name_i)
            if j==0:
                df = data
            else:
                df = pd.concat([df, data], axis=0)
            j += 1
            if j%5==0:
                print(j)
                #time.sleep(65)
        except Exception as e:
            print(e, name_i)
            continue
    df.to_csv('SP500_price2.csv')

if __name__ == '__main__':
    print('test') 
    down_load_sp_500()
