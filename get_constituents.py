import csv
import pandas as pd
from datetime import datetime
from pandas_datareader import data as web

index = pd.read_csv("./constituents_csv.csv")

start = "2000-01-01"
end = datetime.today().strftime('%Y-%m-%d')

US_Stock_List = index
target= US_Stock_List.stock_id
data=[]
for stock_id in target:
    try:
        df = web.get_data_yahoo(stock_id , start = start , end = end)
        df['stock_id'] = stock_id
        df=df.reset_index()
        print(stock_id )
        data.append(df)
    except:
        print('error',stock_id )
        pass

data = pd.concat(data)
data = data.rename(columns={'Date':'date'})
data=data.set_index(['stock_id'])
data['US_StockName']=US_Stock_List['US_StockName']
data=data.reset_index()
data=data.set_index(['stock_id','date'])