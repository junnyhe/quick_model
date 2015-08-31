import pandas as pd
import numpy as np
from numpy import *
import sys
sys.path.append("tools/model_tools")
sys.path.append("tools/csv_operations")

################### training #####################
# load train data
df = pd.read_csv('../data/data1_val_imp_woe.csv.gz',compression="gzip")
field_names=df.columns.values
field_types=df.dtypes

# select
df['target'] # by column
df[['amount','target']] # by columns
df[0:10] #by rows
df[['amount','target']][0:10] # by columns and rows ###### most straight forward
df[0:10][['amount','target']] # same as above
df.ix[0:10,['amount','target']] # by label
df.loc[0:10,['amount','target']]# by label (same as above)
df.ix[0:10,0:2] # by position
df.iloc[0:10,0:2]# by position
df[df['target']==1] # filter/where by row, input is a boolean

# merge
df1=df[['payment_request_id','amount']]
df2=df[['payment_request_id','target']]
df12=pd.merge(df1,df2,on='payment_request_id')

# group by
df.groupby(['target','signal_638']).count() # for all fields
df.groupby(['target','signal_638'])['amount'].mean() # for one field to calculate agg
df.groupby(['target','signal_638'])['amount'].agg(mean,count) # another syntax
df.groupby(['target','signal_638'])['amount'].agg(['mean','count','std','size','max','min','sum']) #multiple aggs

# pivot

pd.pivot_table(df, values='amount', index=['target','signal_638'], columns=['signal_2'],aggfunc=len) # sum, mean, len, size, max, min, median, std