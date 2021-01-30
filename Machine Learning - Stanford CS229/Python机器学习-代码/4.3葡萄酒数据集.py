# export PATH=/home/yule/anaconda3/bin:$PATH

import pandas as pd 
import numpy as np

df_wine = pd.read_csv('wine.data',header=None)

#print(df_wine.head())

# 划分出测试集和训练集
from sklearn.model_selection import train_test_split
X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#归一化：最小-最大缩放
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
#X_train_norm = mms.fit_transform(X_train)
#X_test_norm = mms.transform(X_test)


#标准化：均值为0，方差为1
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std =stdsc.transform(X_test)

