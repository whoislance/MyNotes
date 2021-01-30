# export PATH=/home/yule/anaconda3/bin:$PATH
import pandas as pd 
from io import StringIO
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data)) #将CSV格式的数据读取到数据框dataframe中

#打印df
print(df)

#访问value属性得到相关的numpy数组
print(df.values) 

#显示每列中的缺失值
print(df.isnull().sum())

#删除数据集里包含缺失值的行
print(df.dropna())

#设置参数，删除数据集里至少包含一个缺失值的列
print(df.dropna(axis=1))

#均值补差
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN',strategy='mean',axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data)

# export PATH=/home/yule/anaconda3/bin:$PATH
import pandas as pd 
from io import StringIO
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data)) #将CSV格式的数据读取到数据框dataframe中

#打印df
print(df)

#访问value属性得到相关的numpy数组
print(df.values) 

#显示每列中的缺失值
print(df.isnull().sum())

#删除数据集里包含缺失值的行
print(df.dropna())

#设置参数，删除数据集里至少包含一个缺失值的列
print(df.dropna(axis=1))

#均值补差
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN',strategy='mean',axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data)







