# export PATH=/home/yule/anaconda3/bin:$PATH

import pandas as pd 

df = pd.DataFrame([
	['green','M',10.1,'class1'],
	['red','L',13.5,'class2'],
	['blue','XL',15.3,'class1']
])
df.columns = ['color','size','price','classlabel']

#打印df
print(df)

#将类别字符串转换为整数
size_mapping = {
	'XL':3,
	'L':2,
	'M':1
}
inv_size_mapping = {v:k for k,v in size_mapping.items()} #逆映射字典
df['size'] = df['size'].map(size_mapping)
print(df)

# 用枚举的方式设定类标
import numpy as np
class_mapping = {
	label:idx for idx,label in enumerate(np.unique(df['classlabel']))
}
inv_class_mapping = {v:k for k,v in class_mapping.items()} #逆映射字典
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)

# 对类标的整数编码工作
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)

# 处理标称格式的color列
X = df[['color','size','price']].values
color_le = LabelEncoder()
X[:,0] = color_le.fit_transform(X[:,0])
print(X)

# 独热编码技术:创建一个新的虚拟特征，每一列各代表标称数据的一个值
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
print(ohe.fit_transform(X).toarray())

#使用pands中的get_dummies方法：实现独热编码技术
print(pd.get_dummies(df[['price','color','size']]))

