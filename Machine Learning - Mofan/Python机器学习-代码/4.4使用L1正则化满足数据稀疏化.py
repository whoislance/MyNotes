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


# 数据稀疏处理
from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l1')  # L1正则化
lr = LogisticRegression(penalty='l1',C=0.1)
lr.fit(X_train_std,y_train)
print('训练精度：',lr.score(X_train_std,y_train))
print('测试精度：',lr.score(X_test_std,y_test))
#训练精度： 0.983870967742
#测试精度： 0.98148148148  说明未出现过拟合
print(lr.intercept_) #得到截距项
# [-0.38377184 -0.1580767  -0.70047352]
# 三个截距项分别表示第i个类别相对于其他两个类别的匹配结果
print(lr.coef_) #得到权重数组包含三个权重系数向量，每个向量包含13个权重值，对应葡萄酒的13个特征


import matplotlib.pyplot as plt 
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue','green','red','cyan','magenta',
'yellow','black','pink','lightgreen','lightblue',
'gray','indigo','orange']
weights,params = [],[]
for c in np.arange(-4,6):
	lr = LogisticRegression(penalty='l1',C=10.0**c,random_state=0)
	lr.fit(X_train_std,y_train)
	weights.append(lr.coef_[1])
	params.append(10.0**c)
weights = np.array(weights)
for column,color in zip(range(weights.shape[1]),colors):
	plt.plot(params,weights[:,column],label=df_wine.columns[column+1],color=color)

plt.xlim([10.0**(-5),10.0**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',bbox_to_anchor=(1.38,1.03),ncol=1,fancybox=True)
plt.show()

