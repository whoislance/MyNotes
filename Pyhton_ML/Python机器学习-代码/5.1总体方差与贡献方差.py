# export PATH=/home/yule/anaconda3/bin:$PATH

import pandas as pd 
import numpy as np

df_wine = pd.read_csv('wine.data',header=None)

#print(df_wine.head())

# 划分出测试集和训练集
from sklearn.model_selection import train_test_split
X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#标准化：均值为0，方差为1
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std =stdsc.transform(X_test)

#计算协方差矩阵的特征对
cov_mat = np.cov(X_train_std.T) #得到协方差矩阵
eigen_vals,eigen_vecs = np.linalg.eig(cov_mat) #得到特征向量，存入13*13的eigen_vecs中
print('\nEigenvalues \n%s' % eigen_vals)
'''
Eigenvalues
[ 4.8923083   2.46635032  1.42809973  1.01233462  0.84906459  0.60181514
  0.52251546  0.08414846  0.33051429  0.29595018  0.16831254  0.21432212
  0.2399553 ]
'''

#绘制特征值的方差贡献率图像
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals,reverse=True)] #计算每个的方差贡献率
cum_var_exp = np.cumsum(var_exp)  #计算出累计方差
 
import matplotlib.pyplot as plt 
plt.bar(range(1,14),var_exp,alpha=0.5,align='center',label='individual explained variance')
plt.step(range(1,14),cum_var_exp,where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()
