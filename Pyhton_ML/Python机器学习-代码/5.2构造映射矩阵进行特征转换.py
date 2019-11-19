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
#print('\nEigenvalues \n%s' % eigen_vals)
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
 

#排列特征对-降序
eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

#选取两个对应的特征值最大的特征向量：得到13*2的映射矩阵
w = np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))
# print('Matrix W:\n',w) 

#将124*13的数据集转换到包含两个主成分的子空间上
X_train_pca = X_train_std.dot(w)

#可视化子空间
import matplotlib.pyplot as plt 
colors = ['r','b','g']
markers = ['s','x','o']

for l,c,m in zip(np.unique(y_train),colors,markers):
	plt.scatter(X_train_pca[y_train==l,0],X_train_pca[y_train==l,1],c=c,label=l,marker=m)
	# l依次等于1,2,3，因为数据第一列是类标
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()