#export PATH=/home/yule/anaconda3/bin:$PATH

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

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


#决策区域可视化
from matplotlib.colors import ListedColormap
def versiontuple(v):
 return tuple(map(int,(v.split('.'))))
def plot_decision_regions(X,y,classifier,resolution=0.02):
 #设置标记点和颜色
 markers=('s','x','o','^','v')
 colors=('red','blue','lightgreen','gray','cyan')
 cmap=ListedColormap(colors[:len(np.unique(y))])
 
 # 绘制决策面
 #对两个特征的最大值、最小值做了限定
 x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
 x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
 #使用meshgrid()将最大值、最小值值向量生成二维数组xx1,xx2
 xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
       np.arange(x2_min, x2_max, resolution))
 #将二维组展开，创建一个与训练数据集的列数相同的矩阵，以预测多维数组中所有对应点的类标z
 # numpy.ravel()返回的是视图（view，也颇有几分C/C++引用reference的意味），会影响（reflects）原始矩阵。
 Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
 Z = Z.reshape(xx1.shape)
 #使用contourf()画三维等高线图，对于网格数组中每个预测的类以不同的颜色绘制出决策区域
 plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
 plt.xlim(xx1.min(), xx1.max())
 plt.ylim(xx2.min(), xx2.max())
 #对于一个可迭代的/可遍历的对象（如列表、字符串），
 #enumerate将其组成一个索引序列，利用它可以同时获得索引和值
 #X_test,y_test = X[test_idx,:],y[test_idx]
 for idx, cl in enumerate(np.unique(y)):
  plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
     alpha=0.8, c=cmap(idx),
     marker=markers[idx], label=cl)


from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
pca = PCA(n_components=2) #2个主成分
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca,y_train)
plot_decision_regions(X_test_pca,y_test,classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

# 研究方差贡献率
pca = PCA(n_components=None) #保留所有主成分
X_train_pca = pca.fit_transform(X_train_std)
#print(pca.explained_variance_ratio_) #得到相应的方差贡献率
'''
X_train_pca = pca.fit_transform(X_train_std)
'''