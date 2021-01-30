# export PATH=/home/yule/anaconda3/bin:$PATH

import numpy as np
import matplotlib.pyplot as plt

#通过随机噪声得到一个异或数据集
np.random.seed(0)
# randn函数返回一个或一组样本，具有标准正态分布。返回值为指定维度的array。
X_xor = np.random.randn(200,2)
#异或操作，使得一三象限值为1，二四象限值为-1
y_xor = np.logical_xor(X_xor[:,0]>0,X_xor[:,1]>0)
y_xor = np.where(y_xor,1,-1)


#显示二维分布图
plt.scatter(X_xor[y_xor==1,0],X_xor[y_xor==1,1],c='b',marker='x',label='1')
plt.scatter(X_xor[y_xor==-1,0],X_xor[y_xor==-1,1],c='r',marker='s',label='-1')
plt.ylim(-3.0)
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
def versiontuple(v):
 return tuple(map(int,(v.split('.'))))
def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
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
 #高亮测试集
 if test_idx:
  # 绘制所有数据点
  if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
   X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
   warnings.warn('Please update to NumPy 1.9.0 or newer')
  else:
   X_test, y_test = X[test_idx, :], y[test_idx]
  plt.scatter(X_test[:, 0], X_test[:, 1], c='',
    alpha=1.0, linewidth=1, marker='^',
    s=55, label='test set')

# 使用RBF核函数发现分离超平面
from sklearn.svm import SVC
svm = SVC(kernel='rbf',random_state=0,gamma=0.10,C=10.0)  #rbf:radial basis function kernel
svm.fit(X_xor,y_xor)
plot_decision_regions(X_xor,y_xor,classifier=svm)
plt.legend(loc='upper left')
plt.show()
