# export PATH=/home/yule/anaconda3/bin:$PATH
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
 
iris=datasets.load_iris()
X=iris.data[:,[2,3]]
y=iris.target
 
#训练数据和测试数据分为7:3
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
sc.fit(x_train)
x_train_std=sc.transform(x_train) #使用上一行计算的均值和标准差来对数据做标准化处理
x_test_std=sc.transform(x_test)
X_combined_std = np.vstack((x_train_std, x_test_std))
y_combined = np.hstack((y_train, y_test))

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


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0,random_state=0)
#上一行中的参数C是正则化系数\lambda的倒数
lr.fit(x_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined,classifier=lr,test_idx=range(105,150))

#lr.predict_proba(x_test_std[2,:])
#得到结果：array=[ 0.70793846  1.50872803].



plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

