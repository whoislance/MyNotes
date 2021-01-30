# export PATH=/home/yule/anaconda3/bin:$PATH

import numpy as np

class Perceptron(object):
	
	def __init__(self,eta=0.01,m_iter=10):
		self.eta = eta   #学习速率，浮点型，[0,1]
		self.m_iter = m_iter  #整型，passes over the training dataset，迭代次数？

	def fit(self,X,y):
		#np.zeros():返回来一个给定形状和类型的用0填充的数组
		#形状：1 + the number of samples
		self.w_ = np.zeros(1+X.shape[1]) #ld-array,weights after fitting
		self.errors_ = []  #list,number of misclassifications in every epoch
							#列表存放了每轮迭代中错误分类样本的数量，便于后续观察
		for _ in range(self.m_iter):
			errors = 0
			for xi,target in zip(X,y):
				update = self.eta * (target - self.predict(xi))
				self.w_[1:] += update*xi
				self.w_[0] += update
				errors += int(update !=0.0)
			self.errors_.append(errors)
		return self

	#得到净输入z
	def net_input(self,X):
		#np.dot(A, B)：
		#对于二维矩阵，计算真正意义上的矩阵乘积，同线性代数中矩阵乘法的定义。
		return np.dot(X,self.w_[1:]) + self.w_[0]
		#np.dot(a,b)  等价于  sum(i*j for i,j in zip(a,b))
	
	#得到当前迭代下的输出y
	def predict(self,X):
		#numpy.where(condition[], x, y)三者的维度相同
		#当conditon的某个位置的为true时，输出x的对应位置的元素，否则选择y对应位置的元素；
		return np.where(self.net_input(X)>=0.0,1,-1)


import pandas as pd
#读取CSV（逗号分割）文件到DataFrame
#header:指定行数用来作为列名，数据开始行数。如果文件中没有列名，则默认为0，否则设置为None。
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
#df.tail()


import matplotlib.pyplot as plt 
#把df中第四列（表示花的品种）赋给y，1和-1各代表一种花
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa',-1,1)
#把df中第0列和第2列（花的两个特征x1,x2）赋给X
X = df.iloc[0:100,[0,2]].values

#setosa:山鸢尾(-1)
plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
#versicolor：变色鸢尾(1)
plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
#萼片长度x1
plt.xlabel('petal length')
#花瓣长度x2
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()



#每次迭代的错误分类数量的折线图
ppn = Perceptron(eta=0.1,m_iter=10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

#决策边界的可视化
from matplotlib.colors import ListedColormap
def plot_decision_regions(X,y,classifier,resolution=0.02):
	markers=('s','x','o','^','v')
	colors=('red','blue','lightgreen','gray','cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])
	#对两个特征的最大值、最小值做了限定
	x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
	x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1
	#使用meshgrid()将最大值、最小值值向量生成二维数组xx1,xx2
	xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
	#将二维组展开，创建一个与训练数据集的列数相同的矩阵，以预测多维数组中所有对应点的类标z
	Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T) #array.T类似转秩吧
	Z = Z.reshape(xx1.shape)
	#使用contourf()画三维等高线图，对于网格数组中每个预测的类以不同的颜色绘制出决策区域
	plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
	plt.xlim(xx1.min(),xx1.max())
	plt.ylim(xx2.min(),xx2.max())
	#对于一个可迭代的/可遍历的对象（如列表、字符串），
	#enumerate将其组成一个索引序列，利用它可以同时获得索引和值
	for idx,cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)


plot_decision_regions(X,y,classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()