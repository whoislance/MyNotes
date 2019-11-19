# export PATH=/home/yule/anaconda3/bin:$PATH
import numpy as np
from numpy.random import seed

# Adaline算法分类器
class AdalineGD(object):
	
	def __init__(self,eta=0.01,m_iter=10,shuffle=True,random_state=None):
		self.eta = eta   #学习速率，浮点型，[0,1]
		self.m_iter = m_iter  #整型，passes over the training dataset，迭代次数
		self.w_initialized = False
		self.shuffle = shuffle  #洗牌，布尔型
		if random_state:	#征信，可用来指定随机数种子
			seed(random_state)

	# 使用单个训练样本更新权重
	def fit(self,X,y):
		#np.zeros():返回来一个给定形状和类型的用0填充的数组
		#形状：1 + the number of samples
		#self.w_ = np.zeros(1+X.shape[1]) #ld-array,weights after fitting
		self._initialize_weights(X.shape[1]) 	# 权重初始化为0
		self.cost_ = []  
		for i in range(self.m_iter):
			if self.shuffle:
				X,y = self._shuffle(X,y) #打乱顺序，还是向量
			cost = []
			for xi,target in zip(X,y):
				cost.append(self._update_weights(xi,target))
			avg_cost = sum(cost) / len(y)
			self.cost_.append(avg_cost)
		return self
	
	# 在线学习，不会重置权重
	def partial_fit(self,X,y):
		if not self.w_initialized:
			self._initialize_weights(X.shape[1])
		if y.ravel().shape[0] > 1:
			for xi,target in zip(X,y):
				self._update_weights(xi,target)
		else:
			self._update_weights(X,y)
		return self

	# 洗牌，重排数据避免在优化代价函数时陷入循环
	def _shuffle(self,X,y):
		r=np.random.permutation(len(y)) #生成一个不重复的随机序列？
		return X[r],y[r]
	
	# 权重初始化为0
	def _initialize_weights(self,m):
		self.w_ = np.zeros(1+m)
		self.w_initialized = True

	# 根据随机梯度下降的公式，更新权重
	def _update_weights(self,xi,target):
		output = self.net_input(xi)  # 1个数字
		error= (target - output) # 1个数字，target=[y0]
		self.w_[1:] += self.eta*xi.dot(error) #公式，之所以是点乘，是因为xi是一个二维的[x1,x2]
		self.w_[0] += self.eta*error
		cost = 0.5 * error**2
		return cost

	# 得到净输入z
	def net_input(self,X):
		#np.dot(A, B)：
		#对于二维矩阵，计算真正意义上的矩阵乘积，同线性代数中矩阵乘法的定义。
		return np.dot(X,self.w_[1:]) + self.w_[0]
		#np.dot(a,b)  等价于  sum(i*j for i,j in zip(a,b))
	
	# 输出激励函数 $$\phi (z) = z$$（一个恒等函数）
	def activation(self,X):
		return self.net_input(X)

	# 量化器，得到当前迭代下的输出y
	def predict(self,X):
		#numpy.where(condition[], x, y)三者的维度相同
		#当conditon的某个位置的为true时，输出x的对应位置的元素，否则选择y对应位置的元素；
		return np.where(self.activation(X)>=0.0, 1, -1)


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
	# numpy.ravel()返回的是视图（view，也颇有几分C/C++引用reference的意味），会影响（reflects）原始矩阵。
	Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T) #array.T：转秩
	Z = Z.reshape(xx1.shape)
	#使用contourf()画三维等高线图，对于网格数组中每个预测的类以不同的颜色绘制出决策区域
	plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
	plt.xlim(xx1.min(),xx1.max())
	plt.ylim(xx2.min(),xx2.max())
	#对于一个可迭代的/可遍历的对象（如列表、字符串），
	#enumerate将其组成一个索引序列，利用它可以同时获得索引和值
	for idx,cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)

# 先进行标准化操作
# 使数据具备标准正态分布的特性：特征值的均值为0，标准差为1
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = AdalineGD(m_iter=15, eta=0.01,random_state=1)
ada.fit(X_std, y)
plot_decision_regions(X_std,y,classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('sum-squared-error')
plt.show()

