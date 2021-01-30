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

#SBS算法
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
class SBS():
	def __init__(self,estimator,k_features,scoring=accuracy_score,test_size=0.25,random_state=1):
		self.scoring = scoring
		self.estimator = clone(estimator)
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state

	def fit(self,X,y):
		# 在函数内划分号测试集和训练集
		X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=self.test_size,random_state=self.random_state)
		dim = X_train.shape[1]	#当前维数
		self.indices_ = tuple(range(dim)) #指数
		self.subsets_ = [self.indices_]  #子集
		score = self._calc_score(X_train,y_train,X_test,y_test,self.indices_)
		self.scores_ = [score]
		# k_features指定需返回的特征数量
		while dim > self.k_features:
			scores = []
			subsets = []
			# combinations(iterable,r) 创建一个迭代器，
			#返回iterable中所有长度为r的子序列，返回的子序列中的项按输入iterable中的顺序排序
			for p in combinations(self.indices_,r=dim-1): #得到特征子集
				score = self._calc_score(X_train,y_train,X_test,y_test,p)
				scores.append(score)
				subsets.append(p)
			best = np.argmax(scores)
			self.indices_ = subsets[best] #最终特征子集的列标
			self.subsets_.append(self.indices_)
			dim -= 1
			self.scores_.append(scores[best]) #存储最优特征子集的准确度分值
		self.k_score_ = self.scores_[-1]
		return self

	# 返回由选定特征列构成的新数组
	def transfrom(self,X):
		return X[:,self.indices_]

	# 返回分值
	def _calc_score(self,X_train,y_train,X_test,y_test,indices):
		self.estimator.fit(X_train[:,indices],y_train)
		y_pred = self.estimator.predict(X_test[:,indices])
		score = self.scoring(y_test,y_pred)
		return score

#最近邻分类器
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn,k_features=1)
sbs.fit(X_train_std,y_train)

k_feat = [len(k) for k in sbs.subsets_] #得到1到13数字
plt.plot(k_feat,sbs.scores_,marker='o')
plt.ylim([0.75,1.05])
plt.ylabel('Accuracy')
plt.xlabel("Number of features")
plt.grid()
plt.show()

#从第9列获取五个特征子集的列标
k5 = list(sbs.subsets_[8]) #13-8=5，该列有5个特征
#显示表现最好的5个特征
print(df_wine.columns[1:][k5])
#Int64Index([1, 2, 4, 11, 13], dtype='int64')

#（一）使用所有的特征
knn.fit(X_train_std,y_train)
print('Training accuracy:',knn.score(X_train_std,y_train))
#Training accuracy: 0.983870967742
print('Test accuracy:',knn.score(X_test_std,y_test)) #存在过拟合
#Test accuracy: 0.944444444444

#（二）使用选定的五个特征
knn.fit(X_train_std[:,k5],y_train)
print('Training accuracy:',knn.score(X_train_std[:,k5],y_train))
#Training accuracy: 0.959677419355
print('Test accuracy:',knn.score(X_test_std[:,k5],y_test))
#Test accuracy: 0.962962962963
