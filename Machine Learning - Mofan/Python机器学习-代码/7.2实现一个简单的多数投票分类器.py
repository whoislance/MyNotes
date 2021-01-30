# export PATH=/home/yule/anaconda3/bin:$PATH

import matplotlib.pyplot as plt 
from scipy.misc import comb 
import math 
import numpy as np 
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six #为了使该类与python2.7兼容
from sklearn.pipeline import _name_estimators
import operator

#该类在 sklearn.ensemble.VotingClassifier类中
class MajorityVoteClassifier(BaseEstimator,ClassifierMixin):
	def __init__(self,classifiers,vote='classlabel',weights=None):
		self.classifiers = classifiers #数组：[n_classifiers]
		self.named_classifiers = {key:value for key,value in _name_estimators(classifiers)}
		self.vote = vote  #字符串：{'类标','概率'}，二选一
		self.weights = weights #类似数组：[n_classifiers]

	#预处理了类标，fit每一个分类器
	def fit(self,X,y):
		#X：形式[n_samples,n_samples].{类似数组，稀疏矩阵}
		#y：形式[n_samples]，类标向量
		self.lablenc_ = LabelEncoder() #确保类标从0开始，LabelEncoder是对不连续的数字或者文本进行编号
		self.lablenc_.fit(y)
		self.classes_ = self.lablenc_.classes_ #这里结果等于[1 2]
		self.classifiers_ = []
		for clf in self.classifiers:
			fitted_clf = clone(clf).fit(X,self.lablenc_.transform(y)) #这里才是从0开始标号
			#self.lablenc_.transform(y)=[0 ... 0 1 ...1]
			self.classifiers_.append(fitted_clf)
		return self

	#返回预测的类标
	def predict(self,X):
		if self.vote == 'probability':  #基于类别成员的概率进行类标预测
			maj_vote = np.argmax(self.predict_proba(X),axis=1)
		else:   #否则vote='classlabel'，基于多数投票来预测类标
			predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T 
			#numpy.apply_along_axis(func, axis, arr, *args, **kwargs)：
			#其中func是我们自定义的一个函数，函数func(arr)中的arr是一个数组，
			#函数的主要功能就是对数组里的每一个元素进行变换，得到目标的结果。
			maj_vote = np.apply_along_axis(lambda x:np.argmax(np.bincount(x,weights=self.weights)),axis=1,arr=predictions)
		maj_vote = self.lablenc_.inverse_transform(maj_vote) #进行编号
		return maj_vote

	#返回平均概率
	def predict_proba(self,X):  
		probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
		avg_probas = np.average(probas,axis=0,weights=self.weights)
		return avg_probas #shape=[n_samples,n_features]

	#为了使用网格搜索来调整超参，获得分类器参数
	#方便使用_name_estimators函数来访问集成分类器中独立成员函数的参数？
	def get_params(self,deep=True):
		if not deep:
			return super(MajorityVoteClassifier,self).get_params(deep=False)
		else:
			out = self.named_classifiers.copy()
			for name,step in six.iteritems(self,named_classifiers):
				for key,value in six.iteritems(step.get_params(deep=True)):
					out['%s__%s' %(name,key)] = value
		return out


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X,y = iris.data[50:,[1,2]],iris.target[50:]
'''
y = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
'''
le = LabelEncoder()
y = le.fit_transform(y)
'''
y = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
'''
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=1)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
#c1：逻辑斯底回归分类器
clf1 = LogisticRegression(penalty='l2',C=0.001,random_state=0)
#c2：决策树分类器
clf2 = DecisionTreeClassifier(max_depth=1,criterion='entropy',random_state=0)
#c3：K-邻近分类器
clf3 = KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski')
#p1：逻辑斯底回归分类器
pipe1 = Pipeline([['sc',StandardScaler()],['clf',clf1]])
#p3：K-邻近分类器
pipe3 = Pipeline([['sc',StandardScaler()],['clf',clf3]])

#给每个独立分类器打分
clf_labels = ['Logistic Regression','Decision Tree','KNN']
print('10-fold cross validation:\n')
for clf,label in zip([pipe1,clf2,pipe3],clf_labels):
	scores = cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10,scoring='roc_auc')
	print("ROC AUC:%.2f (+/- %.2f) [%s]" % (scores.mean(),scores.std(),label))	

'''
10-fold cross validation:

ROC AUC:0.92 (+/- 0.20) [Logistic Regression]
ROC AUC:0.92 (+/- 0.15) [Decision Tree]
ROC AUC:0.93 (+/- 0.10) [KNN]
'''

#给组合分类器打分
mv_clf = MajorityVoteClassifier(classifiers=[pipe1,clf1,pipe3])
mv_labels = 'Majority Voting'
scores = cross_val_score(estimator=mv_clf,X=X_train,y=y_train,cv=10,scoring='roc_auc')
print("Accuracy: %.2f (+/- %.2f) [%s]" % (scores.mean(),scores.std(),mv_labels))
'''
Accuracy: 0.92 (+/- 0.20) [Logistic Regression]
Accuracy: 0.92 (+/- 0.15) [Decision Tree]
Accuracy: 0.93 (+/- 0.10) [KNN]
Accuracy: 0.97 (+/- 0.10) [Majority Voting]
'''	
