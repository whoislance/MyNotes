# export PATH=/home/yule/anaconda3/bin:$PATH

#读取数据集
import pandas as pd 
df = pd.read_csv('wdbc.data',header=None) #[569 rows x 32 columns]

#将类标从原始的字符串表示(M/B)转换为整数
from sklearn.preprocessing import LabelEncoder
X = df.loc[:,2:].values #一共30个特征
y = df.loc[:,1].values #值为M/B
le = LabelEncoder()
y = le.fit_transform(y) #值为1/0

#划分数据集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=1)

#在流水线中集成数据转换及评估操作
from sklearn.preprocessing import StandardScaler #1.StandardScaler标准化处理
from sklearn.decomposition import PCA #2.PCA主成分分析
from sklearn.linear_model import LogisticRegression #3.LogisRegression分类器
from sklearn.pipeline import Pipeline 

#1.StandardScaler标准化处理；2.PCA主成分分析；3.LogisRegression分类器
pipe_lr = Pipeline([('scl',StandardScaler()),('pca',PCA(n_components=2)),('clf',LogisticRegression(random_state=1))])
pipe_lr.fit(X_train,y_train) #直接三步完成
#print('Test Accuracy: %.3f' % pipe_lr.score(X_test,y_test)) #输入测试集
# Test Accuracy: 0.947

#方法1：
#手写k折交叉验证
import numpy as np
from sklearn.model_selection import StratifiedKFold,KFold
#kfold = KFold(n_splits=10,random_state=1,shuffle=False)
#kfold = StratifiedKFold(y=y_train,n_folds=10,random_state=1)
#kfold = StratifiedKFold(n_splits=104,random_state=1,shuffle=False)
kfold =  StratifiedKFold(n_splits=10,random_state=1)
scores = []
for train,test in kfold.split(X_train, y_train):
	pipe_lr.fit(X_train[train],y_train[train])
	score = pipe_lr.score(X_train[test],y_train[test])
	scores.append(score)
	print('Class dist: %s,Acc: %.3f' % (np.bincount(y_train[train]),score))
'''
Class dist: [256 153],Acc: 0.891
Class dist: [256 153],Acc: 0.978
Class dist: [256 153],Acc: 0.978
Class dist: [256 153],Acc: 0.913
Class dist: [256 153],Acc: 0.935
Class dist: [257 153],Acc: 0.978
Class dist: [257 153],Acc: 0.933
Class dist: [257 153],Acc: 0.956
Class dist: [257 153],Acc: 0.978
Class dist: [257 153],Acc: 0.956
'''
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
#CV accuracy: 0.950 +/- 0.029


#方法2：调用API直接得到scores
#使用分层k折交叉验证对模型进行评估：
from sklearn.model_selection import cross_val_score
#使用API的好处是能将不同分块的评估分布到多个CPU上进行
scores = cross_val_score(estimator=pipe_lr,X=X_train,y=y_train,cv=10,n_jobs=1)
print('CV accuracy sores: %s' % scores)
#CV accuracy sores: [ 0.89130435  0.97826087  0.97826087  0.91304348  0.93478261  0.97777778
#  0.93333333  0.95555556  0.97777778  0.95555556]
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
#CV accuracy: 0.950 +/- 0.029