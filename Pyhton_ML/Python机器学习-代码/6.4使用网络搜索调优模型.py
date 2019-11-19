# export PATH=/home/yule/anaconda3/bin:$PATH

#读取数据集
import pandas as pd 
import numpy as np
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


#方法1：使用网格搜索调优超参
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
pipe_svc = Pipeline([('scl',StandardScaler()),('clf',SVC(random_state=1))])
param_range = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
param_grid = [{'clf__C':param_range,'clf__kernel':['linear']},{'clf__C':param_range,'clf__gamma':param_range,'clf__kernel':['rbf']}]:
#字典中第一个是线性SVM（有一个参数C），第二个是基于RBF的核SVM（有两个参数C,gamma）
gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=10,n_jobs=-1)
gs = gs.fit(X_train,y_train)
print(gs.best_score_)
#0.978021978021978
print(gs.best_params_)
#{'clf__C': 0.1, 'clf__kernel': 'linear'}
#也就是受，C=0.1时得到最优准确率97.8%（验证集）

#使用best_estimator_属性对最优模型进行评估
clf = gs.best_estimator_
clf.fit(X_train,y_train)
print('Test accuracy: %.3f' % clf.score(X_test,y_test)) #这里使用独立的测试集
#Test accuracy: 0.965


#方法2：借助scikit-learn，使用嵌套交叉验证
from sklearn.model_selection import cross_val_score
gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=10,n_jobs=-1)
scores = cross_val_score(gs,X,y,scoring='accuracy',cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
# CV accuracy: 0.972 +/- 0.012