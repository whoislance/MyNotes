# export PATH=/home/yule/anaconda3/bin:$PATH

import matplotlib.pyplot as plt 

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
from sklearn.svm import SVC

pipe_svc = Pipeline([('scl',StandardScaler()),('clf',SVC(random_state=1))])

from sklearn.metrics import confusion_matrix
pipe_svc.fit(X_train,y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test,y_pred=y_pred)
print(confmat)
'''
[[71  1]
 [ 2 40]]
'''
fig,ax = plt.subplots(figsize=(2.5,2.5))
ax.matshow(confmat,cmap=plt.cm.Blues,alpha=0.3)
for i in range(confmat.shape[0]):
	for j in range(confmat.shape[1]):
		ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score,f1_score 
print('Precision: %.3f' % precision_score(y_true=y_test,y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test,y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test,y_pred=y_pred))
'''
Precision: 0.976
Recall: 0.952
F1: 0.96
'''