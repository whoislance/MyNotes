# export PATH=/home/yule/anaconda3/bin:$PATH

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

df_wine = pd.read_csv('wine.data',header=None)

#print(df_wine.head())

# 划分出测试集和训练集
from sklearn.model_selection import train_test_split
X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#随机森林
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=-1) #10000棵决策树
forest.fit(X_train,y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
	print("%2d) %-*s %f" % (f+1,30,feat_labels[f],importances[indices[f]]))
'''
 1)1                              0.182169
 2)2                              0.158680
 3)3                              0.151389
 4)4                              0.132085
 5)5                              0.106648
 6)6                              0.078086
 7)7                              0.060561
 8)8                              0.032131
 9)9                              0.025390
10)10                             0.022380
11)11                             0.022035
12)12                             0.014670
13)13                             0.013774
'''
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),importances[indices],color='lightblue',align='center')
plt.xticks(range(X_train.shape[1]),feat_labels,rotation=90)
plt.xlim([-1,X_train.shape[1]])
plt.tight_layout()
plt.show()
