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

pipe_lr = Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(penalty='l2',random_state=0))])

#ROC曲线
from sklearn.metrics import roc_curve,auc
# auc: area under the curve(用来刻画性能)
from scipy import interp
from sklearn.model_selection import StratifiedKFold
X_train2 = X_train[:,[4,14]] #只取部分特征
n_splits = 3 # 分块数量是3
cv = StratifiedKFold(n_splits=n_splits,random_state=1)
fig = plt.figure(figsize=(7,5))
#真正率：
mean_tpr = 0.0
#假正率：
mean_fpr = np.linspace(0,1,100)

i = 1
for train,test in cv.split(X_train, y_train):
	probas = pipe_lr.fit(X_train2[train],y_train[train]).predict_proba(X_train2[test])
	fpr,tpr,thresholds = roc_curve(y_train[test],probas[:,1],pos_label=1)
	mean_tpr += interp(mean_fpr,fpr,tpr)
	mean_tpr[0] = 0.0
	roc_auc = auc(fpr,tpr)
	plt.plot(fpr,tpr,lw=1,label='Roc fold %d area = %0.2f' % (i,roc_auc))
	i += 1

plt.plot([0,1],[0,1],linestyle='--',color=(0.6,0.6,0.6),label='random guessing')
mean_tpr /= n_splits #3应该是StratifiedKFold中的参数n_splits(fold=3)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr,mean_tpr)
plt.plot(mean_fpr,mean_tpr,'k--',label='mean ROC (area=%.2f)' % mean_auc,lw=2)
plt.plot([0,0,1],[0,1,1],lw=2,linestyle=':',color='black',label = 'perfect performance')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Opetator Characteristic')
plt.legend(loc='lower right')
plt.show()

#直接计算ROC AUC得分：
pipe_svc = Pipeline([('scl',StandardScaler()),('clf',SVC(random_state=1))])
pipe_svc = pipe_svc.fit(X_train2,y_train)
y_pred2 = pipe_svc.predict(X_test[:,[4,14]])
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
print('ROC AUC: %.3f' % roc_auc_score(y_true=y_test,y_score=y_pred2))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test,y_pred=y_pred2))
'''
ROC AUC: 0.671
Accuracy: 0.728
'''