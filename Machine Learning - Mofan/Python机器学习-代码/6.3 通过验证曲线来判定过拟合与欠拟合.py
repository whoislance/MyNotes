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


#使用验证曲线函数评估模型
import matplotlib.pyplot as plt 
from sklearn.model_selection import validation_curve
param_range = [0.001,0.01,0.1,1.0,10.0,100.0]
pipe_lr = Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(penalty='l2',random_state=0))])
#得到训练准确率、验证准确率
train_scores,test_scores = validation_curve(estimator=pipe_lr,X=X_train,y=y_train,param_name='clf__C',param_range=param_range,cv=10)
train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)
plt.plot(param_range,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(param_range,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
#fill_between():加入了平均准确率标准差的信息，表示评价结果的方差
plt.plot(param_range,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='validation accuracy')
plt.fill_between(param_range,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green')
plt.grid()
plt.xscale('log')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,1.0])
plt.show()