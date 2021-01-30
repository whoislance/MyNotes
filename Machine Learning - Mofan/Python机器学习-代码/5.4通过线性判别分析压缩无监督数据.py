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

#计算均值向量
np.set_printoptions(precision=4) #决定输出数字的位数
mean_vecs = []
for label in range(1,4):
	mean_vecs.append(np.mean(X_train_std[y_train==label],axis=0)) #axis=0：对各列求均值，返回1*n
	print('MV %s: %s\n' % (label,mean_vecs[label-1]))

'''
MV 1: [ 0.9259 -0.3091  0.2592 -0.7989  0.3039  0.9608  1.0515 -0.6306  0.5354
  0.2209  0.4855  0.798   1.2017]

MV 2: [-0.8727 -0.3854 -0.4437  0.2481 -0.2409 -0.1059  0.0187 -0.0164  0.1095
 -0.8796  0.4392  0.2776 -0.7016]

MV 3: [ 0.1637  0.8929  0.3249  0.5658 -0.01   -0.9499 -1.228   0.7436 -0.7652
  0.979  -1.1698 -1.3007 -0.3912]
'''

#计算类内散布矩阵S_W
d = 13 #特征数
S_W = np.zeros((d,d))
for label,mv in zip(range(1,4),mean_vecs):
	#协方差矩阵
	class_scatter = np.cov(X_train_std[y_train==label].T) 
	S_W += class_scatter
	'''#未归一化
	class_scatter = np.zeros((d,d)) 
	for row in X[y==label]:
		row,mv = row.reshape(d,1),mv.reshape(d,1)
		class_scatter += (row-mv).dot((row-mv).T)
	S_W += class_scatter
	'''
print('Within-class scatter matrix:%sx%s' % (S_W.shape[0],S_W.shape[1]))
print('Class label distribution: %s' % np.bincount(y_train.astype(np.int32))[1:])
#数出每个值依次在数组中出现的次数，这里是1,2,3出现的次数
'''
Within-class scatter matrix:13x13
Class label distribution: [40 49 35] #说明训练的类标没有均匀分布
'''

#计算类间散布矩阵
mean_overall = np.mean(X_train_std,axis=0)  #1*13
d = 13
S_B = np.zeros((d,d))
for i,mean_vec in enumerate(mean_vecs): #enumetate：同时获得索引和值
	n = X[y==i+1,:].shape[0]
	#n依次等于59,71,48.一共178个总样本
	mean_vec = mean_vec.reshape(d,1) #13*1
	mean_overall = mean_overall.reshape(d,1) #13*1
	S_B += n * (mean_vec-mean_overall).dot((mean_vec-mean_overall).T) #13*13
print('Between-class scatter matrix:%sx%s' % (S_B.shape[0],S_B.shape[1])) #S_B.shape = (13,13)
'''
Between-class scatter matrix:13x13
'''


#求解广义特征值(特征值：eigenvalue)
eigen_vals,eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B)) #对应第四步
#对特征值排序：特征对（特征值-特征向量）
eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))] 
eigen_pairs = sorted(eigen_pairs,key=lambda k: k[0], reverse=True) #key:用来比较的元素
print('Eigenvalues in decreasing order:\n')
for eigen_val in eigen_pairs:
	print(eigen_val[0])

#绘制特征对线性判别信息保持程度的图像
tot = sum(eigen_vals.real) #实数部分相加
discr = [(i/tot) for i in sorted(eigen_vals.real,reverse=True)] #计算出单个的区分能力
cum_discr = np.cumsum(discr) #返回累加的和
plt.bar(range(1,14),discr,alpha=0.5,align='center',label='individual "discriminability"')
plt.step(range(1,14),cum_discr,where='mid',label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1,1.1])
plt.legend(loc='best') #best指自适应方式
plt.show()

#得到转换矩阵（这里只有前两行中的特征值是非零特征值）
w = np.hstack((eigen_pairs[0][1][:,np.newaxis].real,eigen_pairs[1][1][:,np.newaxis].real))
#np.newaxis:插入新的维度
print('Matrix W:\n',w) #w:13*2

#将样本映射到新的特征空间
X_train_lda = X_train_std.dot(w)
colors = ['r','b','g']
markers = ['s','x','o']
for l,c,m in zip(np.unique(y_train),colors,markers):
	plt.scatter(X_train_lda[y_train==l,0],X_train_lda[y_train==l,1],c=c,label=l,marker=m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='upper right')
plt.show()