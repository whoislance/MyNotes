# export PATH=/home/yule/anaconda3/bin:$PATH

from scipy.spatial.distance import pdist,squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt 

def rbf_kernel_pac(X,gamma,n_components):
	#X：n*n features
	#gamma：公式中的系数伽马

	#1.计算核相似矩阵（输入是原始空间）
	sq_dists = pdist(X,'sqeuclidean') #计算两辆之间的欧式距离：|x-y|^2
	mat_sq_dists = squareform(sq_dists) #将M×N距离矩阵转换到方阵中
	K = exp(- gamma * mat_sq_dists) #计算对称核矩阵K

	#2.把新特征空间的中心放在零点
	N = K.shape[0]
	one_n = np.ones((N,N)) / N #所有值为1/n
	K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n) #聚集核矩阵得到新核矩阵

	#3.得到特征值和特征向量
	eigvals,eigvecs = eigh(K) #n*n
	X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components + 1))) #选择前k个
	return X_pc  #n*k

#实例1：分离半月形数据
from sklearn.datasets import make_moons
X,y = make_moons(n_samples=100,random_state=123)
plt.scatter(X[y==0,0],X[y==0,1],color='red',marker='^',alpha=0.5)
plt.scatter(X[y==1,0],X[y==1,1],color='blue',marker='o',alpha=0.5)
plt.show()

from matplotlib.ticker import FormatStrFormatter
X_kpca = rbf_kernel_pac(X,gamma=15,n_components=2)
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(7,3))
ax[0].scatter(X_kpca[y==0,0],X_kpca[y==0,1],color='red',marker='^',alpha=0.5)
ax[0].scatter(X_kpca[y==1,0],X_kpca[y==1,1],color='blue',marker='o',alpha=0.5)
ax[1].scatter(X_kpca[y==0,0],np.zeros((50,1)),color='red',marker='^',alpha=0.5)
ax[1].scatter(X_kpca[y==1,0],np.zeros((50,1)),color='blue',marker='o',alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_xlabel('PC2')
ax[1].set_ylim([-1,1])
#ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
plt.show()

##实例2:分离同心圆（代码基本一样，gamma=15）