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
	#X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components + 1))) #选择前k个
	alphas = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1))) #和之前的X_pc一样
	lambdas = [eigvals[-i] for i in range(1,n_components+1)] #特征值？

	return alphas,lambdas  #n*k


def project_x(x_new,X,gamma,alphas,lambdas):
	pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
	k = np.exp(-gamma * pair_dist)
	return k.dot(alphas / lambdas) #归一化处理

#实例：分离半月形数据
from sklearn.datasets import make_moons
X,y = make_moons(n_samples=100,random_state=123)
alphas,lambdas = rbf_kernel_pac(X,gamma=15,n_components=1)
x_new=X[25]
x_proj = alphas[25]

x_reproj = project_x(x_new,X,gamma=15,alphas=alphas,lambdas=lambdas)

#将第一主成分上的映射进行可视化
plt.scatter(alphas[y==0,0],np.zeros((50)),color='red',marker='^',alpha=0.5)
plt.scatter(alphas[y==1,0],np.zeros((50)),color='blue',marker='o',alpha=0.5)
plt.scatter(x_proj,0,color='black',label='original projection of point X[25]',marker='^',s=100)
plt.scatter(x_reproj,0,color='green',label='remapped point X[25]',marker='x',s=500)
plt.legend(scatterpoints=1)
plt.show()
