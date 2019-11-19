# export PATH=/home/yule/anaconda3/bin:$PATH

from scipy.spatial.distance import pdist,squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt 


#实例：分离半月形数据
from sklearn.datasets import make_moons
X,y = make_moons(n_samples=100,random_state=123)

#使用scikit-learn
from sklearn.decomposition import KernelPCA
scikit_kpca = KernelPCA(n_components=2,kernel='rbf',gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

#可视化
plt.scatter(X_skernpca[y==0,0],X_skernpca[y==0,1],color='red',marker='^',alpha=0.5)
plt.scatter(X_skernpca[y==1,0],X_skernpca[y==1,1],color='blue',marker='o',alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
