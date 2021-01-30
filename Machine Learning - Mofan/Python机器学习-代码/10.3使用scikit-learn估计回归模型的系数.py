# PATH=/home/yule/anaconda3/bin:$PATH
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('housing.data',header=None,sep='\s+')
df.columns=[
	'CRIM','ZN','INDUS','CHAS',
	'NOX','RM','AGE','DIS','RAD',
	'TAX','PTRATIO','B','LSTAT','MEDV'
]
X = df[['RM']].values
y = df['MEDV'].values


from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X,y)
print('Slope:%.3f' % slr.coef_[0])
print('Intercept:%.3f' % slr.intercept_)
'''
Slope:9.102
Intercept:-34.671
'''

lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.tight_layout()
# plt.savefig('./figures/scikit_lr_fit.png', dpi=300)
plt.show()