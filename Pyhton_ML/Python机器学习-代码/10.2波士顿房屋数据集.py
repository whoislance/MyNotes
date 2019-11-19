# PATH=/home/yule/anaconda3/bin:$PATH
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('housing.data',header=None,sep='\s+')
df.columns=[
	'CRIM','ZN','INDUS','CHAS',
	'NOX','RM','AGE','DIS','RAD',
	'TAX','PTRATIO','B','LSTAT','MEDV'
]

#print(df.head())

#绘制散点图矩阵
'''
sns.set(style='whitegrid',context='notebook')
cols = ['CHAS','ZN','NOX','RM','MEDV']
sns.pairplot(df[cols],size=1.5)
plt.show()
'''

#绘制热度图
cols = ['LSTAT','INDUS','NOX','RM','MEDV']
cm  = np.corrcoef(df[cols].values.T) #九三相关系数矩阵
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':15},yticklabels=cols,xticklabels=cols)
plt.show()