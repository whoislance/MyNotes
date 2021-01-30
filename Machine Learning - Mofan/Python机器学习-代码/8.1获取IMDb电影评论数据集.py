# export PATH=/home/yule/anaconda3/bin:$PATH

#Aim:将各文本文档组合为一个CSV文件
import pyprind  #python progress indicator
import pandas as pd 
import numpy as np
import os
pbar = pyprind.ProgBar(50000) #一个进度条，包含5万次迭代
labels = {'pos':1,'neg':0}
df = pd.DataFrame()
for s in ('test','train'):
	for l  in ('pos','neg'):
		path = './aclImdb/%s/%s' % (s,l)
		for file in os.listdir(path):
			with open(os.path.join(path,file),'r') as infile:
				txt = infile.read()
			df = df.append([[txt,labels[l]]],ignore_index=True) #加入文档对应类标
			pbar.update()
df.columns = ['review','sentiment']

np.random.seed(0) #重排
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv',index=False)

df = pd.read_csv('./movie_data.csv')
print(df.head(3))
