import pandas as pd
import numpy as np



print('----------ch.1------------------')
s = pd.Series([1,3,6,np.nan,44,1])
print('s:\n',s)
dates = pd.date_range('20180416',periods=6)
print('dates:\n',dates)
df = pd.DataFrame(np.arange(12).reshape((3,4)))
print('df:\n',df)
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
print('df:\n',df)
print('df.dtypes:\n',df.dtypes)
print('df.index:\n',df.index)
print('df.columns:\n',df.columns)
print('df.values:\n',df.values)
print('df.describe():\n',df.describe())
print('df.sort_index:\n',df.dtypes)
print('df.sort_index1:\n',df.sort_index(axis=1,ascending=False))
print('df.sort_index2:\n',df.sort_index(axis=0,ascending=False))
print('df.sort_values:\n',df.sort_values(by='a'))
print('\n')

print('----------ch.2------------------')
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])
print('df:\n',df)
print('column=A:\n',df['A'])
print('column=A:\n',df.A)
print('index:0-3\n',df[0:3])
print('index:0-3\n',df['20180416':'20180418'])
print('select by label:loc\n',df.loc['20180421'])
print('select A and D:\n',df.loc[:,['A','D']])
print('select A and D where index is 0421:\n',df.loc['20180421',['A','D']])
print('select by position:iloc\n',df.iloc[5,[0,3]])
print('mixed selection:ix:\n',df.ix[2:4,['A','D']])
print('df[df.A<9]:\n',df[df.A<9])
print('\n')

print('----------ch.3------------------')
df2 = df.copy()
df2[df2.A>10] = 0
print('df2:\n',df2)
df3 = df.copy()
df3.B[df3.A>10] = 0
print('df3:\n',df3)
df['E'] = pd.Series([99,99,99,99,99,99],index=dates)
df['F'] = np.nan
print('df:\n',df)
print('\n')

print('----------ch.4------------------')
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
print('df:\n',df)
print('判断每个元素是否存在NAN:\n',df.isnull())
print('判断整个矩阵是否存在NAN:\n',np.any(df.isnull()==True))
print('直接丢掉有NAN的数据行/列：\n',df.dropna(axis=1,how='any'))
print('填入值给NAN:\n',df.fillna(value=0))
print('\n')

print('----------ch.5------------------')
student_data = r"C:\Users\YuL_e\Desktop\tf-test\numpy&pandas\student.csv"
data = pd.read_csv(student_data)
print(data)

print('\n')

print('----------ch.6------------------')
#定义资料集
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])
print('df1:\n',df1,'\ndf2:\n',df2,'\ndf3\n',df3)

#concat
res = pd.concat([df1,df2,df3],axis=0)
print('res_h:\n',res)
res = pd.concat([df1,df2,df3],axis=1)
print('res_v:\n',res)
res = pd.concat([df1,df2,df3],axis=0,ignore_index=True)
print('res_ignore_index:\n',res)

# join (合并方式)
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])
print('df1:\n',df1,'\ndf2:\n',df2)

res = pd.concat([df1,df2],join='inner',ignore_index=True)
print('inner:\n',res)
res = pd.concat([df1,df2],join='outer',ignore_index=True)
print('outer:\n',res)
res = pd.concat([df1,df2],axis=1,join_axes=[df1.index])
print('使用某个df的index处理：\n',res)


#append：只有纵向合并，没有横向合并。
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
res = df1.append([df2,df3], ignore_index=True)
print('df1 + df2 + df3:\n',res)
s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])
res = df1.append(s1,ignore_index=True)
print('df1 + s1:\n',res)

print('\n')

print('----------ch.7------------------')
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                             'A': ['A0', 'A1', 'A2', 'A3'],
                             'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                              'C': ['C0', 'C1', 'C2', 'C3'],
                              'D': ['D0', 'D1', 'D2', 'D3']})
print('left:\n',left)
print('right:\n',right)
res = pd.merge(left, right, on='key')
print('res:\n',res)

# 依据两组key合并
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                      'key2': ['K0', 'K1', 'K0', 'K1'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                       'key2': ['K0', 'K0', 'K0', 'K0'],
                       'C': ['C0', 'C1', 'C2', 'C3'],
                       'D': ['D0', 'D1', 'D2', 'D3']})
print('left:\n',left)
print('right:\n',right)
res = pd.merge(left, right, on=['key1', 'key2'], how='inner')
print('res_inner:\n',res)
res = pd.merge(left, right, on=['key1', 'key2'], how='outer')
print('res_outer:\n',res)
res = pd.merge(left, right, on=['key1', 'key2'], how='left')
print('res_left:\n',res)
res = pd.merge(left, right, on=['key1', 'key2'], how='right')
print('res_right:\n',res)

# Indicator 
df1 = pd.DataFrame({'col1':[0,1], 'col_left':['a','b']})
df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})
print('df1:\n',df1)
print('df2:\n',df2)
res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
print('res_indicator:\n',res)
res = pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_column')
print('res_indicator_column:\n',res)

print('\n')

print('----------ch.8------------------')
import matplotlib.pyplot as plt 
#plot data

#Series
data = pd.Series(np.random.randn(1000),index=np.arange(1000))
data = data.cumsum()
#data.plot()
#plt.show()

# DataFrame
data = pd.DataFrame(np.random.randn(1000,4),index=np.arange(1000),columns=list("ABCD"))
data = data.cumsum()
#data.plot()
#plt.show()

# plot methods:
# 'bar','hist','box','kde','area','scatter','hexbin','pie'
ax = data.plot.scatter(x='A',y='B',color='DarkBlue',label='Class 1')
data.plot.scatter(x='A',y='C',color='DarkGreen',label='Class 2',ax=ax)
#plt.scatter(x=data['A'],y=data['B'])
plt.show()