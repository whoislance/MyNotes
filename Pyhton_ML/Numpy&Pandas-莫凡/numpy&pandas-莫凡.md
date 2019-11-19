# Numpy学习

### 几种属性

```python
import numpy as np 
array = np.array([
	[1,2,3],
	[2,3,4]
])
print(array)
print('number of dim:',array.ndim)
print('shape:',array.shape)
print('size:',array.size)
```

### 创建array

- 关键字
- 创建数组
- 指定数据dtype
- 创建特定数据

```
a = np.array([2,23,4],dtype=np.float)
print('a\'s tpye:',a.dtype)
b = np.ones( (3,4))
print('ones:',b)
c = np.zeros( (3,4))
print('zeros:',c)
d = np.empty( (3,4) )
print('empty:',d)
e = np.arange(10,20,2)
print('arange from 10 to 20, step is 2:',e)
f = np.arange(12).reshape( (3,4) )
print('reshape 3*4:',f)
g = np.linspace(1,10,19)
print('linspace:',g)
print('\n')
```

输出：

> a's tpye: float64
> ones: [[1. 1. 1. 1.]
>  [1. 1. 1. 1.]
>  [1. 1. 1. 1.]]
> zeros: [[0. 0. 0. 0.]
>  [0. 0. 0. 0.]
>  [0. 0. 0. 0.]]
> empty: [[0. 0. 0. 0.]
> 	      [0. 0. 0. 0.]
> 	      [0. 0. 0. 0.]]
> arange from 10 to 20, step is 2: [10 12 14 16 18]
> reshape 3*4: [[ 0  1  2  3]
> 			 [ 4  5  6  7]
> 			 [ 8  9 10 11]]
> linspace: [ 1.   1.5  2.   2.5  3. 3.5  4.   4.5  5.   5.5  6.   6.5  7.   7.5 8.5  9.   9.5 10. ]



### 基础运算一

```
a = np.array([10,20,30,40])
b = np.arange(4)
print('a and b:',a,b)
print('a - b:',a - b)
print('b^2:',b**2)
print('10*np.sin(a):',10*np.sin(a))
print('a<25?:',a < 25)
print('b=2?:',b == 2)
c = np.array([
	[0,1],
	[2,3]
])
d  = np.arange(4).reshape((2,2))
print('c and d:',c,d)
print('c*d:',c*d)
print('dot(c,d):',np.dot(c,d))
print('c.dot(d):',c.dot(d))
e = np.random.random((2,4))
print('random',e)

print('sum:',np.sum(e))
print('sum(axis=1):',np.sum(e,axis=1))
print('sum(axis=0):',np.sum(e,axis=0))
print('min',np.min(e))
print('max',np.max(e))
```

输出：

> a and b: [10 20 30 40][0 1 2 3]
> a - b: [10 19 28 37]
> b^2: [0 1 4 9]
> 10*np.sin(a): [-5.44021111  9.12945251 -9.88031624  7.4511316 ]
> a<25?: [ True  True False False]
> b=2?: [False False  True False]
> c and d: [[0 1]
>  		[2 3]]
>
> ​		[[0 1]
> 		 [2 3]]
> c*d: [[0 1]
> 	 [4 9]]
> dot(c,d): [[ 2  3]
> 		 [ 6 11]]
> c.dot(d): [[ 2  3]
> 		 [ 6 11]]
> random: [[0.19604817 0.13546968 0.95992382 0.0788865 ]
> 		 [0.29497431 0.5686708  0.75586072 0.25264342]]
> sum: 3.2424774090576403
> sum(1): [1.37032816 1.87214925]
> sum(0): [0.49102248 0.70414048 1.71578453 0.33152992]
> min 0.07888649705748918
> max 0.959923816719641

### 基础运算二

```python
A = np.arange(0,12).reshape( (3,4) )
print('A:',A)
print('arg_min:',np.argmin(A))
print('arg_max:',np.argmax(A))
print('median_of_A::',np.median(A))
print('mean_of_A::',np.mean(A))
print('mean_of_A::',A.mean())
print('average_of_A::',np.average(A))
print('cumsum_of_A::',np.cumsum(A))
print('diff_of_A::',np.diff(A))
print('nonzero_of_A::',np.nonzero(A))  #输出非零元素的横纵坐标
print('diff_of_A::',np.diff(A))
print('sort:',np.sort(A))
print('transpose_of_A::',np.transpose(A))
print('transpose_of_A::',A.T)
print('clip_of_A::',np.clip(A,5,9))
```

输出：

> A: [[ 0  1  2  3]
>  [ 4  5  6  7]
>  [ 8  9 10 11]]
> arg_min: 0
> arg_max: 11
> median_of_A:: 5.5
> mean_of_A:: 5.5
> mean_of_A:: 5.5
> average_of_A:: 5.5
> cumsum_of_A:: [ 0  1  3  6 10 15 21 28 36 45 55 66]
> diff_of_A:: [[1 1 1]
> 		   [1 1 1]
> 		   [1 1 1]]
> nonzero_of_A:: (array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int64), 
>
> ​			    array([1, 2, 3, 0,1, 2, 3, 0, 1, 2, 3], dtype=int64))
> diff_of_A:: [[1 1 1]
> 		 [1 1 1]
> 		 [1 1 1]]
> sort: [[ 0  1  2  3]
> 	 [ 4  5  6  7]
> 	 [ 8  9 10 11]]
> transpose_of_A:: [[ 0  4  8]
> 			     [ 1  5  9]
> 			     [ 2  6 10]
>  			     [ 3  7 11]]
> transpose_of_A:: [[ 0  4  8]
> 			    [ 1  5  9]
> 			    [ 2  6 10]
> 			    [ 3  7 11]]
> clip_of_A:: [[5 5 5 5]
>  		   [5 5 6 7]
>  		   [8 9 9 9]]

### 索引

```python
A = np.arange(3,15).reshape( (3,4) )
print('A:',A)
print('A[2,2]:',A[2,2])
print('A[2]:',A[2])
print('A[:,2]:',A[:,2])
print('flatten:',A.flatten())
for row in A:
	print('every row:',row)
for column in A.T:
	print('every column',column)
for item in A.flat:
	print('every item',item)
```

输出：

> A: [[ 3  4  5  6]
>  [ 7  8  9 10]
>  [11 12 13 14]]
> A[2,2]: 13
> A[2]: [11 12 13 14]
> A[:,2]: [ 5  9 13]
> flattern: [ 3  4  5  6  7  8  9 10 11 12 13 14]
> every row: [3 4 5 6]
> every row: [ 7  8  9 10]
> every row: [11 12 13 14]
> every column [ 3  7 11]
> every column [ 4  8 12]
> every column [ 5  9 13]
> every column [ 6 10 14]
> every item 3
> every item 4
> every item 5
> every item 6
> every item 7
> every item 8
> every item 9
> every item 10
> every item 11
> every item 12
> every item 13
> every item 14

### 合并

```python
A = np.array([1,1,1])
B = np.array([2,2,2])
print('vertical stack:',np.vstack((A,B)))
print('horizontal stack:',np.hstack((A,B)))
#print(A.T)
print('newaxis1:',A[np.newaxis,:])
print('newaxis2:',A[:,np.newaxis])
print('mul_stack1:',np.concatenate((A,B,B,A)))
```

输出：

> vertical stack: [[1 1 1]
>  [2 2 2]]
> horizontal stack: [1 1 1 2 2 2]
> newaxis1: [[1 1 1]]
> newaxis2: [[1]
>  [1]
>  [1]]
> mul_stack1: [1 1 1 2 2 2 2 2 2 1 1 1]

### 分割

```python
A = np.arange(12).reshape( (3,4) )
print('A:',A)
print('split:',np.split(A,2,axis=1))
print('split:',np.array_split(A,3,axis=1))
print('split:',np.hsplit(A,2))
```

输出：

> A: [[ 0  1  2  3]
>  [ 4  5  6  7]
>  [ 8  9 10 11]]
> split: [array([[0, 1],
>        [4, 5],
>        [8, 9]]), array([[ 2,  3],
>        [ 6,  7],
>        [10, 11]])]
> split: [array([[0, 1],
>        [4, 5],
>        [8, 9]]), array([[ 2],
>        [ 6],
>        [10]]), array([[ 3],
>        [ 7],
>        [11]])]
> split: [array([[0, 1],
>        [4, 5],
>        [8, 9]]), array([[ 2,  3],
>        [ 6,  7],
>        [10, 11]])]

### 复制

```python
a = np.arange(4)
print('a:',a)
b = a
c = a.copy()
print('b:',b)
print('c:',c)
a[0] = 99
print('new a:',a)
print('new b:',b)
print('new c(deep copy):',c)
```

> a: [0 1 2 3]
> b: [0 1 2 3]
> c: [0 1 2 3]
> new a: [99  1  2  3]
> new b: [99  1  2  3]
> new c(deep copy): [0 1 2 3]

# Pandas

### 基本操作

```python
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
```

输出：

> s:
> 0     1.0
> 1     3.0
> 2     6.0
> 3     NaN
> 4    44.0
> 5     1.0
>
> dtype: float64
>
> dates:
>  DatetimeIndex(['2018-04-16', '2018-04-17', '2018-04-18', '2018-04-19','2018-04-20', '2018-04-21'],
>               dtype='datetime64[ns]', freq='D')
>
> df:
>     0  1   2   3
> 0  0  1   2   3
> 1  4  5   6   7
> 2  8  9  10  11
>
> df:
>                     a         b         c         d
> 2018-04-16 -1.545180 -1.315450  0.893666 -2.023726
> 2018-04-17 -0.295566 -0.072707  0.549131  1.041857
> 2018-04-18 -0.701071 -0.157813 -0.093699  1.879216
> 2018-04-19 -0.398786 -1.693589 -0.664546  0.019741
> 2018-04-20 -1.290263  0.740058  1.491312  0.280536
> 2018-04-21 -0.037868 -1.459952  0.321975 -0.835499
>
> df.dtypes:
>  a    float64
> b    float64
> c    float64
> d    float64
>
> dtype: object
>
> df.index:
>  DatetimeIndex(['2018-04-16', '2018-04-17', '2018-04-18', '2018-04-19',
>                '2018-04-20', '2018-04-21'],
>               dtype='datetime64[ns]', freq='D')
>
> df.columns:
>  Index(['a', 'b', 'c', 'd'], dtype='object')
>
> df.values:
>  [[-1.54517955 -1.31544975  0.89366609 -2.02372617]
>  [-0.29556643 -0.07270714  0.54913061  1.04185654]
>  [-0.70107059 -0.15781304 -0.09369912  1.87921592]
>  [-0.39878564 -1.6935889  -0.66454624  0.01974063]
>  [-1.29026335  0.74005826  1.49131217  0.28053564]
>  [-0.03786787 -1.45995231  0.3219754  -0.83549916]]
>
> df.describe():
>                a         b         c         d
> count  6.000000  6.000000  6.000000  6.000000
> mean  -0.711456 -0.659909  0.416306  0.060354
> std    0.592397  0.969032  0.753445  1.376058
> min   -1.545180 -1.693589 -0.664546 -2.023726
> 25%   -1.142965 -1.423827  0.010220 -0.621689
> 50%   -0.549928 -0.736631  0.435553  0.150138
> 75%   -0.321371 -0.093984  0.807532  0.851526
> max   -0.037868  0.740058  1.491312  1.879216
>
> df.sort_index:
>  a    float64
> b    float64
> c    float64
> d    float64
>
> dtype: object
>
> df.sort_index1:
>                     d         c         b         a
> 2018-04-16 -2.023726  0.893666 -1.315450 -1.545180
> 2018-04-17  1.041857  0.549131 -0.072707 -0.295566
> 2018-04-18  1.879216 -0.093699 -0.157813 -0.701071
> 2018-04-19  0.019741 -0.664546 -1.693589 -0.398786
> 2018-04-20  0.280536  1.491312  0.740058 -1.290263
> 2018-04-21 -0.835499  0.321975 -1.459952 -0.037868
>
> df.sort_index2:
>                     a         b         c         d
> 2018-04-21 -0.037868 -1.459952  0.321975 -0.835499
> 2018-04-20 -1.290263  0.740058  1.491312  0.280536
> 2018-04-19 -0.398786 -1.693589 -0.664546  0.019741
> 2018-04-18 -0.701071 -0.157813 -0.093699  1.879216
> 2018-04-17 -0.295566 -0.072707  0.549131  1.041857
> 2018-04-16 -1.545180 -1.315450  0.893666 -2.023726
>
> df.sort_values:
>                     a         b         c         d
> 2018-04-16 -1.545180 -1.315450  0.893666 -2.023726
> 2018-04-20 -1.290263  0.740058  1.491312  0.280536
> 2018-04-18 -0.701071 -0.157813 -0.093699  1.879216
> 2018-04-19 -0.398786 -1.693589 -0.664546  0.019741
> 2018-04-17 -0.295566 -0.072707  0.549131  1.041857
> 2018-04-21 -0.037868 -1.459952  0.321975 -0.835499

### 选择数据

```python
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
```

输出：

> df:
>               A   B   C   D
> 2018-04-16   0   1   2   3
> 2018-04-17   4   5   6   7
> 2018-04-18   8   9  10  11
> 2018-04-19  12  13  14  15
> 2018-04-20  16  17  18  19
> 2018-04-21  20  21  22  23
>
> column=A:
>  2018-04-16     0
> 2018-04-17     4
> 2018-04-18     8
> 2018-04-19    12
> 2018-04-20    16
> 2018-04-21    20
>
> Freq: D, Name: A, dtype: int32
>
> column=A:
>  2018-04-16     0
> 2018-04-17     4
> 2018-04-18     8
> 2018-04-19    12
> 2018-04-20    16
> 2018-04-21    20
>
> Freq: D, Name: A, dtype: int32
>
> index:0-3
>              A  B   C   D
> 2018-04-16  0  1   2   3
> 2018-04-17  4  5   6   7
> 2018-04-18  8  9  10  11
>
> index:0-3
>              A  B   C   D
> 2018-04-16  0  1   2   3
> 2018-04-17  4  5   6   7
> 2018-04-18  8  9  10  11
>
> select by label:loc
>  A    20
> B    21
> C    22
> D    23
> Name: 2018-04-21 00:00:00, dtype: int32
>
> select A and D:
>               A   D
> 2018-04-16   0   3
> 2018-04-17   4   7
> 2018-04-18   8  11
> 2018-04-19  12  15
> 2018-04-20  16  19
> 2018-04-21  20  23
>
> select A and D where index is 0421:
>  A    20
> D    23
> Name: 2018-04-21 00:00:00, dtype: int32
>
> select by position:iloc
>  A    20
> D    23
> Name: 2018-04-21 00:00:00, dtype: int32
>
> mixed selection:ix:
>               A   D
> 2018-04-18   8  11
> 2018-04-19  12  15
>
> df[df.A<9]:
>              A  B   C   D
> 2018-04-16  0  1   2   3
> 2018-04-17  4  5   6   7
> 2018-04-18  8  9  10  11

### 设置值

```python
df2 = df.copy()
df2[df2.A>10] = 0
print('df2:\n',df2)
df3 = df.copy()
df3.B[df3.A>10] = 0
print('df3:\n',df3)
df['E'] = pd.Series([99,99,99,99,99,99],index=dates)
df['F'] = np.nan
print('df:\n',df)
```

> df2:
>              A  B   C   D
> 2018-04-16  0  1   2   3
> 2018-04-17  4  5   6   7
> 2018-04-18  8  9  10  11
> 2018-04-19  0  0   0   0
> 2018-04-20  0  0   0   0
> 2018-04-21  0  0   0   0
>
> df3:
>               A  B   C   D
> 2018-04-16   0  1   2   3
> 2018-04-17   4  5   6   7
> 2018-04-18   8  9  10  11
> 2018-04-19  12  0  14  15
> 2018-04-20  16  0  18  19
> 2018-04-21  20  0  22  23
>
> df:
>               A   B   C   D   E   F
> 2018-04-16   0   1   2   3  99 NaN
> 2018-04-17   4   5   6   7  99 NaN
> 2018-04-18   8   9  10  11  99 NaN
> 2018-04-19  12  13  14  15  99 NaN
> 2018-04-20  16  17  18  19  99 NaN
> 2018-04-21  20  21  22  23  99 NaN

### 处理丢失数据

```python
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
print('df:\n',df)
print('判断每个元素是否存在NAN:\n',df.isnull())
print('判断整个矩阵是否存在NAN:\n',np.any(df.isnull()==True))
print('直接丢掉有NAN的数据行/列：\n',df.dropna(axis=1,how='any'))
print('填入值给NAN:\n',df.fillna(value=0))
```

> df:
>               A     B     C   D   E   F
> 2018-04-16   0   NaN   2.0   3  99 NaN
> 2018-04-17   4   5.0   NaN   7  99 NaN
> 2018-04-18   8   9.0  10.0  11  99 NaN
> 2018-04-19  12  13.0  14.0  15  99 NaN
> 2018-04-20  16  17.0  18.0  19  99 NaN
> 2018-04-21  20  21.0  22.0  23  99 NaN
>
> 判断每个元素是否存在NAN:
>                  A      B      C      D      E     F
>
> 2018-04-16  False   True  False  False  False  True
> 2018-04-17  False  False   True  False  False  True
> 2018-04-18  False  False  False  False  False  True
> 2018-04-19  False  False  False  False  False  True
> 2018-04-20  False  False  False  False  False  True
> 2018-04-21  False  False  False  False  False  True
>
> 判断整个矩阵是否存在NAN:
>  True
>
> 直接丢掉有NAN的数据行/列：
>              		 A   D   E
> 2018-04-16   0   3  99
> 2018-04-17   4   7  99
> 2018-04-18   8  11  99
> 2018-04-19  12  15  99
> 2018-04-20  16  19  99
> 2018-04-21  20  23  99
>
> 填入值给NAN:
>               A     B     C   D   E    F
> 2018-04-16   0   0.0   2.0   3  99  0.0
> 2018-04-17   4   5.0   0.0   7  99  0.0
> 2018-04-18   8   9.0  10.0  11  99  0.0
> 2018-04-19  12  13.0  14.0  15  99  0.0
> 2018-04-20  16  17.0  18.0  19  99  0.0
> 2018-04-21  20  21.0  22.0  23  99  0.0

### 导入/导出

- pandas object:
  - **read_csv**
  - read_excel
  - read_hdf
  - read_sql
  - **read_json**
  - **read_html**
  - read_stata
  - read_sas
  - read_clipboard
  - **read_pickle**
- 导出

```python
student_data = r"C:\Users\YuL_e\Desktop\tf-test\numpy&pandas\student.csv"
data = pd.read_csv(student_data)
print(data)
data.to_pickle('student.pickle')
data.to_csv('new_student.csv')
```

### 合并一:concat

```python
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
```

> df1:
>       a    b    c    d
> 0  0.0  0.0  0.0  0.0
> 1  0.0  0.0  0.0  0.0
> 2  0.0  0.0  0.0  0.0
>
> df2:
>       a    b    c    d
> 0  1.0  1.0  1.0  1.0
> 1  1.0  1.0  1.0  1.0
> 2  1.0  1.0  1.0  1.0
>
> df3
>       a    b    c    d
> 0  2.0  2.0  2.0  2.0
> 1  2.0  2.0  2.0  2.0
> 2  2.0  2.0  2.0  2.0
>
> res_h:
>       a    b    c    d
> 0  0.0  0.0  0.0  0.0
> 1  0.0  0.0  0.0  0.0
> 2  0.0  0.0  0.0  0.0
> 0  1.0  1.0  1.0  1.0
> 1  1.0  1.0  1.0  1.0
> 2  1.0  1.0  1.0  1.0
> 0  2.0  2.0  2.0  2.0
> 1  2.0  2.0  2.0  2.0
> 2  2.0  2.0  2.0  2.0
>
> res_v:
>       a    b    c    d    a    b    c    d    a
>    b    c    d
> 0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  2.0
> 2.0  2.0  2.0
> 1  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  2.0
> 2.0  2.0  2.0
> 2  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  2.0
> 2.0  2.0  2.0
>
> res_ignore_index:
>       a    b    c    d
> 0  0.0  0.0  0.0  0.0
> 1  0.0  0.0  0.0  0.0
> 2  0.0  0.0  0.0  0.0
> 3  1.0  1.0  1.0  1.0
> 4  1.0  1.0  1.0  1.0
> 5  1.0  1.0  1.0  1.0
> 6  2.0  2.0  2.0  2.0
> 7  2.0  2.0  2.0  2.0
> 8  2.0  2.0  2.0  2.0
>
> df1:
>       a    b    c    d
> 1  0.0  0.0  0.0  0.0
> 2  0.0  0.0  0.0  0.0
> 3  0.0  0.0  0.0  0.0
>
> df2:
>       b    c    d    e
> 2  1.0  1.0  1.0  1.0
> 3  1.0  1.0  1.0  1.0
> 4  1.0  1.0  1.0  1.0
>
> inner:
>       b    c    d
> 0  0.0  0.0  0.0
> 1  0.0  0.0  0.0
> 2  0.0  0.0  0.0
> 3  1.0  1.0  1.0
> 4  1.0  1.0  1.0
> 5  1.0  1.0  1.0
>
> outer:
>       a    b    c    d    e
> 0  0.0  0.0  0.0  0.0  NaN
> 1  0.0  0.0  0.0  0.0  NaN
> 2  0.0  0.0  0.0  0.0  NaN
> 3  NaN  1.0  1.0  1.0  1.0
> 4  NaN  1.0  1.0  1.0  1.0
> 5  NaN  1.0  1.0  1.0  1.0
>
> 使用某个df的index处理：
>       a    b    c    d    b    c    d    e
> 1  0.0  0.0  0.0  0.0  NaN  NaN  NaN  NaN
> 2  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
> 3  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
>
> df1 + df2 + df3:
>       a    b    c    d
> 0  0.0  0.0  0.0  0.0
> 1  0.0  0.0  0.0  0.0
> 2  0.0  0.0  0.0  0.0
> 3  1.0  1.0  1.0  1.0
> 4  1.0  1.0  1.0  1.0
> 5  1.0  1.0  1.0  1.0
> 6  1.0  1.0  1.0  1.0
> 7  1.0  1.0  1.0  1.0
> 8  1.0  1.0  1.0  1.0
>
> df1 + s1:
>       a    b    c    d
> 0  0.0  0.0  0.0  0.0
> 1  0.0  0.0  0.0  0.0
> 2  0.0  0.0  0.0  0.0
> 3  1.0  2.0  3.0  4.0

### 合并二:merge

https://morvanzhou.github.io/tutorials/data-manipulation/np-pd/3-7-pd-merge/

### 出图

```python
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
```





