# PATH=/home/yule/anaconda3/bin:$PATH
import pyprind  
import pandas as pd  
import os  
import numpy as np  
import re  
import time  
import pickle  
from nltk.corpus import stopwords  
#HashingVectorizer是一个处理文本信息的向量处理器
from sklearn.feature_extraction.text import HashingVectorizer  
#使用SGDClassifier中的partial_fit函数来读取本地存储设备
from sklearn.linear_model import SGDClassifier  
  
start = time.clock()  
  
homedir = os.getcwd()#获取当前文件的路径  
#导入数据并输出到moive_data.csv  
''''' 
pbar=pyprind.ProgBar(50000) 
labels={'pos':1,'neg':0}#正面和负面评论标签 
df = pd.DataFrame() 
for s in ('test','train'): 
    for l in ('pos','neg'): 
        path=homedir+'/aclImdb/%s/%s' %(s,l) 
        for file in os.listdir(path): 
            with open(os.path.join(path,file),'r') as infile: 
                txt =infile.read() 
            df =df.append([[txt,labels[l]]],ignore_index=True) 
            pbar.update() 
df.columns=['review','sentiment'] 
np.random.seed(0) 
df=df.reindex(np.random.permutation(df.index))#重排数据集，打散正负样本数据 
df.to_csv(homedir+'/movie_data.csv',index=False) 
'''  
#文本向量化，并训练模型和更新  
df=pd.read_csv(homedir+'/movie_data.csv')  
stop = stopwords.words('english')#获得英文停用词集  

#【第一步】清理文本中未经处理的数据
def tokenizer(text):  
    text=re.sub('<[^>]*>','',text)#移除HTML标记，#把<>里面的东西删掉包括内容  
    emotions=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)  
    text=re.sub('[\W]+',' ',text.lower())+' '.join(emotions).replace('-','')  
    tokenized = [w for w in text.split() if w not in stop]  
    return tokenized  

#【第二步】生成器函数，每次读取且返回一个包含评论和类标的元组
def stream_docs(path):  
    with open(path,'r') as csv:  
        next(csv) #skip header  
        for line in csv:  
            text,label = line[:-3] ,int(line[-2])  
            yield text,label  

#【第三步】获得小型子批次
def get_minibatch(doc_stream,size):    #获得小型子批次
    docs,y =[],[]  
    try:  
        for _ in range(size):  #返回size个文档内容
            text,label =next(doc_stream)  
            docs.append(text)  
            y.append(label)  
    except StopIteration:  
        return None,None  
    return docs,y  

#【第四步】HashingVectorizer是一个处理文本信息的向量处理器
vect=HashingVectorizer(decode_error='ignore',n_features=2**21,preprocessor=None,tokenizer=tokenizer)  
clf = SGDClassifier (loss='log',random_state=1,n_iter=1)#随机梯度下降，每次用一个样本更新权重  
doc_stream = stream_docs(path=homedir+'/movie_data.csv')  

#【第五步】外存学习
pbar = pyprind.ProgBar(45)  #进度条
classes=np.array([0,1])  
for _ in range(45):  
    X_train,y_train = get_minibatch(doc_stream, size=1000)  #每个子批次含1000个文档
    if not X_train:break  
    X_train = vect.transform(X_train)  
    clf.partial_fit(X_train, y_train, classes=classes)#部分训练  
    pbar.update()  


#测试  
X_test,y_test=get_minibatch(doc_stream, size=5000)  
X_test=vect.transform(X_test)  
print ('Accuracy:%.3f' %clf.score(X_test,y_test))  
clf=clf.partial_fit(X_test,y_test)#更新模型  
# Accuracy:0.876


end = time.clock()
print('finish all in %s' % str(end - start))

'''
python3 "/home/yule/Dropbox/于乐的笔记/机器学习的笔记/Python机器学习-代码/8.4在线算法与外存学习.py"
/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [#                             ] 100% | ETA: 00:01:59/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [##                            ] 100% | ETA: 00:01:55/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [###                           ] 100% | ETA: 00:01:48/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [####                          ] 100% | ETA: 00:01:41/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [#####                         ] 100% | ETA: 00:01:37/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [######                        ] 100% | ETA: 00:01:34/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [#######                       ] 100% | ETA: 00:01:28/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [########                      ] 100% | ETA: 00:01:25/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [#########                     ] 100% | ETA: 00:01:20/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [##########                    ] 100% | ETA: 00:01:15/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [###########                   ] 100% | ETA: 00:01:13/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [############                  ] 100% | ETA: 00:01:11/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [#############                 ] 100% | ETA: 00:01:06/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [##############                ] 100% | ETA: 00:01:03/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [###############               ] 100% | ETA: 00:00:58/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [################              ] 100% | ETA: 00:00:53/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [#################             ] 100% | ETA: 00:00:50/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [##################            ] 100% | ETA: 00:00:48/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [###################           ] 100% | ETA: 00:00:43/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [####################          ] 100% | ETA: 00:00:40/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [#####################         ] 100% | ETA: 00:00:34/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [######################        ] 100% | ETA: 00:00:29/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [#######################       ] 100% | ETA: 00:00:26/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [########################      ] 100% | ETA: 00:00:23/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [#########################     ] 100% | ETA: 00:00:18/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [##########################    ] 100% | ETA: 00:00:16/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [###########################   ] 100% | ETA: 00:00:10/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [############################  ] 100% | ETA: 00:00:05/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [############################# ] 100% | ETA: 00:00:02/home/yule/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
  DeprecationWarning)
0% [##############################] 100% | ETA: 00:00:00
Total time elapsed: 00:01:59
Accuracy:0.876
'''