# PATH=/home/yule/anaconda3/bin:$PATH
import pyprind  
import pandas as pd  
import os  
import numpy as np  
import re  
import time  
import pickle  
from nltk.corpus import stopwords  
from sklearn.feature_extraction.text import HashingVectorizer  
from sklearn.linear_model import SGDClassifier  

start = time.clock()  

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir,'pkl_objects','stopwords.pkl'),'rb'))


#清理文本中未经处理的数据
def tokenizer(text):  
    text=re.sub('<[^>]*>','',text)#移除HTML标记，#把<>里面的东西删掉包括内容  
    emotions=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)  
    text=re.sub('[\W]+',' ',text.lower())+' '.join(emotions).replace('-','')  
    tokenized = [w for w in text.split() if w not in stop]  
    return tokenized  

vect = HashingVectorizer(decode_error='ignore',n_features=2**21,preprocessor=None,tokenizer=tokenizer)

#导入模型预测  

clf =pickle.load(open(os.path.join('pkl_objects','classifier.pkl'),'rb'))  
label ={0:'negative',1:'positive'}  

example=['it is terrible'] 
X=vect.transform(example)  
print ('Prediction:%s \nProbability:%.2f%%'%(label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))  
  
end = time.clock()
print('finish all in %s' % str(end - start))

'''
Prediction:positive
Probability:84.33%
'''
