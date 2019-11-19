# export PATH=/home/yule/anaconda3/bin:$PATH

import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer() #默认一元组，参数ngram_range(2,2)可调
#CountVectorizer以文本数据数组作为输入，其中文本数据可以是文档或句子
#				返回的是构建的词袋模型
docs = np.array([
	'The sun is shining',
	'The weather is sweet',
	'The sun is shining and the weather is sweet'
])

#【第一步】：将单词转换为特征向量
bag = count.fit_transform(docs)
#创建词袋模型的词汇库，把三个句子转换为稀疏的特征向量
print(count.vocabulary_) #词汇
print(bag.toarray()) #特征向量
'''
{'the': 5, 'sun': 3, 'is': 1, 'shining': 2, 'weather': 6, 'sweet': 4, 'and': 0} #一元组
[[0 1 1 1 0 1 0]
 [0 1 0 0 1 1 1]
 [1 2 1 1 1 2 1]]
'''

#【第二步】：通过词频-逆文档频率计算单词关联度
from sklearn.feature_extraction.text import TfidfTransformer #转换为tf-idf
tfidf = TfidfTransformer()
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
'''
[[ 0.    0.43  0.56  0.56  0.    0.43  0.  ]
 [ 0.    0.43  0.    0.    0.56  0.43  0.56]
 [ 0.4   0.48  0.31  0.31  0.31  0.48  0.31]]
'''

#【第三步】：清洗文本数据
df = pd.read_csv('./movie_data.csv')
print(df.loc[0,'review'][-50:])
# 'erty, it is a beautiful experience for the viewer.'
#此时包含HTML标记和标点符号

import re #导入正则表达式库regex
def preprocessor(text):
    text = re.sub('<[^>]*>','',text)  #把<>里面的东西删掉包括内容
    emoticns = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text = re.sub('[\W]+', ' ',text.lower()) + ''.join(emoticns).replace('-','')
    return text

print(preprocessor(df.loc[0,'review'][-50:]))
# 'erty it is a beautiful experience for the viewer' 
#此时消除掉了HTML标记和标点符号

#移除df中的电影评论信息
df['review'] = df['review'].apply(preprocessor)

# 【第四步】：标记文档(tokenize)
#标记
def tokenizer(text):
	return text.split()

#词干提取
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
	return [porter.stem(word) for word in text.split()]

#停用词移除
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop])
# ['runner', 'like', 'run', 'run', 'lot']
