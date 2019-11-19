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


#【第二步】：通过词频-逆文档频率计算单词关联度
from sklearn.feature_extraction.text import TfidfTransformer #转换为tf-idf
tfidf = TfidfTransformer()
np.set_printoptions(precision=2)


#【第三步】：清洗文本数据
df = pd.read_csv('./movie_data.csv')

import re #导入正则表达式库regex
def preprocessor(text):
    text = re.sub('<[^>]*>','',text)  #把<>里面的东西删掉包括内容
    emoticns = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text = re.sub('[\W]+', ' ',text.lower()) + ''.join(emoticns).replace('-','')
    return text

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

X_train = df.loc[:25000,'review'].values
y_train = df.loc[:25000,'sentiment'].values
X_test = df.loc[25000:,'review'].values
y_test = df.loc[25000:,'sentiment'].values

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

#TfidfVectorizer组合使用了CountVectorizer和TfidfTransformer
tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)

#param_grid包含两个字典
param_grid = [
	#第一个字典：TfidfVectorizer的默认设置
	{
		'vect__ngram_range':[(1,1)],
		'vect__stop_words':[stop,None],
		'vect__tokenizer':[tokenizer,tokenizer_porter],
		'clf__penalty':['l1','l2'],
		'clf__C':[1.0,10.0,100.0] #改变L2和L1正则化的强度
	},
	#第二个字典：在原始词频上完成模型的训练
	{
		'vect__ngram_range':[(1,1)],
		'vect__stop_words':[stop,None],
		'vect__tokenizer':[tokenizer,tokenizer_porter],
		'vect__use_idf':[False],
		'vect__norm':[None],
		'clf__penalty':['l1','l2'],
		'clf__C':[1.0,10.0,100.0] #改变L2和L1正则化的强度
	}
]
lr_tfidf = Pipeline([('vect',tfidf),('clf',LogisticRegression(random_state=0))])
#5折分层交叉验证找到逻辑斯谛回归模型的最佳参数组合
gs_lr_tfidf = GridSearchCV(lr_tfidf,param_grid,scoring='accuracy',cv=5,verbose=1,n_jobs=-1)
gs_lr_tfidf.fit(X_train,y_train)

#输出最佳的参数集
print('Best parameter set: %s' % gs_lr_tfidf.best_params_)
#训练集上5折交叉验证的准确率得分
print('CV Accuracy:%.3f' % gs_lr_tfidf.best_score_)
#测试集准确率得分
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy:%.3f' % clf.score(X_test,y_test))


