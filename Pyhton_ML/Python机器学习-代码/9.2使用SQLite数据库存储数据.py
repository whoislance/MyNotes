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
import sqlite3

#创建访问数据库的连接
conn = sqlite3.connect('reviews.sqlite')
#创建一个游标，用来遍历记录
c = conn.cursor()

#创建一个表：有文本、情感、日期三个属性
c.execute('CREATE TABLE review_db'\
'(review TEXT, sentiment INTEGER, date TEXT)')

example1 = 'I love this movie'
c.execute("INSERT INTO review_db"\
"(review, sentiment, date) VALUES"\
"(?, ?, DATETIME('now))",(example1,1)) #？是占位符
example2 = 'I disliked this movie'
c.execute("INSERT INTO review_db"\
"(review, sentiment, date) VALUES"\
"(?, ?, DATETIME('now'))",(example2,0))

#保存修改
conn.commit()
#关闭连接
conn.close()
