import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

# 读取数据
data = pd.read_csv('./abcnews-date-text(1).csv',error_bad_lines=False,usecols=['headline_text'])
data.head()
data = data.head(10000)

# print(data.info())

# 删除重复数据

# 查看重复行
# print(data[data['headline_text'].duplicated(keep=False)])
'''
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.duplicated.html
# df = pd.DataFrame({
#     'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
#     'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
#     'rating': [4, 4, 3.5, 15, 5] })
# print(df,'\n',df.duplicated(keep=False),'\n',df[df.duplicated(keep=False)])
'''

# 删除重复行
'''
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html
'''
data = data.drop_duplicates(subset=['headline_text'])
# print(len(data))

# 数据预处理
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
desc = data['headline_text'].values # np.ndarray

# TfidfVectorizer
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# vectorizer = TfidfVectorizer(stop_words=stop_words)
# X = vectorizer.fit_transform(desc)
# word_features = vectorizer.get_feature_names()
# print(X.shape)
# print(len(word_features))
# print(word_features[5000:5100])

# Stemming 讲单词还原为词干
stemmer = SnowballStemmer('english') # https://www.kite.com/python/docs/nltk.SnowballStemmer
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+') # https://www.kite.com/python/docs/nltk.RegexpTokenizer
def tokenize(text):
    """
    先进行 stemming 然后 tokenize
    params:
        text: 一个句子

    return:
        tokens 列表
    """
    text = ' '.join([stemmer.stem(word) for word in text.split(' ')])
    tokens = tokenizer.tokenize(text)

    return tokens

# Tfidf向量化
vectorizer = TfidfVectorizer(stop_words=stop_words,tokenizer=tokenize,max_features=2000)
X = vectorizer.fit_transform(desc)
word_features = vectorizer.get_feature_names()
# print(X.shape)
# print(len(word_features))
# print(word_features[:50])

# K-means聚类
# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=3,random_state=0).fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('elbow.png')
plt.show()