# from sklearn.feature_extraction.text import TfidfVectorizer
# corpus = [
#    'This is the first document.',
#     'This document is the second document.',
#     'And this is the third one.',
#     'Is this the first document?' ]
# vec = TfidfVectorizer()
# X = vec.fit_transform(corpus)
# print(type(X))
# print(X)

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[d]\w+')
# print(stemmer)
text = 'This document is the second document.'.split(' ')
text = ' '.join([stemmer.stem(word) for word in text])
print(text,len(text))
tokenlist = tokenizer.tokenize(text)
print(tokenlist)