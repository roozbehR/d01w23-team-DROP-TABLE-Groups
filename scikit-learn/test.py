from sklearn import clone
from sklearn.feature_extraction.text import CountVectorizer

cntvec = CountVectorizer(vocabulary={'g': 0, 'a': 1, 't': 2, 'c': 3})
print(cntvec.vocabulary is clone(cntvec, safe=False).vocabulary)

