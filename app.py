from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0) for words in X])

with open("glove.6B.50d.txt", "rb") as lines:
	w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}

svm_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),("SVM", SVC())])

X = [['Berlin', 'London'],['cow', 'cat'],['pink', 'yellow']]
y = ['capitals', 'animals', 'colors']
svm_w2v.fit(X, y)

# never before seen words!!!
test_X = [['dog'], ['red'], ['Madrid']]

print(svm_w2v.predict(test_X))