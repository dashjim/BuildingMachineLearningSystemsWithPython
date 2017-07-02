# coding=utf-8
# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import os
import sys

import scipy as sp

from sklearn.feature_extraction.text import CountVectorizer

from utils import DATA_DIR

TOY_DIR = os.path.join(DATA_DIR, "toy")
posts = [open(os.path.join(TOY_DIR, f)).read() for f in os.listdir(TOY_DIR)]

new_post = "imaging databases"

import nltk.stem
english_stemmer = nltk.stem.SnowballStemmer('english')


class StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

# vectorizer = CountVectorizer(min_df=1, stop_words='english',
# preprocessor=stemmer)
vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')

from sklearn.feature_extraction.text import TfidfVectorizer


class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(
    min_df=1, stop_words='english', decode_error='ignore')

X_train = vectorizer.fit_transform(posts)

num_samples, num_features = X_train.shape
print("#samples: %d, #features: %d" % (num_samples, num_features))

new_post_vec = vectorizer.transform([new_post])
print(new_post_vec, type(new_post_vec)) # 稀疏表示
print(new_post_vec.toarray())
print(vectorizer.get_feature_names())

# 不能处理重复的句子
def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

# 处理重复的句子 (什么鬼？两种方法结果一样？)
def dist_norm(v1, v2):
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())
    v2_normalized = v2 / sp.linalg.norm(v2.toarray())

    delta = v1_normalized - v2_normalized
    # norm()向量的长度
    return sp.linalg.norm(delta.toarray())

dist = dist_norm
# dist = dist_raw (什么鬼？两种方法结果一样？)

best_dist = sys.maxsize
best_i = None

for i in range(0, num_samples):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist(post_vec, new_post_vec)

    print("=== Post %i with dist=%.2f: %s" % (i, d, post))

    if d < best_dist:
        best_dist = d
        best_i = i

print("Best post is %i with dist=%.2f" % (best_i, best_dist))
