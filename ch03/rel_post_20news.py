# coding=utf-8
# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import sklearn.datasets
import scipy as sp

new_post = \
    """Disk drive problems. Hi, I have a problem with my hard disk.
After 1 year it is working only sporadically now.
I tried to format it, but now it doesn't boot any more.
Any ideas? Thanks.
"""

print("""\
Dear reader of the 1st edition of 'Building Machine Learning Systems with Python'!
For the 2nd edition we introduced a couple of changes that will result into
results that differ from the results in the 1st edition.
E.g. we now fully rely on scikit's fetch_20newsgroups() instead of requiring
you to download the data manually from MLCOMP.
If you have any questions, please ask at http://www.twotoreal.com
""")

all_data = sklearn.datasets.fetch_20newsgroups(subset="all")
print("Number of total posts: %i" % len(all_data.filenames))
# Number of total posts: 18846

groups = [
    'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']
train_data = sklearn.datasets.fetch_20newsgroups(subset="train",
                                                 categories=groups)
print("Number of training posts in tech groups:", len(train_data.filenames))
# Number of training posts in tech groups: 3529

labels = train_data.target
num_clusters = 50  # sp.unique(labels).shape[0]

import nltk.stem
english_stemmer = nltk.stem.SnowballStemmer('english')

from sklearn.feature_extraction.text import TfidfVectorizer


class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5,
                                    stop_words='english', decode_error='ignore'
                                    )

vectorized = vectorizer.fit_transform(train_data.data)
# shape 是一个tuple
num_samples, num_features = vectorized.shape
print("#samples: %d, #features: %d" % (num_samples, num_features))
# samples: 3529, #features: 4712

from sklearn.cluster import KMeans
# make it verbose will print some information -jim
km = KMeans(n_clusters=num_clusters, n_init=1, verbose=1, random_state=3)
# 传给模型的是一个二维数组，一共有3529行，每一行代表一个post，每一行的宽度是4712，其中的数字是TF-IDF的得分
# 在平面上，k means是计算两个点之间的距离，并移动中心点。这里的每一行（4712个数字）相当于一个“点”
clustered = km.fit(vectorized)

# 模型结果的解读...
print("km.labels_=%s" % km.labels_)
# km.labels_=[ 6 34 22 ...,  2 21 26]

print("km.labels_.shape=%s" % km.labels_.shape)
# km.labels_.shape=3529

# 第二版本的内容

from sklearn import metrics
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
# Homogeneity: 0.400
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
# Completeness: 0.206
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
# V-measure: 0.272
print("Adjusted Rand Index: %0.3f" %
      metrics.adjusted_rand_score(labels, km.labels_))
# Adjusted Rand Index: 0.064
print("Adjusted Mutual Information: %0.3f" %
      metrics.adjusted_mutual_info_score(labels, km.labels_))
# Adjusted Mutual Information: 0.197
print(("Silhouette Coefficient: %0.3f" %
       metrics.silhouette_score(vectorized, labels, sample_size=1000)))
# Silhouette Coefficient: 0.006

new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]

similar_indices = (km.labels_ == new_post_label).nonzero()[0]

similar = []
for i in similar_indices:
    dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
    similar.append((dist, train_data.data[i]))

similar = sorted(similar)
print("Count similar: %i" % len(similar))

show_at_1 = similar[0]
show_at_2 = similar[int(len(similar) / 10)]
show_at_3 = similar[int(len(similar) / 2)]

print("=== #1 ===")
print(show_at_1)
print()

print("=== #2 ===")
print(show_at_2)
print()

print("=== #3 ===")
print(show_at_3)
