# coding=utf-8
# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License


from matplotlib import pyplot as plt
import numpy as np

# We load the data with load_iris from sklearn
from sklearn.datasets import load_iris
data = load_iris()

# load_iris returns an object with several fields
features = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names

for t in range(3):
 if t == 0:
     c = 'r'
     marker = '>'
 elif t == 1:
     c = 'g'
     marker = 'o'
 elif t == 2:
     c = 'b'
     marker = 'x'
 # target数组中=0的是一类，作为features的下标，来选出同类的，再把列选出来。
 # feature一共有四种（四列）【0，1,2,3】，一共可以组合出6种图，这里只显示一种
 plt.scatter(features[target == t, 0],
            features[target == t, 2],
            marker=marker,
            c=c)
# We use NumPy fancy indexing to get an array of strings:
# 可以理解为新的数组中的元素是不断的用每一个下标（target）选出来的。
labels = target_names[target]

# The petal length is the feature at position 2
plength = features[:, 2]

# Build an array of booleans:
is_setosa = (labels == 'setosa')

# This is the important step:
max_setosa =plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()
print('Maximum of setosa: {0}.'.format(max_setosa))

print('Minimum of others: {0}.'.format(min_non_setosa))

# setosa 是最好区分的，可以直接选出来，后面都是要区分另外两种
# ~ is the boolean negation operator
features = features[~is_setosa]
labels = labels[~is_setosa]
# Build a new target variable, is_virigina
# 这是后面用来检测预测效果用的。
is_virginica = (labels == 'virginica')

# Initialize best_acc to impossibly low value
best_acc = -1.0
for fi in range(features.shape[1]): # features 是一个N * 4的二维数组
    # We are going to test all possible thresholds
    thresh = features[:,fi]
    for t in thresh: # 分别取第一个feature类别（一列数据）中的每一个数据。

        # Get the vector for feature `fi`
        feature_i = features[:, fi]
        # apply threshold `t`
        pred = (feature_i > t)
        # 预测和真实值一致则为True也就是1，其他为零，然后求平均数
        acc = (pred == is_virginica).mean()
        rev_acc = (pred == ~is_virginica).mean()
        if rev_acc > acc:
            reverse = True
            acc = rev_acc
        else:
            reverse = False

        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t
            best_reverse = reverse

print(best_fi, best_t, best_reverse, best_acc)
# result:
#    (3, 1.6000000000000001, False, 0.93999999999999995)

# The following code has error and thus disable for now - Jim
# def is_virginica_test(fi, t, reverse, example):
#     'Apply threshold model to a new example'
#     test = example[fi] > t
#     if reverse:
#         test = not test
#     return test

# 交叉验证：1 - 极端情况是每次取出来一个作为测试数据，其他的作为训练数据。
from threshold import fit_model, predict

# ning accuracy was 96.0%.
# ing accuracy was 90.0% (N = 50).
correct = 0.0

for ei in range(len(features)):
    # select all but the one at position `ei`:
    training = np.ones(len(features), bool) # 全部为True
    training[ei] = False # 一个为false
    testing = ~training # 全部为false, 一个为True
    model = fit_model(features[training], is_virginica[training])
    predictions = predict(model, features[testing])
    correct += np.sum(predictions == is_virginica[testing])
acc = correct/float(len(features))
print('Accuracy: {0:.1%}'.format(acc))


###########################################
############## SEEDS DATASET ##############
###########################################

from load import load_dataset

feature_names = [
    'area',
    'perimeter',
    'compactness',
    'length of kernel',
    'width of kernel',
    'asymmetry coefficien',
    'length of kernel groove',
]
features, labels = load_dataset('seeds')


# 交叉验证：2 - k fold
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=1)
from sklearn.cross_validation import KFold

kf = KFold(len(features), n_folds=5, shuffle=True)
means = []
# kf只是下标，还是要事先准备好训练数据和标记。
for training,testing in kf:
   # We learn a model for this fold with `fit` and then apply it to the
   # testing data with `predict`:
   classifier.fit(features[training], labels[training]) # features包含的多列数据
   prediction = classifier.predict(features[testing])

   # np.mean on an array of booleans returns fraction
 # of correct decisions for this fold:
   curmean = np.mean(prediction == labels[testing])
   means.append(curmean)
print('Mean accuracy: {:.1%}'.format(np.mean(means)))

# 同样是KNN，对比参数标准化后的效果
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

classifier = KNeighborsClassifier(n_neighbors=1)
classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])

means = []
for training,testing in kf:
    # We learn a model for this fold with `fit` and then apply it to the
    # testing data with `predict`:
    classifier.fit(features[training], labels[training])
    prediction = classifier.predict(features[testing])

    # np.mean on an array of booleans returns fraction
    # of correct decisions for this fold:
    curmean = np.mean(prediction == labels[testing])
    means.append(curmean)
# 尼玛，每次结果都不一样！什么鬼？
print('Mean accuracy: {:.1%}'.format(np.mean(means)))
