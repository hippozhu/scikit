#!/usr/bin/python

import numpy as np
from scikit import *
from relieff import *

#if __name__=="__main__":
#datafile = '/home/yzhu7/data/ionosphere/ionosphere.data'
#nfeature = 34
datafile = '/home/yzhu7/data/diabetes/pima-indians-diabetes.data'
nfeature = 8
nfold = 10
data = np.loadtxt(datafile, delimiter=',', usecols=xrange(nfeature))
target = np.loadtxt(datafile, delimiter=',', usecols=[-1], dtype=np.int32)
dataset=Dataset(data, target, 2, nfeature, nfold)
dataset.saling()

outputfile = open("output.csv", 'w')
baseline = []
best_error_test = []
idx_fold = 0
for train, test in dataset.skf:
  idx_fold += 1
  feature_weight = np.array([1.0] * nfeature)
  cls = Classification(dataset.data[train], dataset.target[train], dataset.data[test], dataset.target[test])
  # before
  clf = neighbors.KNeighborsClassifier(3, weights='uniform', metric='wminkowski', p=1, w=feature_weight)
  cls.classify(clf)
  cls.accuracy()
  baseline.append(len(cls.mis_test))
  
  rlf = Relieff(dataset.data[train], dataset.target[train], feature_weight, 3)
  min_error = len(test)
  train_error = []
  test_error = []
  for i in xrange(200):
    mis_train = rlf.updateWeight()
    print "\nIter %d, fold %d, %s" %(i, idx_fold, rlf.weight)
    clf = neighbors.KNeighborsClassifier(3, weights='uniform', metric='wminkowski', p=1, w=rlf.weight)
    cls.classify(clf)
    cls.accuracy()
    if len(cls.mis_test) < min_error:
      min_error = len(cls.mis_test)
    #train_error.append(len(cls.mis_train))
    train_error.append(len(mis_train))
    test_error.append(len(cls.mis_test))

  best_error_test.append(min_error)
  outputfile.write(','.join(str(x) for x in train_error) + '\n')
  outputfile.write(','.join(str(x) for x in test_error) + '\n\n')

outputfile.close()

print baseline
print best_error_test
'''
print 'final:', w  
clf = neighbors.KNeighborsClassifier(3, weights='uniform', metric='wminkowski', p=1, w=w)
report_knn = Report('knn', clf)
report_knn.cv1(dataset)
report_knn.accuracy()
data_train = dataset.data
target_train = [x==1 for x in dataset.target]
s=[i for i in xrange(100)]
w = np.array([1.0/nfeature] * nfeature)
w = relieff(data_train, target_train, w, s, 3)
clf = svm.SVC(kernel='linear', C=10)
report = Report('svm', clf)
report.cv(dataset)
#report.accuracy()
  dc = Dataclean(2, nfeature)
  dc.fitData(dataset.data, dataset.target)
  clf = neighbors.KNeighborsClassifier(5, weights='uniform')
  report_knn = Report('knn', clf)
  report_knn.cv(dataset)
  result.addReport(report_knn)

  clf = svm.SVC(kernel='linear', C=10)
  report_svm = Report('svm', clf)
  report_svm.cv(dataset)
  result.addReport(report_svm)

  clf = tree.DecisionTreeClassifier()
  report_dt = Report('dt', clf)
  report_dt.cv(dataset)
  result.addReport(report_dt)
'''
