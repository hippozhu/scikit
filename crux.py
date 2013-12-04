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
data_train = dataset.data
target_train = [x==1 for x in dataset.target]
s=[i for i in xrange(100)]
w = np.array([1.0/nfeature] * nfeature)
w = relieff(data_train, target_train, w, s, 3)
'''
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
