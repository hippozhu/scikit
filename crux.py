#!/usr/bin/python

from scikit import *
#from scikit import Dataset, Result, Report, cv

datafile = 'diabetes/pima-indians-diabetes.data'
nfeature = 8
data = np.loadtxt(datafile, delimiter=',', usecols=xrange(nfeature))
target = np.loadtxt(datafile, delimiter=',', usecols=[-1], dtype=np.int32)
pima=Dataset(data, target)
pima.saling()

clf = neighbors.KNeighborsClassifier(5, weights='uniform')
report_knn=cv(pima.data, pima.target, clf, pima.kfold(10))

clf = svm.SVC(kernel='linear', C=1000)
report_svm=cv(pima.data, pima.target, clf, pima.kfold(10))

clf = tree.DecisionTreeClassifier()
report_dt=cv(pima.data, pima.target, clf, pima.kfold(10))
