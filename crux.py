#!/usr/bin/python

from scikit import *

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
clf = naive_bayes.GaussianNB()
report = Report('nb', clf)
report.cv(dataset)
#report.accuracy()
'''
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
