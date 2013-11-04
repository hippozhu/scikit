#!/usr/bin/python
from sklearn.datasets.mldata import fetch_mldata
from sklearn import svm, metrics, neighbors, tree, preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import numpy as np

class Dataset:
  def __init__(self, data, target):
    #self.data = np.loadtxt(datafile, delimiter=',', usecols=xrange(nfeature))
    # target has to be 0's or 1's
    #self.target = np.loadtxt(datafile, delimiter=',', usecols=[-1], dtype=np.int32)
    self.data = data
    self.target = target
    self.base = self.baseline(self.target) 
    self.skf = None

  def saling(self):
    min_max_scaler = preprocessing.MinMaxScaler()
    self.data = min_max_scaler.fit_transform(self.data)

  def baseline(self, target):
    counts = np.bincount(target)
    p = 1.0*counts[0]/len(target)
    return p if p>=0.5 else 1-p

  def kfold(self, k):
    if self.skf == None:
      self.skf = StratifiedKFold(self.target, k)
    return self.skf

  def svmTuning(self):
    print round(self.base, 3)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},\
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10, scoring='accuracy', n_jobs=4)
    clf.fit(self.data, self.target)
    for params, mean_score, scores in clf.grid_scores_:
      print("%0.3f) %0.3f (+/-%0.03f) for %r" % (mean_score - self.base, mean_score, scores.std() / 2, params))
    
class Result:
  def __init__(self, n):
    self.crux = [0] * n

  def addReport(self, r):
    self.crux = [self.crux[i] + 1 if i in r.missed() else self.crux[i] for i in xrange(len(self.crux))]

  def cruxList(self, c):
    return [i for i in xrange(len(self.crux)) if self.crux[i] == c]
    
class Report:
  def __init__(self, clf):
    self.clf = clf
    self.index = np.array([], dtype=np.int32)
    self.expected = np.array([], dtype=np.int32)
    self.predicted = np.array([], dtype=np.int32)
    self.mis = None
    self.hit = None

  def missed(self):
    if self.mis == None:
      self.mis = [self.index[i] for i in xrange(len(self.index)) if self.predicted[i]!=self.expected[i]]
    return self.mis

  def hitted(self):
    if self.hit == None:
      self.hit = [self.index[i] for i in xrange(len(self.index)) if self.predicted[i]==self.expected[i]]
    return self.hit
 
  def accuracy(self):
    return 1.0*len(self.hitted())/(len(self.hitted())+len(self.missed()))

  def report(self):
    print("Classification report for classifier %s:\n%s\n" % (self.clf, metrics.classification_report(self.expected, self.predicted)))
   
  def confusion(self):
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(self.expected, self.predicted))

def cv(data, target, classifier, skf):
  report = Report(classifier)
  for train, test in skf:
    report.index = np.append(report.index, test)
    report.expected = np.append(report.expected, target[test])
    classifier.fit(data[train], target[train])
    report.predicted = np.append(report.predicted, classifier.predict(data[test]))
  return report


'''
set000=setknn.intersection(setdt.intersection(setsvm))
mis_list = svm.missed()
np.mean(np.average(distances[mis_list, 1:2], axis=1))
np.std(np.average(distances[mis_list, 1:2], axis=1))
mis_dist = np.average(distances[mis_list, 1:2], axis=1)
np.histogram(mis_dist, bins=[0,1,2,3,4,5,6,7,8,9,10])
'''

if __name__=="__main__":
  #clf = svm.SVC(kernel='rbf', C=10, gamma=0.001)
  #clf = neighbors.KNeighborsClassifier(5, weights='uniform')
  #cv(bc.data, target, clf, skf);
  bc = fetch_mldata('uci-20070111-breast-w', data_home='~/data', data_name=0,target_name='Class')
  target = np.array([0 if bc.target[i]=='benign' else 1 for i in xrange(len(bc.target))], dtype=np.int32)
  skf = StratifiedKFold(target, 10)

  clf = neighbors.KNeighborsClassifier(5, weights='uniform')
  report_knn=cv(bc.data, target, clf, skf)
  clf = svm.SVC(kernel='rbf', C=10, gamma=0.001)
  report_svm=cv(bc.data, target, clf, skf)
  clf = tree.DecisionTreeClassifier()
  report_dt=cv(bc.data, target, clf, skf)

  nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(bc.data)
  distances, indices = nbrs.kneighbors(bc.data)

  result = Result(len(target))
  result.addReport(report_svm)
  result.addReport(report_knn)
  result.addReport(report_dt)
