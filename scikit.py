#!/usr/bin/python
from sklearn.datasets.mldata import fetch_mldata
from sklearn import svm, metrics, neighbors, naive_bayes, tree, preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import numpy as np
from scipy.spatial.distance import *

from sc import *

class Dataset:
  def __init__(self, data, target, nc, nf, nfold):
    #self.data = np.loadtxt(datafile, delimiter=',', usecols=xrange(nfeature))
    # target has to be 0's or 1's
    #self.target = np.loadtxt(datafile, delimiter=',', usecols=[-1], dtype=np.int32)
    self.data = data
    self.target = target
    self.nclass = nc
    self.nfeature = nf
    self.ninst = len(target)
    self.base = self.baseline(self.target) 
    #self.skf = StratifiedKFold(self.target, 10)
    self.nfold = nfold
    self.skf = StratifiedKFold(self.target, self.nfold)
    self.dm = None

  def saling(self):
    min_max_scaler = preprocessing.MinMaxScaler()
    self.data = min_max_scaler.fit_transform(self.data)

  def baseline(self, target):
    counts = np.bincount(target)
    p = 1.0*counts[0]/len(target)
    return p if p>=0.5 else 1-p

  def resetKfold(self, k):
    self.skf = StratifiedKFold(self.target, k)

  def svmTuning(self):
    print round(self.base, 3)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},\
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10, scoring='accuracy', n_jobs=4)
    clf.fit(self.data, self.target)
    for params, mean_score, scores in clf.grid_scores_:
      print("%0.3f) %0.3f (+/-%0.03f) for %r" % (mean_score - self.base, mean_score, scores.std() / 2, params))

  def distMatrix(self):
    if self.dm == None:
      self.dm = squareform(pdist(self.data, 'minkowski', 1))
    return self.dm
    
  def purity(self, rep):
    nnIdx = np.argmin(self.distMatrix()[:, rep], axis=1)
    nn = [rep[i] for i in nnIdx]
    hits = [1 if self.target[i] == self.target[nn[i]] else 0 for i in xrange(self.ninst)]
    loo = sum(hits)/float(len(hits))
    purity = .0
    for r in rep:
      hit = [hits[i] for i in xrange(len(hits)) if nn[i] == r]
      p = sum(hit)/float(len(hit))
      purity += p if p>=0.5 else 1-p
    return purity/len(rep)

  def weightedPurity(self, rep, w):
    nnIdx = np.argmin(self.distMatrix()[:, rep], axis=1)
    nn = [rep[i] for i in nnIdx]
    purity = .0
    self.ided = 0
    self.misIded = 0
    self.clusterSize = []
    self.clusterCrux = []
    self.cruxPurity = []
    for r in rep:
      memberIdx = [i for i in xrange(self.ninst) if nn[i] == r]
      self.clusterSize.append(len(memberIdx))
      memberTarget = [self.target[i] for i in memberIdx]
      p = sum(memberTarget)/float(len(memberTarget))
      if p>=0.5:
        purity += w * p
	self.ided += sum(memberTarget)
	self.misIded += len(memberTarget) - sum(memberTarget)
	self.clusterCrux.append(1)
	self.cruxPurity.append(p)
      else:
        purity += (1-w)*(1-p)
	self.clusterCrux.append(0)
    return purity/len(rep)

  def clusterStat(self):
    cruxSize = [self.clusterSize[i] for i in xrange(len(self.clusterSize)) if self.clusterCrux[i] == 1]
    nonCruxSize= [self.clusterSize[i] for i in xrange(len(self.clusterSize)) if self.clusterCrux[i] == 0]
    print "total: %s, %s" %(len(self.clusterSize), np.mean(self.clusterSize))
    print "crux : %s, %s" %(len(cruxSize), np.mean(cruxSize))
    print "non  : %s, %s" %(len(nonCruxSize), np.mean(nonCruxSize))
    print "Improved: %s-%s=%s" % (self.ided, self.misIded, self.ided-self.misIded)
    print cruxSize, np.mean(cruxSize)
    print ["%0.3f" % i for i in self.cruxPurity], "%0.3f" % np.mean(self.cruxPurity)
    
class Result:
  def __init__(self, n):
    self.crux = [0] * n
    self.report = []

  def addReport(self, r):
    self.report.append(r)
    self.crux = np.add(self.crux, r.cruxList())

  def accuracy(self):
    for r in self.report:
      r.accuracy()

  def cruxness(self, c):
    return [i for i in xrange(len(self.crux)) if self.crux[i] == c]

  def nonCruxness(self, c):
    return [i for i in xrange(len(self.crux)) if self.crux[i] != c]

  def outputCruxness(self, c, filename):
    cruxness = [[1] if self.crux[i] >= c else [0] for i in xrange(len(self.crux))]
    np.savetxt(filename, cruxness, fmt="%d")


class Report:
  def __init__(self, cname, clf):
    self.cname = cname
    self.clf = clf
    self.index = np.array([], dtype=np.int32)
    self.expected = np.array([], dtype=np.int32)
    self.predicted = np.array([], dtype=np.int32)
    self.crux_pred = np.array([], dtype=np.int32)
    self.mis = None
    self.hit = None
    self.crux = None

  def cv(self, dataset):
    dc = Dataclean(dataset.nclass, dataset.nfeature)
    for train, test in dataset.skf:
      self.index = np.append(self.index, test)
      self.expected = np.append(self.expected, dataset.target[test])
      '''
      dc.fitData(dataset.data[train], dataset.target[train], dataset.nfold-1)
      noncrux = dc.cleanedData()
      self.clf.fit(dataset.data[train][noncrux], dataset.target[train][noncrux])
      '''
      self.clf.fit(dataset.data[train], dataset.target[train])
      train_pred = self.clf.predict(dataset.data[train])
      test_pred = self.clf.predict(dataset.data[test])
      mysc = SupervisedClustering(dataset.data[train], train_pred != dataset.target[train], dataset.data[test], test_pred != dataset.target[test])
      best_ind = runGA(mysc)
      self.predicted = np.append(self.predicted, self.clf.predict(dataset.data[test]))

  def cv1(self, dataset):
    for train, test in dataset.skf:
      self.index = np.append(self.index, test)
      self.expected = np.append(self.expected, dataset.target[test])
      self.clf.fit(dataset.data[train], dataset.target[train])
      self.predicted = np.append(self.predicted, self.clf.predict(dataset.data[test]))

  def missed(self):
    if self.mis == None:
      self.mis = [self.index[i] for i in xrange(len(self.index)) if self.predicted[i]!=self.expected[i]]
    return self.mis

  def hitted(self):
    if self.hit == None:
      self.hit = [self.index[i] for i in xrange(len(self.index)) if self.predicted[i]==self.expected[i]]
    return self.hit
 
  def cruxList(self):
    if self.crux == None:
      self.crux = [1 if self.predicted[i]!=self.expected[i] else 0 for i in xrange(len(self.index))]
    return self.crux
    
  def accuracy(self):
    acc = 1.0*len(self.hitted())/(len(self.hitted())+len(self.missed()))
    print self.cname, acc, '(', len(self.hitted()),':', len(self.missed()),')'
    #return acc

  def report(self):
    print("Classification report for classifier %s:\n%s\n" % (self.clf, metrics.classification_report(self.expected, self.predicted)))
   
  def confusion(self):
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(self.expected, self.predicted))

  def outputCrux(self, filename):
    cruxness = [[1] if self.predicted[i]!=self.expected[i] else [0] for i in xrange(len(self.index))]
    np.savetxt(filename, cruxness, fmt="%d")

class Dataclean:
  def __init__(self, nc, nf):
    self.nclass = nc
    self.nfeature = nf
    self.clfs = []
    self.clfs.append(('knn', neighbors.KNeighborsClassifier(5, weights='uniform')))
    self.clfs.append(('svm', svm.SVC(kernel='linear', C=10)))
    self.clfs.append(('dt', tree.DecisionTreeClassifier()))
    self.clfs.append(('nb', naive_bayes.GaussianNB()))

  def fitData(self, d, t, nfold):
    self.result = Result(len(t))
    for cname, clf in self.clfs:
      ds = Dataset(d, t, self.nclass, self.nfeature, nfold)
      report = Report(cname, clf)
      report.cv1(ds)
      self.result.addReport(report)

  def cleanedData(self):
    return self.result.nonCruxness(len(self.clfs))

