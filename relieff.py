import numpy as np
from scipy.spatial.distance import *

def relieff(data_train, target_train, weight, selectedInst, k):
  # positive and negative instances
  pos = [i for i in xrange(len(target_train)) if target_train[i]]
  neg = [i for i in xrange(len(target_train)) if not target_train[i]]

  # weighted distances wrt positive and negative instances
  weightedDist_pos = cdist(data_train[selectedInst], data_train[pos], 'wminkowski', 1, w=weight)
  weightedDist_neg = cdist(data_train[selectedInst], data_train[neg], 'wminkowski', 1, w=weight)

  sorted_pos = weightedDist_pos.argsort(axis=1)
  sorted_neg = weightedDist_neg.argsort(axis=1)

  knn_hit = [sorted_pos[i, 1:k+1] if target_train[i] else sorted_neg[i, 1:k+1] for i in selectedInst]
  knn_mis = [sorted_pos[i, :k] if not target_train[i] else sorted_neg[i, :k] for i in selectedInst]

  grads = np.array(
  [np.subtract(np.absolute(np.subtract(data_train[selectedInst[i]], data_train[knn_hit[i]])),
  np.absolute(np.subtract(data_train[selectedInst[i]], data_train[knn_mis[i]])))
  for i in xrange(len(selectedInst))])

  grad = sum(sum(grads)) / (len(selectedInst)*k)

  print grads.shape, grad.shape
  return weight - grad
