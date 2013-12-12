import numpy as np
from scipy.spatial.distance import *
#from scikit import *
from numpy import linalg as LA

class Relieff:
  def __init__(self, data, target, weight, k):
    self.data = data
    self.target = target
    self.weight = weight
    self.k = k
    self.ninst = len(self.target)
    # positive and negative instances
    self.pos = [i for i in xrange(self.ninst) if self.target[i]]
    self.neg = [i for i in xrange(self.ninst) if not self.target[i]]

  def selectInst(self, w):
    paired_dist = cdist(self.data, self.data, 'wminkowski', 1, w=w)
    sorted = paired_dist.argsort(axis=1)
    neighbor_hit = np.array([self.target == self.target[sorted[:,i]] for i in xrange(1,2*self.k + 1)]).transpose()
    #return [i for i in xrange(self.ninst) if sum(neighbor_hit[i,:self.k]) < self.k/2 + 1 or sum(neighbor_hit[i]) < self.k]
    return [i for i in xrange(self.ninst) if sum(neighbor_hit[i,:self.k]) < self.k/2 + 1]
    
  def updateWeight(self):
    alpha = .01
    norm_diff = .0
    '''
    for iter in xrange(m):
    while True:
    if iter%200 == 0:
      print "iter %s: |grad|=%s\n%s" %(iter, norm_diff, self.weight)
      print ''
    '''
    #selected_inst = np.random.randint(0, len(inst_pool), size=batch)
    selected_inst = self.selectInst(self.weight)
    print "target inst #: %d" % (len(selected_inst))

    # weighted distances wrt positive and negative instances
    weightedDist_pos = cdist(self.data[selected_inst], self.data[self.pos],
    'wminkowski', 1, w=self.weight)
    weightedDist_neg = cdist(self.data[selected_inst], self.data[self.neg],
    'wminkowski', 1, w=self.weight)
    sorted_pos = weightedDist_pos.argsort(axis=1)
    sorted_neg = weightedDist_neg.argsort(axis=1)

    #print selected_inst
    #kk = self.k
    kk = self.k/2 + 1
    knn_hit = [sorted_pos[i, 1:kk+1] if self.target[selected_inst[i]] else sorted_neg[i, 1:kk+1]
    for i in xrange(len(selected_inst))]
    knn_mis = [sorted_pos[i, :kk] if not self.target[selected_inst[i]] else sorted_neg[i, :kk]
    for i in xrange(len(selected_inst))]

    grads = np.array(
    [np.subtract(np.absolute(np.subtract(self.data[selected_inst[i]], self.data[knn_hit[i]])),
    np.absolute(np.subtract(self.data[selected_inst[i]], self.data[knn_mis[i]])))
    for i in xrange(len(selected_inst))])

    grad = sum(sum(grads)) / (self.k*len(selected_inst))
    norm_diff = LA.norm(grad, 1)
    #new_weight = self.weight - alpha * grad
    #while selected_inst == self.selectInst(new_weight):
    new_si = []
    while True:
      new_si = self.selectInst(self.weight - alpha * grad)
      if selected_inst == new_si:
        alpha *= 2
      else:
        break
    #new_weight -= alpha * grad
    self.weight = self.weight - alpha * grad
    print "selected #: %d, step size=%f" %(len(selected_inst), alpha)
    #print "gone: %s, new:%s" %(list(set(
    return new_si
