import random, math
import numpy as np
from scipy.spatial.distance import *
from scikit import *

from deap import base
from deap import creator
from deap import tools

def probRandom(prob):
  return 1 if random.random()<prob else 0

def kRandom(k):
  return 1 if random.random() < k/float(len(self.data)) else 0

toolbox = base.Toolbox()
random.seed(64)
CXPB, MUTPB, NGEN = 0.5, 0.2, 50
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("attr_bool", probRandom, 0.1)
#toolbox.register("attr_bool", kRandom, 30)
toolbox.register("mate", tools.cxTwoPoints)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def runGA(sc):
  toolbox.register("individual", tools.initRepeat, creator.Individual, 
  toolbox.attr_bool, len(sc.data))
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  toolbox.register("evaluate", sc.myevaluate)

  pop = toolbox.population(n=300)
  print("Start of evolution")

  # Evaluate the entire population
  fitnesses = list(map(toolbox.evaluate, pop))
  for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

  print("  Evaluated %i individuals" % len(pop))

  for g in range(NGEN):
    print("-- Generation %i --" % g)

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
	if random.random() < CXPB:
	    toolbox.mate(child1, child2)
	    del child1.fitness.values
	    del child2.fitness.values

    for mutant in offspring:
	if random.random() < MUTPB:
	    toolbox.mutate(mutant)
	    del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
	ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(invalid_ind))

    # The population is entirely replaced by the offspring
    pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5

    print("  Min %s, Max %s, Avg %s, Std %s" % (min(fits), max(fits), mean, std))

    best_ind = tools.selBest(pop, 1)[0]
    sc.clustering(best_ind)

  print("-- End of (successful) evolution --")

  best_ind = tools.selBest(pop, 1)[0]
  #print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
  print("Best individual fitness %s" % (best_ind.fitness.values))


class SupervisedClustering:
  def __init__(self, data, target, test, test_target):
    self.data = data 
    self.target = target
    self.test = test
    self.test_target = test_target
    self.nclass = 2
    self.ninst = len(target)
    self.dm = squareform(pdist(self.data, 'minkowski', 1))
      
  def myevaluate(self, individual):
    beta = 0.8
    weight = 1.0
    rep = [i for i in xrange(len(individual)) if individual[i]==1]
    nnIdx = np.argmin(self.dm[:, rep], axis=1)
    nn = [rep[i] for i in nnIdx]
    purity = .0
    nCruxCluster = 0
    cruxCluster = []
    for r in rep:
      memberIdx = [i for i in xrange(self.ninst) if nn[i] == r]
      memberTarget = [self.target[i] for i in memberIdx]
      p = sum(memberTarget)/float(len(memberTarget))
      if self.isCruxCluster(memberTarget):
        purity += weight * p
	nCruxCluster += 1
      else:
        purity += (1-weight)*(1-p)
    return purity/len(rep) - beta * math.sqrt((len(rep)- self.nclass)/float(self.ninst)),

  def clustering(self, individual):
    rep = [i for i in xrange(len(individual)) if individual[i]==1]
    nnIdx = np.argmin(self.dm[:, rep], axis=1)
    nn = [rep[i] for i in nnIdx]
    cruxCluster = []
    train_error = []
    for r in rep:
      memberIdx = [i for i in xrange(self.ninst) if nn[i] == r]
      memberTarget = [self.target[i] for i in memberIdx]
      isCrux = self.isCruxCluster(memberTarget)
      cruxCluster.append(isCrux)
      train_error += [memberTarget[i] != isCrux for i in xrange(len(memberTarget))]
    
    print "# of clusters:", len(rep), sum(cruxCluster)
    nrIdx = np.argmin(cdist(self.test, self.data[rep, :], 'minkowski', 1), axis=1)
    test_error = [cruxCluster[i] for i in nrIdx] != self.test_target
    print sum(test_error), sum(self.test_target), ",", sum(train_error), sum(self.target)
    #print sum(test_error), sum(self.test_target)
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(self.test_target, [cruxCluster[i] for i in nrIdx]))
    #print sum(train_error), sum(self.target)

  def isCruxCluster(self, memberTarget):
    return sum(memberTarget)/float(len(memberTarget))> 0.6 and len(memberTarget)>4

'''
  def clusterStat(self):
    cruxSize = [self.clusterSize[i] for i in xrange(len(self.clusterSize)) if self.clusterCrux[i] == 1]
    nonCruxSize= [self.clusterSize[i] for i in xrange(len(self.clusterSize)) if self.clusterCrux[i] == 0]
    print "total: %s, %s" %(len(self.clusterSize), np.mean(self.clusterSize))
    print "crux : %s, %s" %(len(cruxSize), np.mean(cruxSize))
    print "non  : %s, %s" %(len(nonCruxSize), np.mean(nonCruxSize))
    print "Improved: %s-%s=%s" % (self.ided, self.misIded, self.ided-self.misIded)
    print cruxSize, np.mean(cruxSize)
    print ["%0.3f" % i for i in self.cruxPurity], "%0.3f" % np.mean(self.cruxPurity)

if __name__ == "__main__":
  datafile = 'pima2.csv'
  nfeature = 8 
  nclass = 2
  data = np.loadtxt(datafile, delimiter=',', usecols=xrange(nfeature))
  target = np.loadtxt(datafile, delimiter=',', usecols=[-1], dtype=np.int32)
  #deapInit()
  mysc = SupervisedClustering(data, target)
  mysc.runGA()
'''
