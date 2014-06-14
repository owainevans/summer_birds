from math import exp
from numpy.random import multinomial

from utils import *

from model import num_features


# these are ground truths
hypers = [5, 10, 10, 10]

def mem(f):
  table = {}
  def memmed(*args):
    if args not in table:
      table[args] = f(*args)
    return table[args]
  return memmed

def generate(dataset=None, name=None, cells=None, years=None, days=None, total_birds=None, **params):
  features_file = "data/input/dataset%d/%s-features.csv" % (dataset, name)
  print "Loading features from %s" % features_file
  features = readFeatures(features_file)
  
  def phi(y, d, i, j):
    fs = features[(y, d, i, j)]
    return exp(sum([fs[k] * hypers[k] for k in range(num_features)]))

  def bird_dist(y, d, i):
    return normalize([phi(y, d, i, j) for j in range(cells)])

  @mem
  def move_birds(y, d, i):
    return multinomial(count_birds(y, d, i), bird_dist(y, d, i))

  @mem
  def count_birds(y, d, i):
    if d == 0: return total_birds if i == 0 else 0
    return sum([move_birds(y, d-1, j)[i] for j in range(cells)])

  def bird_locs(y, d):
    return [count_birds(y, d, i) for i in range(cells)]

  for y in years:
    path = 'generated%d/%d/' % (dataset, y)
    ensure(path)
    for d in days:
      drawBirds(bird_locs(y, d), path + '%02d.png' % d, **params)

generate(**getParams(2))
