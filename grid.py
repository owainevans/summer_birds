import venture.shortcuts as s
ripl = s.make_puma_church_prime_ripl()

from model import OneBird, num_features
from itertools import product

name = "onebird"
width = 4
height = 4
cells = width * height

Y = 1
D = 20

params = {
  "name":name,
  "cells":cells,
  "years":[0],
  "days":range(D)
}

onebird = OneBird(ripl, params)
onebird.loadAssumes()
onebird.loadObserves()

def checkHypers(hypers):
  print hypers
  for i in range(num_features):
    onebird.ripl.force('hypers%d' % i, hypers[i])
  for i in range(5):
    onebird.inferMove()
  return onebird.ripl.get_global_logscore()

def gridHypers(grid):
  return {hypers:checkHypers(hypers) for hypers in product(*grid)}

grid = [range(-10, 11, 4) for i in range(4)]
scoreTable = gridHypers(grid)
scores = sorted(scoreTable.items(), key=lambda (k,v): v, reverse=True)
