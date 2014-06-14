import venture.shortcuts as s
ripl = s.make_puma_church_prime_ripl()

from utils import *
from model import Continuous

num_features = 4

width = 10
height = 10
cells = width * height

total_birds = 1000
name = "%dx%dx%d-test" % (width, height, total_birds)

Y = 1
D = 2

runs = 1

# these are estimates from onebird
hypers = [1.011776136710042, 3.315465986337299, 2.6465177320342272, 3.6932748455768913]

params = {
  "name":name,
  "cells":cells,
  "total_birds":total_birds,
  "Y":Y,
  "D":D,
  "hypers":hypers,
}

#thousand_birds = Continuous(ripl, parameters)

Continuous.loadAssumes(ripl, **params)

observes = {}

for y in range(Y):
  for d in range(D):
    for i in range(cells):
      observes[(y, d, i)] = ripl.predict('(observe_birds %d %d %d)' % (y, d, i))

bird_moves = ripl.sample('(get_birds_moving4)')

def sweep(r, *args):
  for y in range(Y):
    r.infer("(pgibbs %d ordered 2 1)" % y)
  r.infer("(mh default one %d)" % (5 * cells))



score = 0
s0 = [s.values[-1] for s in history.nameToSeries['get_birds_moving4']]
for y in range(Y):
  s1 = [s[y] for s in s0]
  for d in range(D):
    s2 = [s[d] for s in s1]
    for i in range(cells):
      s3 = [s[i] for s in s2]
      for j in range(cells):
        s4 = [s[j] for s in s3]
        score += (history.groundTruth['get_birds_moving4'] - sum(s4)/len(s4)) ** 2

print score
