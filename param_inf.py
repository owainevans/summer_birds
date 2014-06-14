import scipy.stats
from train_birds_param import *
from utils import writeHypers
import numpy as np
import sys

## notes of dependencies:
# makeModel from trainbirds calls Poisson and gives params (rename poisson class)
# call getMoves, calls posteriorSamples, calls run
# run calls ensure (from utils) 
# log, called from run, calls writeLog, which writes to basedDirectory\
# basedirectory comes from getMoves. should be harmless to do this write
# as long as no absolute paths involve. 

dataset = 2
if len(sys.argv) > 1:
    dataset = int(sys.argv[1])
elif len(sys.argv)>2:
    output_dir = sys.argv[2]
else:
    output_dir = None
    #output_dir = 'data/output/dataset%s/'%dataset

print output_dir


def getHypers(ripl):
  return tuple([ripl.sample('hypers%d'%i) for i in range(4)])

# def writeHypers(hypers, path=None, dataset=None):
#   if path is None:
#     path = "data/output/"
#   filename = path + "dataset%d/estimated-parameters.csv" % dataset
#   ensure(filename)
  
#   with open(filename, 'w') as f:
#     f.write("b1,b2,b3,b4\n")
#     f.write(','.join(map(str, hypers)))
#     f.write('\n')

#   print "Wrote parameters to " + filename


runs = 1 # 4
transitions = (1,1,1) # (100,100,25)
iterations = 2 # 50

allHypers = []

for run in range(runs):
    model = makeModel(dataset=dataset, D=3, learnHypers=True, hyperPrior='(gamma 1 .1)')
    
    posteriorLogs,model,moves,locs = getMoves(model,slice_hypers=False,
                                              transitions=transitions,iterations=iterations,
                                              label='parameter_estimate_ds%s/'%str(dataset))

    hypers = getHypers(model.ripl)
    print 'Run %i. Hypers: '%run, hypers

    allHypers.append(hypers)

print allHypers

mean_parameter_values = list( np.mean(np.array( allHypers ),axis=0) )
sem_parameter_values = scipy.stats.sem(np.array( allHypers ),axis=0)

print 'Mean parameter value across runs:', np.round(mean_parameter_values,2)
print 'Standard Error of the Mean parameter value across runs:', scipy.stats.sem(np.array( allHypers ),axis=0)

path = output_dir 
writeHypers(mean_parameter_values, path=path, dataset=dataset)
    
