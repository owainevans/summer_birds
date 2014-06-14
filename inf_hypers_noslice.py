from train_birds import *
import time

priors=['(gamma 1 .1)']
labels=['g1_01']
for prior,label in zip(priors,labels):
    model = makeModel(dataset=3, D=3, learnHypers=True, hyperPrior=prior)
    
    out = getMoves(model,slice_hypers=False,transitions=(100,100,25),iterations=50,
               label='ds3_block_new_cycle/%s/'%label )


