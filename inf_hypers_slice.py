from train_birds import *
model = makeModel(D=6,learnHypers=True,hyperPrior='(normal 0 10)')
out = getMoves(model,slice_hypers=True,transitions=200,iterations=6,
               label='prior10_7day_slice_')

print 'NEW JOB, prior normal 0 20 with slice hypers'
model = makeModel(D=6,learnHypers=True,hyperPrior='(normal 0 20)')
out = getMoves(model,slice_hypers=True,transitions=200,iterations=6,
               label='prior20_7day_slice_')
