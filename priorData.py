import numpy as np
# two runs of priorRuns for 5 days (0 iterations or run() )
priorRuns=[[(-16.60511567215909, 290.0, 4.0840959548950195),
  (-125.1666761382542, 1256.0, 9.911652088165283),
  (-259.35868782563807, 3125.0, 14.979412078857422),
  (-382.5377011046617, 3989.0, 15.026188850402832),
  (-625.382580247449, 6301.0, 15.249516010284424)],
 [(-14.359626819497077, 225.0, 3.907780885696411),
  (-110.94395431687573, 1746.0, 9.174734115600586),
  (-232.25802777844342, 4901.0, 14.005976915359497),
  (-356.48258951176905, 2609.0, 14.429543018341064),
  (-570.3807411931239, 2257.0, 15.60680103302002)]]

priorRunsMeans = []

# want to take the mean logscore and L2 score over the days for each run
#for run in priorRuns:
    
 #   priorRunsMeans = np.mean

for run in priorRuns:
    ar = np.array(run)
    priorRunsMeans.append( np.mean(ar,axis=0) )
    # will give the means for logscore, L2 and time




# collect samples
