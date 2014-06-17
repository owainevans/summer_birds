import time
import venture.shortcuts as s
ripl = s.make_puma_church_prime_ripl()

from utils import *
from model import Poisson, num_features

def makeModel(dataset=2, D=6, Y=1, learn_hypers=True, hypers_prior='(normal 0 20)'):
  width,height = 10,10
  cells = width * height
  num_birds = 1000 if dataset == 2 else 1000000
  name = "%dx%dx%d-train" % (width, height, num_birds)
  runs = 1
  hypers = [5, 10, 10, 10] if not learn_hypers else [hypers_prior]*4

  params = {
  "name":name,
  "width":width,
  "height":height,
  "dataset":dataset,
  "num_birds":num_birds,
  "years":range(Y),
  "days":[],
  "hypers":hypers,
  "maxDay":D}

  model = Poisson(ripl,params)

  return model

  
def getHypers(ripl):
  return tuple([ripl.sample('hypers%d'%i) for i in range(num_features)])

def log(t,day,iteration,transitions,model,baseDirectory=None):
  dt = time.time() - t[0]
  data = (day, iteration, transitions,
          model.ripl.get_global_logscore(),
          model.computeScoreDay(model.days[-2]),
          getHypers(model.ripl), dt)
  
  print map(lambda ar:np.round(ar,2), data)
  writeLog(day,data,baseDirectory)
  
  t[0] += dt
  return data


def writeLog(day,data,baseDirectory):
  with open(baseDirectory+'scoreLog.dat','a') as f:
        f.write('\nDay:%d \n'%day + str(data) )
        


def run(model,iterations=1, transitions=(100,50,50), baseDirectory='',slice_hypers=False):
  day1,day2 = transitions[1],transitions[2]
  transitions = transitions[0]

  
  assert model.parameters['days'] == []
  learn_hypers = isinstance(model.parameters['hypers'][0],str)
  
  D = model.parameters['maxDay']
  Y = max(model.parameters['years'])
  dataset = model.parameters['dataset']
  ensure(baseDirectory)
  
  print "\nStarting run. \nParams: ",model.parameters
  model.ripl.clear()
  model.loadAssumes()
  model.updateObserves(0)

  logs = []
  t = [time.time()]
  dayToHypers = [day1,day2,4,2,1,1] + [0]*D
    
  for d in range(1,D):
    print "\nDay %d" % d
    model.updateObserves(d)  # self.days.append(d)
    logs.append( log(t,d,0,transitions,model,baseDirectory) )
    
    for i in range(iterations): # iterate inference (could reduce from 5)

      if learn_hypers:
        args = (dayToHypers[d-1], d-1, (Y+1)*transitions)
        if slice_hypers:
          s='(cycle ((slice hypers one %d) (mh %d one %d)) 1)'%args
        else:
          s='(cycle ((mh hypers one %d) (mh %d one %d)) 1)'%args

        print 'Inf_prog = %s'%s
        model.ripl.infer(s)

      else:
        model.ripl.infer({"kernel":"mh", "scope":d-1,
                          "block":"one", "transitions": (Y+1)*transitions})

      logs.append( log(t,d,i+1,transitions,model,baseDirectory) )
      continue
      
      bird_locs = model.getBirdLocations(days=[d])
      
      for y in range(Y):  # save data for each year
        #path = baseDirectory+'/bird_moves%d/%d/%02d/' % (dataset, y, d)
        ensure(path)
        #drawBirds(bird_locs[y][d], path + '%02d.png' % i, **model.parameters)
        
  #model.drawBirdLocations()

  return logs, model



def posteriorSamples(model, slice_hypers=False, runs=10, baseDirectory=None, iterations=5, transitions=1000):
  
  if baseDirectory is None:
    baseDirectory = 'posteriorSamples_'+str(np.random.randint(10**4))+'/'

  infoString='''PostSamples:runs=%i,iters=%i, transitions=%s,
  time=%.3f\n'''%(runs,iterations,str(transitions),time.time())

  ensure(baseDirectory)
  with open(baseDirectory+'posteriorRuns.dat','a') as f:
    f.write(infoString)
  
  posteriorLogs = []

  for run_i in range(runs):
    logs,lastModel = run(model, slice_hypers=slice_hypers, iterations=iterations, transitions=transitions,
                         baseDirectory=baseDirectory)
    posteriorLogs.append( logs ) # list of logs for iid draws from prior
    
    with open(baseDirectory+'posteriorRuns.dat','a') as f:
      f.write('\n Run #:'+str(run_i)+'\n logs:\n'+str(logs))
  
  with open(baseDirectory+'posteriorRunsDump.py', 'w') as f:
    info = 'info="""%s"""'%infoString
    logs = '\n\nlogs=%s'%posteriorLogs
    f.write(info+logs) # dump to file

  return posteriorLogs,lastModel


def getMoves(model,slice_hypers=False, transitions=1000,iterations=1,label=''):
  
  basedir = label + 'getMoves_'+str(np.random.randint(10**4))+'/'
  print '====\n getMoves basedir:', basedir
  print '\n getMoves args:'
  print 'transitions=%s, iterations=%i'%(str(transitions),iterations)
  
  kwargs = dict(runs=1, slice_hypers=slice_hypers, iterations=iterations, transitions=transitions,
                baseDirectory=basedir)
  posteriorLogs,lastModel = posteriorSamples(model,**kwargs)
  bird_moves = model.getBirdMoves()
  bird_locs = model.getBirdLocations()

  ensure(basedir)
  with open(basedir+'moves.dat','w') as f:
    f.write('moves='+str(bird_moves))

  return posteriorLogs,model,bird_moves,bird_locs
  

def checkMoves(moves,no_days=5):
  allMoves = {}
  for day in range(no_days):
    allMoves[day] = []
    for i in range(100):
      fromi = sum( [moves[(0,day,i,j)] for j in range(100)] )
      allMoves[day].append(fromi)
      
    if day<6:
      print 'allMoves total for day %i (up to 6): %i'%(day,
                                                       sum(allMoves[day]))
  
  return allMoves



def stepThru():
  ripl.clear()
  model.loadAssumes()
  model.updateObserves(0)

  t=[time.time()]
  logs = []
  daysRange = range(1,D)

  for d in daysRange:
    print 'day %d'% d
    model.updateObserves(d)
    logs.append( log(t,d,0,10,model.ripl) )
    yield
    model.forceBirdMoves(d)
  



def loadFromPrior():
  model.loadAssumes()

  print "Predicting observes"
  observes = []
  for y in range(Y):
    for d in range(D):
      for i in range(cells):
        n = ripl.predict('(observe_birds %d %d %d)' % (y, d, i))
        observes.append((y, d, i, n))
  
  return observes

#observes = loadFromPrior()
#true_bird_moves = getBirdMoves()


#p = multiprocessing.cpu_count() / 2




def sweep(r, *args):
  t0 = time.time()
  for y in range(Y):
    r.infer("(pgibbs %d ordered %d 1)" % (y, p))
  
  t1 = time.time()
  #for y in range(Y):
    #r.infer("(mh %d one %d)" % (y, 1))
  r.infer("(mh default one %d)" % 1000)
  
  t2 = time.time()
  
  print "pgibbs: %f, mh: %f" % (t1-t0, t2-t1)



                                                          

# getMoves basedir: hypers_cfgetMoves_6881/

#  getMoves args:
# transitions=500, iterations=4

#  Starting run
# params: {'venture_random_seed': 0, 'name': '10x10x1000-train', 'hypers': ['(gamma 7 1)', '(gamma 7 1)', '(gamma 7 1)', '(gamma 7 1)'], 'cells': 100, 'days': [], 'years': [0], 'height': 10, 'width': 10, 'maxDay': 8, 'dataset': 2, 'num_birds': 1000} 

# Loading assumes

# Day 1
# [1, 0, 500, -30.239999999999998, 5185.0, array([ 7.06,  3.91,  5.85,  7.85]), 4.0800000000000001]
# Inf_prog = (cycle ((mh hypers one 10) (mh 0 one 0)) 1)
# [1, 1, 500, -23.390000000000001, 676.0, array([ 3.79,  3.91,  4.32,  7.85]), 7.46]
# Inf_prog = (cycle ((mh hypers one 10) (mh 0 one 0)) 1)
# [1, 2, 500, -23.710000000000001, 1156.0, array([ 6.14,  5.05,  8.03,  8.48]), 7.2199999999999998]
# Inf_prog = (cycle ((mh hypers one 10) (mh 0 one 0)) 1)
# [1, 3, 500, -22.579999999999998, 576.0, array([ 4.57,  5.95,  6.03,  8.48]), 7.3099999999999996]
# Inf_prog = (cycle ((mh hypers one 10) (mh 0 one 0)) 1)
# [1, 4, 500, -26.170000000000002, 256.0, array([  2.29,   2.77,  10.33,   8.48]), 7.1200000000000001]

# Day 2
# [2, 0, 500, -1942.02, 659427.0, array([  2.29,   2.77,  10.33,   8.48]), 9.6999999999999993]
# Inf_prog = (cycle ((mh hypers one 10) (mh 1 one 0)) 1)
# [2, 1, 500, -170.84, 5585.0, array([  4.79,   8.19,  10.33,   8.48]), 44.950000000000003]
# Inf_prog = (cycle ((mh hypers one 10) (mh 1 one 0)) 1)
# [2, 2, 500, -170.84, 5585.0, array([  4.79,   8.19,  10.33,   8.48]), 76.680000000000007]
# Inf_prog = (cycle ((mh hypers one 10) (mh 1 one 0)) 1)
# [2, 3, 500, -143.40000000000001, 6460.0, array([  3.8 ,   8.19,  10.33,   8.48]), 58.420000000000002]
# Inf_prog = (cycle ((mh hypers one 10) (mh 1 one 0)) 1)
# [2, 4, 500, -143.40000000000001, 6460.0, array([  3.8 ,   8.19,  10.33,   8.48]), 55.640000000000001]

# Day 3
# [3, 0, 500, -368.92000000000002, 9912.0, array([  3.8 ,   8.19,  10.33,   8.48]), 14.720000000000001]
# Inf_prog = (cycle ((mh hypers one 4) (mh 2 one 0)) 1)
# [3, 1, 500, -296.57999999999998, 8682.0, array([ 3.8 ,  8.19,  8.72,  8.48]), 116.65000000000001]
# Inf_prog = (cycle ((mh hypers one 4) (mh 2 one 0)) 1)
# [3, 2, 500, -296.57999999999998, 8682.0, array([ 3.8 ,  8.19,  8.72,  8.48]), 146.97]
# Inf_prog = (cycle ((mh hypers one 4) (mh 2 one 0)) 1)
# [3, 3, 500, -271.61000000000001, 6916.0, array([ 3.95,  8.19,  8.72,  8.48]), 127.59999999999999]
# Inf_prog = (cycle ((mh hypers one 4) (mh 2 one 0)) 1)
# [3, 4, 500, -271.61000000000001, 6916.0, array([ 3.95,  8.19,  8.72,  8.48]), 144.88]

# Day 4
# [4, 0, 500, -427.02999999999997, 6968.0, array([ 3.95,  8.19,  8.72,  8.48]), 17.100000000000001]
# Inf_prog = (cycle ((mh hypers one 3) (mh 3 one 0)) 1)
# [4, 1, 500, -427.02999999999997, 6968.0, array([ 3.95,  8.19,  8.72,  8.48]), 238.59]
# Inf_prog = (cycle ((mh hypers one 3) (mh 3 one 0)) 1)
# [4, 2, 500, -427.02999999999997, 6968.0, array([ 3.95,  8.19,  8.72,  8.48]), 241.06]
# Inf_prog = (cycle ((mh hypers one 3) (mh 3 one 0)) 1)
# [4, 3, 500, -427.02999999999997, 6968.0, array([ 3.95,  8.19,  8.72,  8.48]), 220.33000000000001]
# Inf_prog = (cycle ((mh hypers one 3) (mh 3 one 0)) 1)
# [4, 4, 500, -427.02999999999997, 6968.0, array([ 3.95,  8.19,  8.72,  8.48]), 279.44999999999999]

# Day 5
# [5, 0, 500, -593.26999999999998, 10229.0, array([ 3.95,  8.19,  8.72,  8.48]), 16.82]
# Inf_prog = (cycle ((mh hypers one 2) (mh 4 one 0)) 1)
