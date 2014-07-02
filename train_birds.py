import time
from venture.venturemagics.ip_parallel import mk_p_ripl, MRipl
from utils import *
from model import Poisson, num_features


## NOTE
# most of this has not been touched since moving to summer_birds repo
# however, we started adapting some parts of it (like params in makeModel)

def makeModel(dataset=2, D=4, Y=1, learn_hypers=True, hypers_prior='(gamma 6 1)', toy=False):
  width,height = 10,10
  num_birds = 1000 if dataset == 2 else 1000000
  name = "%dx%dx%d-train" % (width, height, num_birds)
  hypers = [5, 10, 10, 10] 
  num_features = 4
  hypers_prior = [hypers_prior]*num_features

  
  params = {
  "name":name,
  "width":width,
  "height":height,
  "dataset":dataset,
  "num_birds":num_birds,
  "years":range(Y),
  "days":[],
  "hypers":hypers,
  "learn_hypers": learn_hypers,
  "hypers_prior": hypers_prior,
  "maxDay":D,
  "ground":True,
  "num_features":num_features}

  ripl = mk_p_ripl()
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
        


def run(model,iterations=1, transitions=(100,50,50), baseDirectory=''):
  day1,day2 = transitions[1],transitions[2]
  transitions = transitions[0]

  assert model.days == []
    
  D = model.maxDay
  Y = max(model.years)
  dataset = model.dataset
  learn_hypers = model.learn_hypers
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
    model.updateObserves(d)  # model.days.append(d)
    logs.append( log(t,d,0,transitions,model,baseDirectory) )
    
    for i in range(iterations): # iterate inference (could reduce from 5)

      if learn_hypers:
        args = (dayToHypers[d-1], d-1, (Y+1)*transitions)
        s='(cycle ((mh hypers all %d) (mh %d one %d)) 1)'%args
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
      print 'allMoves total for day %i (up to 6): %i'%(day,sum(allMoves[day]))
  
  return allMoves


##EDIT TO NOT MUTATE RIPL
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
