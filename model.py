from utils import *
from venture.unit import VentureUnit
from venture.ripl.ripl import _strip_types
from itertools import product
import matplotlib.pylab as plt
import cPickle as pickle
import numpy as np
num_features = 4
    
#### OneBirds and Poisson Dataset Loading Functions

def day_features(features,width,y=0,d=0):
  'Python dict of features to features vals for the day'
  lst = [features[(y,d,i,j)] for (i,j) in product(range(cells),range(cells))]
  return lst

def loadFeatures(dataset, name, years, days, maxDay=None):
  'Load features from Birds datasets and convert to Venture dict'
  features_file = "data/input/dataset%d/%s-features.csv" % (dataset, name)
  print "Loading features from %s" % features_file  
  features = readFeatures(features_file, maxYear= max(years)+1, maxDay=maxDay)
  
  for (y, d, i, j) in features.keys():
    if y not in years:
      del features[(y, d, i, j)]
  
  return toVenture(features)


def loadObservations(ripl, dataset, name, years, days):
  'Load observations from Birds dataset'
  observations_file = "data/input/dataset%d/%s-observations.csv" % (dataset, name)
  observations = readObservations(observations_file)

  for y in years:
    for (d, ns) in observations[y]:
      if d not in days: continue
      for i, n in enumerate(ns):
        ripl.observe('(observe_birds %d %d %d)' % (y, d, i), n)





## OneBird & Poisson functions for saving synthetic Observes and 
## loading and running unit.ripl.observe(loaded_observe)


# TODO: less hackish way of storing in unique file/directory for parallel runs
def store_observes(unit,years=None,days=None):
  if years is None: years = unit.years
  if days is None: days = unit.days

  observed_counts={}
  for y in years:
    for d in days:
      counts  = []
      for i in range(unit.cells):
        counts.append( unit.ripl.predict('(observe_birds %i %i %i)'%(y,d,i)) )
      observed_counts[(y,d)] = counts

  path = 'synthetic/%s/%s'%(unit.name, str(np.random.randint(10**9))) ## TODO random dir name
  ensure(path)
  filename = path + 'observes.dat'
  with open(filename,'w') as f:
    pickle.dump(observed_counts,f)
  print 'Stored observes in %s.'%filename
  return filename


def observe_from_file(unit,years_range,days_range,filename=None, no_observe_directives=False):
  if filename is None: # uses attribute if no filename arg given
    filename = unit.observed_counts_filename
  assert isinstance(filename,str)
  with open(filename,'r') as f:
    unit.observed_counts = pickle.load(f)

  assert len( unit.observed_counts[(0,0)] ) == unit.cells

  observes = []
  
  for y in years_range:
    for d in days_range:
      for i,bird_count_i in enumerate(unit.observed_counts[(y,d)]):
        
        obs_tuple = ('(observe_birds %i %i %i)'%(y,d,i), bird_count_i )
        if not no_observe_directives:
          unit.ripl.observe( *obs_tuple )
        observes.append( obs_tuple)

  return observes



## Function for plotting/saving bird images

# calls vlad's *drawBirds* from utils.py for saving 
# and my *make_grid* from utils.py for plotting
# (duplicating same reshape operation)

# TODO inefficiently calls bird_locs twice and with fixed Fortran order
# note that saved images use different order
def drawBirdLocations(bird_locs, name, years, days, height, width,
                      save=True,plot=None,order='F'):
  if save:
    for y in years:
      path = 'bird_moves_%s/%d/' % (name, y)
      ensure(path)
      for d in days:
        drawBirds(bird_locs[y][d], path+'%02d.png'%d, height=height, width=width)

  if plot:
    nrows=len(days)
    ncols=len(years)
    fig,ax = plt.subplots(nrows,ncols,figsize=(4*ncols,2*nrows))
    for y,d in product(years,days):
      im = make_grid(height, width, lst=bird_locs[y][d], order=order)
      ax_dy = ax[d] if len(ax.shape)==1 else ax[d][y]
      ax_dy.imshow(im,cmap=plt.cm.Reds, interpolation='none',
                      extent=[0,width,0,height])
      ax_dy.set_title('%s- y:%i d:%i'%(name,y,d))
      ax_dy.set_xticks(range(width+1))
      ax_dy.set_yticks(range(height+1))
    fig.tight_layout()

  return fig if plot else None



class OneBird(VentureUnit):
  
  def __init__(self, ripl, params):
    self.name = params['name']
    self.width = params['width']
    self.height = params['height']
    self.cells = self.width * self.height
    assert isinstance(self.cells,int) and self.cells > 1
    
    self.years = params['years']
    assert isinstance(self.years, list)
    self.days = params['days']
    assert isinstance(self.days, list)

    if 'features' in params:
      self.features = params['features']
      self.num_features = params['num_features']
    else:
      self.features = loadFeatures(1, self.name, self.years, self.days)
      self.num_features = num_features

    self.learn_hypers = params['learn_hypers']
    if not self.learn_hypers:
      self.hypers = params['hypers']
      
    self.load_observes_file=params.get('load_observes_file',True)
    self.num_birds=params.get('num_birds',1)

    self.softmax_beta=params.get('softmax_beta',1)
    self.observed_counts_filename = params.get('observed_counts_filename',None)
    
    super(OneBird, self).__init__(ripl, params)



  def makeAssumes(self):
    self.loadAssumes(self)
    # ripl arg for *loadAssumes* =self so we use VUnit's *assume* method and mutate
    # the unit object's *assumes* attribute. this way, when we call *getAnalytics*
    # all the assumes are sent in the Kwargs. 
  
  def makeObserves(self):
    if self.load_observes_file:
      self.loadObserves(ripl=self)
    else:
      pass
  
  def loadAssumes(self, ripl = None): ## see point under *makeAssumes*
    if ripl is None:  # allows loading of assumes on an alternative ripl
      ripl = self.ripl
    
    print "Loading assumes"

## UTILITY FUNCTIONS
    ripl.assume('filter',"""
      (lambda (pred lst)
        (if (not (is_pair lst)) (list)
          (if (pred (first lst))
            (pair (first lst) (filter pred (rest lst)) )
            (filter pred (rest lst)))))""")
    ripl.assume('map',"""
      (lambda (f lst)
        (if (not (is_pair lst)) (list)
          (pair (f (first lst)) (map f (rest lst))) ) )""")



## PARAMS FOR SINGLE / MULTIBIRD MODEL
    if not self.learn_hypers:
      for k, value_k in enumerate(self.hypers):
        ripl.assume('hypers%d' % k, '(scope_include (quote hypers) 0 %i)'%value_k)
    else:
      ripl.assume('scale', '(scope_include (quote hypers) (quote scale) (gamma 1 1))')
      for k in range(self.num_features):
        ripl.assume('hypers%d' % k, '(scope_include (quote hypers) %d (* scale (normal 0 5) ))' % k)

    
    ripl.assume('features', self.features)
    ripl.assume('num_birds',self.num_birds)
    
    ## ONLY NEEDED FOR MULTIBIRD MODEL
    bird_ids = ' '.join(map(str,range(self.num_birds)))
    ripl.assume('bird_ids','(list %s)'%bird_ids)   

    ripl.assume('softmax_beta',self.softmax_beta)

    ripl.assume('phi', """
      (mem (lambda (y d i j)
        (let ((fs (lookup features (array y d i j))))
          (exp (* softmax_beta %s)))))"""
       % fold('+', '(* hypers_k_ (lookup fs _k_))', '_k_', self.num_features))

    ripl.assume('get_bird_move_dist',
      '(mem (lambda (y d i) ' +
        fold('simplex', '(phi y d i j)', 'j', self.cells) +
      '))')
    
    ripl.assume('cell_array', fold('array', 'j', 'j', self.cells))


#### SINGLE BIRD MODEL
    ripl.assume('move', """
      (lambda (y d i)
        (let ((dist (get_bird_move_dist y d i)))
          (scope_include (quote move) (array y d)
            (categorical dist cell_array))))""")

# for any day=0, bird starts at 0
    ripl.assume('get_bird_pos', """
      (mem (lambda (y d)
        (if (= d 0) 0
          (move y (- d 1) (get_bird_pos y (- d 1))))))""")

    ripl.assume('count_birds', """
      (lambda (y d i)
        (if (= (get_bird_pos y d) i) 1 0))""")

    ripl.assume('observe_birds_old', '(lambda (y d i) (poisson (+ (count_birds y d i) 0.0001)))')



### MULTIBIRD MODEL
### ALL FUNCTIONS END IN *2*

## TODO scope include could be fixed to be an array as above
## if so, we need inf_progs to be dicts
    ripl.assume('move2', """
      (mem (lambda (bird_id y d i)
        (let ((dist (get_bird_move_dist y d i)))
          (scope_include (quote move2) d
            (categorical dist cell_array)))))""")

    ripl.assume('get_bird_pos2', """
      (mem (lambda (bird_id y d)
        (if (= d 0) 0
          (move2 bird_id y (- d 1) (get_bird_pos2 bird_id y (- d 1))))))""")

    ripl.assume('all_bird_pos2',"""
       (mem (lambda (y d) 
         (map (lambda (bird_id) (get_bird_pos2 bird_id y d)) bird_ids)))""")

# count birds at i. calls get_bird_pos2 which calls move2 which moves
# birds. since moves are all memoized, results here will be derviatively 
# memoized
    ripl.assume('count_birds2', """
      (lambda (y d i)
        (size (filter 
                 (lambda (bird_id) (= i (get_bird_pos2 bird_id y d)) )
                  bird_ids) ) ) """)

    ripl.assume('count_birds22', """
      (mem (lambda (y d i)
        (size (filter
                (lambda (x) (= x i)) (all_bird_pos2 y d)))))""" )
## note that count_22 seems to work faster. haven't looked at whether it harms inference. doesn't obviously do so
    ripl.assume('observe_birds', '(lambda (y d i) (poisson (+ (count_birds22 y d i) 0.0001)))')

  

  def store_observes(self,years=None,days=None):
    return store_observes(self,years,days)
     

  def observe_from_file(self, years_range, days_range,filename=None,no_observe_directives=False):
    return observe_from_file(self,years_range,days_range,filename,no_observe_directives)


  def bird_to_pos(self,year,day,hist=False):
    'Return list of bird positions (or optionally hist) for given day'
    l=[]
    for bird_id in self.ripl.sample('bird_ids'):
      args = bird_id, year, day
      l.append(self.ripl.predict('(get_bird_pos2 %i %i %i)'%args))
                                                           
    all_bird_l = self.ripl.predict('(all_bird_pos2 %i %i)'%(year,day))
    assert all( np.array(all_bird_l)==np.array(l) )
    ## Check that get_bird_pos2 and all_bird_pos2 agree (Could turn off 
    # for speed)

# How np.histogram works:
# np.histogram([0,1,2],bins=np.arange(0,3)) == ar[1,2], ar[0,1,2]
# np.histogram([0,1,2],bins=np.arange(0,4)) == ar[1,1,1], ar[0,1,2,3]
    if hist:
      hist,_ = np.histogram(l,bins=range(self.cells+1))
      assert len(hist)==self.cells
      return hist
    else:
      return l


  def getBirdLocations(self, years=None, days=None):
    if years is None: years = self.years
    if days is None: days = self.days
    
    bird_locations = {}
    for y in years:
      bird_locations[y] = {}
      for d in days:
        bird_locations[y][d] = self.bird_to_pos(y,d,hist=True)
    
    return bird_locations


  def draw_bird_locations(self,years,days,name=None,plot=True,save=True):
    assert isinstance(years,(list,tuple))
    assert isinstance(days,(list,tuple))
    name = self.name if name is None else name
    bird_locs = self.getBirdLocations(years,days)
    bitmaps = drawBirdLocations(bird_locs, self.name, years, days,
                                self.height, self.width, plot=plot,save=save)
    return bitmaps

    
# loadObserves for onebird dataset (prepare for pgibbs inference)
  def loadObserves(self, ripl = None):
    if ripl is None:
      ripl = self.ripl
  
    observations_file = "data/input/dataset%d/%s-observations.csv" % (1, self.name)
    observations = readObservations(observations_file)

    self.unconstrained = []

    for y in self.years:
      for (d, ns) in observations[y]:
        if d not in self.days: continue
        if d == 0: continue
        
        loc = None
        
        for i, n in enumerate(ns):
          if n > 0:
            loc = i
            break
        
        if loc is None:
          self.unconstrained.append((y, d-1))
          #ripl.predict('(get_bird_pos %d %d)' % (y, d))
        else:
          ripl.observe('(get_bird_pos %d %d)' % (y, d), loc)
  
  def inferMove(self, ripl = None):
    if ripl is None:
      ripl = self.ripl
    
    for block in self.unconstrained:
      ripl.infer({'kernel': 'gibbs', 'scope': 'move', 'block': block, 'transitions': 1})
  




## TODO
# We have been making Poisson compatible 
# with synthetically generated dataset inference
# and the functions for this inference in 
# synthetic.py and elsewhere. But Poisson
# is not compatible yet and may be buggy. 

# TODO: we haven't got store_observes or load_observe
# methods, and we haven't got ability to plot birds (in utils near top)

class Poisson(VentureUnit):

  def __init__(self, ripl, params):
    self.name = params['name']
    self.width = params['width']
    self.height = params['height']
    self.cells = self.width * self.height
    assert isinstance(self.cells,int) and self.cells > 1
    
    self.dataset = params.get('dataset',None)
    self.num_birds = params['num_birds']
    self.years = params['years']
    self.days = params['days']

    self.hypers = params["hypers"]
    self.hypers_prior = params.get('hypers_prior',None)
    if self.hypers_prior:
      assert isinstance(self.hypers_prior[0],str)
      
    self.learn_hypers = params.get('learn_hypers',None)
    assert not(self.learn_hypers and not(self.hypers_prior))
    
    self.ground = readReconstruction(params) if 'ground' in params else None
    self.maxDay = params.get('maxDay',None)

    if self.dataset in (2,3):
      self.features = loadFeatures(self.dataset, self.name, self.years, self.days,
                                   maxDay = self.maxDay)
    else:
      self.features = params['features']

    self.num_features = params['num_features']
    self.softmax_beta=params.get('softmax_beta',1)
    self.load_observes_file=params.get('load_observes_file',True)

    val_features = self.features['value']
    self.parsedFeatures = {k:_strip_types(v) for k,v in val_features.items() }

    super(Poisson, self).__init__(ripl, params)




  def feat_i(y,d,i,feat=2):
    'Input *feat in range(3) (default=wind), return all values i,j for fixed i'
    return [self.parsedFeatures[(y,d,i,j)][feat] for j in range(100)] 


# UNIT OVERRIDE METHODS
  def makeObserves(self):
    if self.load_observes_file:
      self.loadObserves(ripl=self)
    else:
      pass

  def loadObserves(self, ripl = None):
    if ripl is None:
      ripl = self.ripl
    
    print "Loading observations"
    loadObservations(ripl, self.dataset, self.name, self.years, self.days)


# Unit overriden methods
  def makeAssumes(self):
    self.loadAssumes(self)
# assumes are stored but not loaded onto ripl (need to run loadAssumes with no args for that)
# if *getAnalytics* method is called on unit object, Analytics obj will have these *assumes*
# as its self.assumes method. 

  def loadAssumes(self, ripl = None):
    if ripl is None:
      ripl = self.ripl
      print "Loading assumes on self.ripl"
    
    ripl.assume('num_birds', self.num_birds)
    ripl.assume('cells', self.cells)

    
    if not self.learn_hypers:
      for k, b in enumerate(self.hypers):
        ripl.assume('hypers%d' % k,  b)
    else:
      for k, prior in enumerate(self.hypers_prior):
        ripl.assume('hypers%d' % k,'(scope_include (quote hypers) %i %s )'%(k,prior) )

    ripl.assume('features', self.features)

    ripl.assume('width', self.width)
    ripl.assume('height', self.height)
    ripl.assume('max_dist2', '18')

    ripl.assume('cell2X', '(lambda (cell) (int_div cell height))')
    ripl.assume('cell2Y', '(lambda (cell) (int_mod cell height))')
    #ripl.assume('cell2P', '(lambda (cell) (make_pair (cell2X cell) (cell2Y cell)))')
    ripl.assume('XY2cell', '(lambda (x y) (+ (* height x) y))')

    ripl.assume('square', '(lambda (x) (* x x))')

    ripl.assume('dist2', """
      (lambda (x1 y1 x2 y2)
        (+ (square (- x1 x2)) (square (- y1 y2))))""")

    ripl.assume('cell_dist2', """
      (lambda (i j)
        (dist2
          (cell2X i) (cell2Y i)
          (cell2X j) (cell2Y j)))""")
    
    # phi is the unnormalized probability of a bird moving from cell i to cell j on day d
    ripl.assume('phi', """
      (mem (lambda (y d i j)
        (if (> (cell_dist2 i j) max_dist2) 0
          (let ((fs (lookup features (array y d i j))))
            (exp %s)))))"""
            % fold('+', '(* hypers__k (lookup fs __k))', '__k', self.num_features))
    

    ripl.assume('get_bird_move_dist', """
      (lambda (y d i)
        (lambda (j)
          (phi y d i j)))""")
    
    ripl.assume('foldl', """
      (lambda (op x min max f)
        (if (= min max) x
          (foldl op (op x (f min)) (+ min 1) max f)))""")

## note used anywhere. presumably a multinomial (vs. poisson)
# but not sure it exactly implement multinomial
    ripl.assume('multinomial_func', """
      (lambda (n min max f)
        (let ((normalize (foldl + 0 min max f)))
          (mem (lambda (i)
            (poisson (* n (/ (f i) normalize)))))))""")
                  
    ripl.assume('count_birds', """
      (mem (lambda (y d i)
        (if (= d 0) (if (= i 0) num_birds 0)""" +
          fold('+', '(get_birds_moving y (- d 1) __j i)', '__j', self.cells) + ")))")
    

    # bird_movements_loc
    # if zero birds at i, no movement to any j from i
    # *normalize* is normalizing constant for probms from i
    # n = birdcount_i * normed prob (i,j)
    # return: (lambda (j) (poisson n) )
  
    ripl.assume('bird_movements_loc', """
      (mem (lambda (y d i)
        (if (= (count_birds y d i) 0)
          (lambda (j) 0)
          (let ((normalize (foldl + 0 0 cells (lambda (j) (phi y d i j)))))
            (mem (lambda (j)
              (if (= (phi y d i j) 0) 0
                (let ((n (* (count_birds y d i) (/ (phi y d i j) normalize))))
                  (scope_include d (array y d i j)
                    (poisson n))))))))))""")

    #ripl.assume('bird_movements', '(mem (lambda (y d) %s))' % fold('array', '(bird_movements_loc y d __i)', '__i', self.cells))
    
    ripl.assume('observe_birds', '(mem (lambda (y d i) (poisson (+ (count_birds y d i) 0.0001))))')

    # returns number birds from i,j (we want to force this value)
    ripl.assume('get_birds_moving', """
      (lambda (y d i j)
        ((bird_movements_loc y d i) j))""")
    
    ripl.assume('get_birds_moving1', '(lambda (y d i) %s)' % fold('array', '(get_birds_moving y d i __j)', '__j', self.cells))
    ripl.assume('get_birds_moving2', '(lambda (y d) %s)' % fold('array', '(get_birds_moving1 y d __i)', '__i', self.cells))
    ripl.assume('get_birds_moving3', '(lambda (d) %s)' % fold('array', '(get_birds_moving2 __y d)', '__y', len(self.years)))
    ripl.assume('get_birds_moving4', '(lambda () %s)' % fold('array', '(get_birds_moving3 __d)', '__d', len(self.days)-1))

  
  def store_observes(self,years=None,days=None):
    return store_observes(self,years,days)

  def observe_from_file(self, years_range, days_range,filename=None,no_observe_directives=False):
    return observe_from_file(self,years_range,days_range,filename,no_observe_directives)

  def loadModel(self, ripl = None):
    if ripl is None:
      ripl = self.ripl
    self.loadAssumes(ripl)
    self.loadObserves(ripl)
  
  def updateObserves(self, d):
    self.days.append(d)
    #if d > 0: self.ripl.forget('bird_moves')
    loadObservations(self.ripl, self.dataset, self.name, self.years, [d])
    #self.ripl.infer('(incorporate)')
    #self.ripl.predict(fold('array', '(get_birds_moving3 __d)', '__d', len(self.days)-1), label='bird_moves')
  

## TODO Should this be sample or predict
  def getBirdLocations(self, years=None, days=None, predict=False):
    if years is None: years = self.years
    if days is None: days = self.days
    
    bird_locations = {}
    for y in years:
      bird_locations[y] = {}
      for d in days:
        r = self.ripl
        predict_sample = lambda s:r.predict(s) if predict else lambda s:r.sample(s)
        bird_locations[y][d] = [predict_sample('(count_birds %d %d %d)' % (y, d, i)) for i in range(self.cells)]
    
    return bird_locations
  

  def drawBirdLocations(self):
    bird_locs = self.getBirdLocations()
  
    for y in self.years:
      path = 'bird_moves%d/%d/' % (self.dataset, y)
      ensure(path)
      for d in self.days:
        drawBirds(bird_locs[y][d], path + '%02d.png' % d, **self.parameters)
  
## NOTE careful of PREDICT for getBirdLocations if doing inference
  def draw_bird_locations(self,years,days,name=None,plot=True,save=True):
      assert isinstance(years,(list,tuple))
      assert isinstance(days,(list,tuple))
      assert not max(days) > self.maxDay
      name = self.name if name is None else name
      bird_locs = self.getBirdLocations(years,days,predict=True)
      bitmaps = drawBirdLocations(bird_locs, self.name, years, days,
                                  self.height, self.width, plot=plot,save=save)
      return bitmaps


  def getBirdMoves(self):
    
    bird_moves = {}
    
    for d in self.days[:-1]:
      bird_moves_raw = self.ripl.sample('(get_birds_moving3 %d)' % d)
      for y in self.years:
        for i in range(self.cells):
          for j in range(self.cells):
            bird_moves[(y, d, i, j)] = bird_moves_raw[y][i][j]
    
    return bird_moves

    
  def forceBirdMoves(self,d,cell_limit=100):
    # currently ignore including years also
    #detvalues = 0
    
    for i in range(self.cells)[:cell_limit]:
      for j in range(self.cells)[:cell_limit]:
        ground = self.ground[(0,d,i,j)]
        current = self.ripl.sample('(get_birds_moving 0 %d %d %d)'%(d,i,j))
        
        if ground>0 and current>0:
          self.ripl.force('(get_birds_moving 0 %d %d %d)'%(d,i,j),ground)
          print 'force: moving(0 %d %d %d) from %f to %f'%(d,i,j,current,ground)
          
    #     try:
    #       self.ripl.force('(get_birds_moving 0 %d %d %d)'%(d,i,j),
    #                       self.ground[(0,d,i,j)] )
    #     except:
    #       detvalues += 1
    # print 'detvalues total =  %d'%detvalues

  def computeScoreDay(self, d):
    bird_moves = self.ripl.sample('(get_birds_moving3 %d)' % d)
    score = 0
    
    for y in self.years:
      for i in range(self.cells):
        for j in range(self.cells):
          score += (bird_moves[y][i][j] - self.ground[(y, d, i, j)]) ** 2
    
    return score
  
  def computeScore(self):
    infer_bird_moves = self.getBirdMoves()
    score = 0
    
    for key in infer_bird_moves:
      score += (infer_bird_moves[key] - self.ground[key]) ** 2

    return score













