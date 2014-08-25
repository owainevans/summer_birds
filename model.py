from utils import *
from features_utils import make_features_dict
from venture.unit import VentureUnit
from venture.venturemagics.ip_parallel import mk_p_ripl, mk_l_ripl
from venture.ripl.utils import strip_types

from nose.tools import eq_, assert_almost_equal
from itertools import product
import matplotlib.pylab as plt
import cPickle as pickle
import numpy as np

    
#### Multinomials and Poisson Dataset Loading Functions

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



## Multinomial & Poisson functions for saving synthetic Observes and 
## loading and running unit.ripl.observe(loaded_observe)

def update_names(generated_data_params):
  
  short_name = 'infer__' + generated_data_params['short_name']
  long_name = generated_data_params['long_name'].replace('gen','infer')
  
  return short_name, long_name


def generate_data_params_to_infer_params(generate_data_params, prior_on_hypers, observes_loaded_from):

  infer_params = generate_data_params.copy()
  assert len( prior_on_hypers ) ==  infer_params['num_features']
  assert isinstance( prior_on_hypers[0], str )
  
  short_name, long_name = update_names(generate_data_params)

  # NOTE: observes_loaded_from has full path
  update_dict = {'learn_hypers':True,
                 'prior_on_hypers': prior_on_hypers,
                 'observes_loaded_from': observes_loaded_from,
                 'short_name': short_name,
                 'long_name': long_name}

  return infer_params.update


def make_infer_unit( generate_data_path_filename, prior_on_hypers, multinomial_or_poisson=True):

  with open(generate_data_path_filename,'r') as f:
    store_dict = pickle.load(f)

  generate_data_params = store_dict['generate_data_params']
  infer_params = generate_data_params_to_infer_params(generate_data_params, prior_on_hypers,
                                                      generate_data_path_filename)

  model_constructor = Multinomial if multinomial_or_poisson else Poisson
  infer_unit = model_constructor( mk_p_ripl(), generate_data_params) # FIXME, lite option

  return infer_unit
  

def example_make_infer(observe_range = None):
  generate_data_params = make_params()
  generate_data_unit = Multinomial(mk_p_ripl(),generate_data_params)

  if observe_range is None:
    observe_range = dict(  years_list = range(1),
                           days_list= range(1),
                           cells_list = None )
  
  path_filename = generate_data_unit.store_observes(observe_range)

  prior_on_hypers = ['(gamma 1 1)'] * generate_data_params['num_features']
  infer_unit = make_infer_unit( path_filename, prior_on_hypers, True)

  return observe_range, generate_data_unit, path_filename, infer_unit


def test_make_infer():
  _, generate_data_unit, path_filename, infer_unit = example_make_infer()
  generate_data_params = generate_data_unit.get_params()
  infer_params = infer_unit.get_params()

  # is infer_params mostly same as generate_data_params?
  for k,v in generate_data_params.items():
    if k not in ('ripl_directives','prior_on_hypers',
                 'learn_hypers','observes_loaded_from'):
      eq_( v, infer_params[k] )

  infer_unit.load_assumes()

  # do constants agree for generate_data_unit and infer_unit?
  expressions = ('features', 'num_birds', '(phi 0 0 0 0)')
  for exp in expressions:
    eq_( generate_data_unit.ripl.sample(exp), infer_unit.ripl.sample(exp) )


def test_load_observations():
  observe_range, generate_data_unit, path_filename, infer_unit = example_make_infer()

  infer_unit.load_observes( observe_range , path_filename)

  def sample_observe( unit,y,d,i):
    return unit.ripl.sample('(observe_birds %i %i %i)'%(y,d,i))

  ydi = product(observe_range['years_list'],
                observe_range['days_list'],
                range(infer_unit.cells))
    
  # do values for *observe_birds* agree for generate_data_unit
  # and infer_unit?
  for y,d,i in ydi:
    eq_( sample_observe( generate_data_unit, y,d,i),
         sample_observe( infer_unit, y,d,i), )

def test_incremental_load_observations():
  observe_range, generate_data_unit, path_filename, infer_unit = example_make_infer()
  
  for cell in range(infer_unit.cells):
    updated_observe_range = observe_range.copy()
    updated_observe_range.update( dict(cells_list = [cell] ) )
    infer_unit.load_observes(updated_observe_range, path_filename)
    print observe_range.update( dict(cells_list = [cell] ) )
    infer_unit.ripl.infer(10)
    print infer_unit.ripl.list_directives()[-1]
                      


def store_observes(unit, observe_range):
  
  observe_unit_pairs = ( ('years_list',unit.years),
                         ('days_list', unit.days),
                         ('cells_list', range(unit.cells) ) )

  for k,v in observe_range.items():
    unit_v = dict(observe_unit_pairs)[k]

    if v is None: 
      observe_range[k] = unit_v
    else:
      assert set(v).issubset( set(unit_v) )
  

  if not unit.assumes_loaded:
    unit.load_assumes()
  
  observe_counts={}

  ydi = product( *map( lambda k: observe_range[k],
                      ('years_list','days_list','cells_list') ) )

  for y,d,i in ydi:
    observe_counts[(y,d,i)] = unit.ripl.predict('(observe_birds %i %i %i)'%(y,d,i))
    if (y,d,i) == (0,0,0):
      print unit.ripl.predict('(observe_birds %i %i %i)'%(y,d,i))

  params = unit.get_params()
  
  store_dict = {'generate_data_params':params,
                'observe_counts':observe_counts,
                'observe_range':observe_range}
  filename = params['long_name'] + '.dat'
 
  date = '21_08_14' ## FIXME ADD DATE
  path = 'synthetic/%s/' % date
  ensure(path)
  path_filename = path + filename
  
  with open(path_filename,'w') as f:
    pickle.dump(store_dict,f)
  print 'Stored observes in %s.'%path_filename

  return path_filename ## FIXME not sure about this




def load_observes(unit, load_observe_range, path_filename=None):
 
  assert isinstance(path_filename, str)
  
  with open(path_filename,'r') as f:
     store_dict = pickle.load(f)
    
  observe_counts = store_dict['observe_counts']
  observe_range = store_dict['observe_range']

  for k,v in load_observe_range.items():
    observe_range_v = observe_range[k]
    if v is None:
      load_observe_range[k] = observe_range_v
    else:
      assert set(v).issubset( set(observe_range_v) )

  #assert observe_counts[(0,0,0)] == unit.num_birds


  def unit_observe(unit, y, d, i, count_i):
    unit.ripl.observe('(observe_birds %i %i %i)'%(y,d,i), count_i )

  ydi = product( *map( lambda k:load_observe_range[k],
                      ('years_list','days_list','cells_list') ) )

  for y,d,i in ydi:
    unit_observe(unit,y,d,i,observe_counts[(y,d,i)])
  
  


# NOTES:
# we want a function that takes filename, hyperpriors, poiss/multi and 
# outputs a unit object with the features loaded. you then can 
# incrementally load whichever observes you want from same filename (which
# should probably be a fixed param -- which is null for synthetic generator)

# then write some python functions that do inference on generic unit objects
# (by doing inference on its constituent ripl). may want to be able to save
# state at any stage, for interactive work.

# then combine these in a function that runs a full experiment. 

 # rewrote store_observes to use the long_name param which is also generated automatically in make_param with the intention of being unique. maybe we need top actually ensure uniqueness by adding some numbers to the end (we could check for duplicate names and add suffixes if necessary. good to have some syste that makes it easy to find all identical-param datasets


def make_params():  
# 'easy_hypers', currently uses 'must move exactly onestep away'
# and 'avoid diagonal', but weigths are [1,0], so diagonal does nothing.
  params = {
    'short_name': 'onestepdiag10',
    'years': range(1),
    'days': range(1),
    'height': 2,
    'width': 2,
    'feature_functions_name': 'one_step_and_not_diagonal',
    'learn_hypers': False,
    'num_birds': 2,
    'softmax_beta': 4,
    'observes_loaded_from': None,
    'venture_random_seed': 1,
    'dataset': None,
    'observes_saved_to': None }

  params['max_day'] = max( params['days'] )
  
  # Generate features dicts
  args = params['height'], params['width'], params['years'], params['days']
  kwargs = dict( feature_functions_name = params['feature_functions_name'] )
  venture_features_dict, python_features_dict = make_features_dict(*args,**kwargs)
                                                 
  params['features'] = venture_features_dict  
  params['num_features'] = len( python_features_dict[(0,0,0,0)] )
  params['prior_on_hypers'] = ['(gamma 6 1)'] * params['num_features']                                  
  params['hypers'] = [1,0,0,0][:params['num_features']]


  def make_long_name( params ):
    ## need to make long_name for inference jobs. need reference to dataset
    # we are doing inference on. but also need info about inference program
    # so that we can join identical inference progs. so defer till we 
    # have set up 'inference_params'

    keys = ('feature_functions_name','hypers',
            'height','width','num_birds','softmax_beta',
            'years','days' )

    s = []
    for k in keys:
      sub = len(params[k]) if k in ('years','days') else params[k]
      s.append( k+'-%s' % sub)
    
    s0 = 'gen__' if not params['observes_loaded_from'] else 'inf__'
    s = [s0] + s
    
    return '__'.join( s )
    
  params['long_name'] = make_long_name( params )

  # Check types
  types = dict(
               short_name = str,
               height=int,
               width=int,
               years=list,
               days=list,
               max_day=int,
               features=dict,
               num_features=int,
               hypers=list,
               learn_hypers=bool,
               prior_on_hypers=list,
               softmax_beta=int,
               venture_random_seed=int,)
  
  for k,v in types.items():
    assert isinstance(params[k],v)

  if params['observes_loaded_from']:
    assert isintance( params['observes_loaded_from'], str)

  if params['observes_saved_to']:
    assert isintance( params['observes_saved_to'], str)
    
  assert isinstance( params['hypers'][0], (int,float) )
  

  ## FIXME ADD PARAMS FOR DATASET AND FUNC FOR ADDING MORE PARAMS

  # elif params_name in ('ds2','ds3'):
  #   dataset = 2 if params_name=='ds2' else 3
  #   width,height = 10,10
  #   num_birds = 1000 if dataset == 2 else 1000000
  #   name = "%dx%dx%d-train" % (width, height, num_birds)
  #   Y,D = 1, 4
  #   years = range(Y)
  #   days = []
  #   max_day = D
  #   hypers = [5, 10, 10, 10] 
  #   num_features = len(hypers)
  #   prior_on_hypers = ['(gamma 6 1)']*num_features
  #   learn_hypers = False
  #   features = None
  #   softmax_beta = None
  #   load_observes_file = None
  #   venture_random_seed = 1


  return params


## CELL NAMES (GRID REFS)
# Multinomial Venture prog just has integer cell indices
# We only convert to ij for Python stuff that displays
# (We use ij form for synthetic data generation also
# and so that has to be converted to an index before conditioning)

class Multinomial(object):
  
  def __init__(self, ripl, params):

    self.ripl = ripl
    self.params = params
    for k,v in self.params.iteritems():
      setattr(self,k,v)

    self.cells = self.width * self.height
 
    self.assumes_loaded = False
    
    # to recreate the model, we just need the params dict
    # later on, to recreate whole thing, we'd like the serialized ripl
    # but a good alternative is just to store the ripl's list_directives in params

    # for reproduction, it's probably best to send features directly, rather than
    # generating them via the models init (want to separate the venture model
    # from the code that calls in some stored features)

    # should we have some default values for features in the init, or should 
    # we be strict about requiring all features to be entered from the outside
    # - probably better to modularize these. to have the init be clean.
    # we already need a separate script which generates params by calling feature_utils (potentially)
    # one use is generating data. for that we turn learn_hypers off. another is observed. for that
    # we need the feature_dict that generated the data (so that table needs to be easy to pull out)
    

  def get_params(self):
    self.params['ripl_directives'] = self.ripl.list_directives()
    return self.params


  def load_assumes(self):
    
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


## PARAMS FOR SINGLE AND MULTIBIRD MODEL

## FIXME incorporate *prior_on_hypers*
    if self.learn_hypers:
      ripl.assume('scale', '(scope_include (quote hypers) (quote scale) (gamma 1 1))')
      for k in range(self.num_features):
        ripl.assume('hypers%d' % k, '(scope_include (quote hypers) %d (* scale (normal 0 5) ))' % k)
    else:
      for k, value_k in enumerate(self.hypers):
        ripl.assume('hypers%d' % k, '(scope_include (quote hypers) 0 %i)'%value_k)

      
    
    ripl.assume('features', self.features)
    ripl.assume('num_birds',self.num_birds)
    
    bird_ids = ' '.join(map(str,range(self.num_birds))) # multibird only
    ripl.assume('bird_ids','(list %s)'%bird_ids) # multibird only

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
    ripl.assume('single_move', """
      (lambda (y d i)
        (let ((dist (get_bird_move_dist y d i)))
          (scope_include (quote single_move) (array y d)
            (categorical dist cell_array))))""")

# single bird is at 0 on d=0
    ripl.assume('single_get_bird_pos', """
      (mem (lambda (y d)
        (if (= d 0) 0
          (single_move y (- d 1) (single_get_bird_pos y (- d 1))))))""")

    ripl.assume('single_count_birds', """
      (lambda (y d i)
        (if (= (single_get_bird_pos y d) i) 1 0))""")

    ripl.assume('single_observe_birds', '(lambda (y d i) (poisson (+ (single_count_birds y d i) 0.0001)))')


### MULTIBIRD MODEL
    ripl.assume('move', """
      (mem (lambda (bird_id y d i)
        (let ((dist (get_bird_move_dist y d i)))
          (scope_include (quote move) d
            (categorical dist cell_array)))))""")

# all birds at 0 on d=0
    ripl.assume('get_bird_pos', """
      (mem (lambda (bird_id y d)
        (if (= d 0) 0
          (move bird_id y (- d 1) (get_bird_pos bird_id y (- d 1))))))""")

    ripl.assume('all_bird_pos',"""
       (mem (lambda (y d) 
         (map (lambda (bird_id) (get_bird_pos bird_id y d)) bird_ids)))""")

# Counts number of birds at cell i. Calls *get_bird_pos* on each bird_id
# which moves each bird via *move*. Since each bird's move from i is 
# memoized by *move*, the counts will be fixed by predict.

    ripl.assume('count_birds', """
      (lambda (y d i)
        (size (filter 
                 (lambda (bird_id) (= i (get_bird_pos bird_id y d)) )
                  bird_ids) ) ) """)

# alternative version of count_birds
    ripl.assume('count_birds_v2', """
      (mem (lambda (y d i)
        (size (filter
                (lambda (x) (= x i)) (all_bird_pos y d)))))""" )
## note that count_birds_v2 seems to work faster. haven't looked at whether it harms inference.
    ripl.assume('observe_birds', '(lambda (y d i) (poisson (+ (count_birds_v2 y d i) 0.0001)))')
    
    self.assumes_loaded = True

  
  def store_observes(self,observe_range):
    return store_observes(self, observe_range)
     

  def load_observes(self, load_observe_range, path_filename):
    return load_observes(self, load_observe_range, path_filename)


  def bird_to_pos(self,year,day,hist=False):
    'Return list [cell_index for bird_i], or optionally hist, for given day'
    l=[]
    for bird_id in self.ripl.sample('bird_ids'):
      args = bird_id, year, day
      l.append(self.ripl.predict('(get_bird_pos %i %i %i)'%args))
                                                           
    all_bird_l = self.ripl.predict('(all_bird_pos %i %i)'%(year,day))
    assert all( np.array(all_bird_l)==np.array(l) )
    ## Check that get_bird_pos and all_bird_pos agree (Could turn off 
    # for speed)

# How np.histogram works:
# np.histogram([0,1,2],bins=np.arange(0,3)) == ar[1,2], ar[0,1,2]
# np.histogram([0,1,2],bins=np.arange(0,4)) == ar[1,1,1], ar[0,1,2,3]
    if hist:
      hist,_ = np.histogram(l,bins=range(self.cells+1))
      assert len(hist)==self.cells
      assert np.sum(hist) == self.num_birds
      return hist
    else:
      return l



  def get_bird_locations(self, years=None, days=None):
    '''Returns dict { y: { d:histogram of bird positions on y,d}  }
       for y,d in product(years,days) '''
    if years is None: years = self.years
    if days is None: days = self.days
    
    bird_locations = {}
    for y in years:
      bird_locations[y] = {}
      for d in days:
        bird_locations[y][d] = self.bird_to_pos(y,d,hist=True)
    
    return bird_locations


  def draw_bird_locations(self, years, days, name=None, plot=True, save=True, order='F',
                          print_features_info=True):

    assert isinstance(years,list)
    assert isinstance(days,list)
    assert order in ('F','C')
    name = self.name if name is None else name
    bird_locs = self.get_bird_locations(years,days)


    bitmaps = plot_save_bird_locations(bird_locs, name, years, days, self.height, self.width,
                                   plot=plot, save=save, order=order,
                                   print_features_info = print_features_info)
    
    if print_features_info:
      features_dict = venturedict_to_pythondict(self.features)
      assert len(features_dict) == (self.height*self.width)**2 * (len(self.years) * len(self.days))

      print '\n Features dict (up to 10th entry) for year,day = 0,0'
      count = 0
      for k,v in features_dict.iteritems():
        if k[0]==0 and k[1]==0 and count<10: 
          print k[2:4],':',v
          assert isinstance(v[0],(int,float))
          count += 1
          
      feature0_from0 = [features_dict[(0,0,0,j)][0] for j in range(self.cells)]

      print '\n feature_0 for jth cell for y,d,i = (0,0,0), order=%s, 0 at top \n'%order
      print make_grid( self.height, self.width, lst=feature0_from0, order=order)
      
    return bitmaps

    
# # loadObserves for multinomial dataset (prepare for pgibbs inference)
#   def loadObserves(self, ripl = None):
#     if ripl is None:
#       ripl = self.ripl
#     ## FIXME STUFF FOR DATASET LOADING
#     #observations_file = "data/input/dataset%d/%s-observations.csv" % (1, self.name)
#     #observations = readObservations(observations_file)

#     self.unconstrained = []

#     for y in self.years:
#       for (d, ns) in observations[y]:
#         if d not in self.days: continue
#         if d == 0: continue
        
#         loc = None
        
#         for i, n in enumerate(ns):
#           if n > 0:
#             loc = i
#             break
        
#         if loc is None:
#           self.unconstrained.append((y, d-1))
#           #ripl.predict('(get_bird_pos %d %d)' % (y, d))
#         else:
#           ripl.observe('(get_bird_pos %d %d)' % (y, d), loc)
  
#   def inferMove(self, ripl = None):
#     if ripl is None:
#       ripl = self.ripl
    
#     for block in self.unconstrained:
#       ripl.infer({'kernel': 'gibbs', 'scope': 'move', 'block': block, 'transitions': 1})
  




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
    self.maxDay = params.get('maxDay',None)
    for attr in self.years,self.days:
      assert isinstance( attr[0], (float,int) )

    self.hypers = params["hypers"]
    self.prior_on_hypers = params['prior_on_hypers']
    assert isinstance(self.prior_on_hypers[0],str)
    self.learn_hypers = params['learn_hypers']
    
    self.ground = readReconstruction(params) if 'ground' in params else None
    

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
      for k, k_value in enumerate(self.hypers):
        ripl.assume('hypers%d' % k, '(scope_include (quote hypers) %i %f )'%(k, k_value) )
    else:
      for k, k_prior in enumerate(self.prior_on_hypers):
        ripl.assume('hypers%d' % k, '(scope_include (quote hypers) %i %s )'%(k, k_prior) )

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
    observes = observe_from_file(self,years_range,days_range,filename,no_observe_directives)
    # assume we always have unbroken sequence of days
    self.days = range(max(days_range))

    string = 'observes_from_file: days_range, self.days %s %s'%(days_range,self.days)
    return observes, string

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
      bitmaps = plot_save_bird_locations(bird_locs, self.name, years, days,
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













