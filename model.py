from utils import *
from features_utils import make_features_dict, load_features
import unit_parameter_dicts
from venture.shortcuts import make_puma_church_prime_ripl as mk_p_ripl

from itertools import product
import cPickle as pickle
import numpy as np


### PLAN NOTES


# TODO
# get inference up and running and have a basic sanity test for inference
# with small grid and small number of birds. elaborate inference programs
# later. 


## NOTE we are using PREDICT for getting locations. Careful of predicts piling up. 

# verbosity
# should probably give everything a verbose mode which is off
# by default. atm just storing produces lots of printing. 
# still want it to be easy to make various things verbose
# think about how to architect such things.

# 3. Pull out certain params to be controlled from experiment runner.
# Place to store synthetic data.
# Optionally place to get
# stored data. 
# Place to save inference results. 
# Optional params for synth data generation.
# Which venture backend to use. IMP FIXME
# Params for hyper_prior and inference Venture model.
# Params (i.e. inference progs) for inference itself.
# Maybe some params for how many repeats / parallel runs to do.

# 4. Add more realistic features. THings like wind-direction

#  5. play with experiment runner and see if there's any issues with integrating. 
#     some functionality that it doesn't have (scaling/timing info for particular
#     elements of inference, e.g. loading vs. single step of MH, etc.) might be 
#     necessary to write into birds using the timeit module. 



# ASK AXCH, VLAD, ANTHONY
# 1. better way of serializing figures (so you can unpickle and display the figure in figure window). ideally
# we would just pickle one table containing observes, params, unit.ripl and figures. then loading is simply
# one unpickling. anthony: can we use save/load methods to define a pickling method for ripls?

# 2. use of dates for naming, use of long_names. what symbols are ok. should we try compress long names.
# long_name vs. short_name. should be sub-divide the synthetic params dict more. 

# probably should also think about what the inference params should look like. need a template
# for parameterizing inference programs of the kind we want to run. 

# 3. conventions for directories vs. filenames (for store_observes, save_images)

# 4. ask about best way of pulling out params. current idea is to use no global variables at all 
# in all utilities for manipulating the venture prog. we then control via a script which defines
# a 'global' set of params (backend, params for synth data or filename, inference_prog_params).
# this is different from what dan lovell did for slam and vlad did. so what is best way here?




                                                      
                                        
#### Multinomials and Poisson Dataset Loading Functions

def day_features(features,width,y=0,d=0):
  'Python dict of features to features vals for the day'
  lst = [features[(y,d,i,j)] for (i,j) in product(range(cells),range(cells))]
  return lst


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





# Store observes in file:
# filename determined by *long_name* unit attribute
# *observe_range* defaults to max range
def store_observes(unit, observe_range=None, synthetic_directory = 'synthetic'):

  unit.ensure_assumes()  # assumes need to be loaded to get observes
  
  max_observe_range = unit.get_max_observe_range();
  if observe_range is None:
    observe_range = max_observe_range
  else:
    assert isinstance(observe_range,Observe_range)
    observe_range.replace_none_super_range(max_observe_range)
    observe_range.assert_is_observe_sub_range(max_observe_range)
  

  # GET OBSERVE VALUES FROM MODEL

  # Observed (noisy) and ground-truth bird counts
  observe_counts={}
  gtruth_counts={}
  
  # alternate way to get gtruth counts
  bird_locs = unit.get_bird_locations( observe_range['years_list'],
                                       observe_range['days_list'])

  year_day_cell_triples = product( *map( lambda k: observe_range[k],
                                         ('years_list','days_list','cells_list') ) )
  
  for y,d,i in year_day_cell_triples:

    gtruth_counts[(y,d,i)] = unit.ripl.predict('(count_birds_v2 %i %i %i)'%(y,d,i))
    observe_counts[(y,d,i)] = unit.ripl.predict('(observe_birds %i %i %i)'%(y,d,i))
    
    # (observe_birds y d i) = Poisson(epsilon) if count for (y,d,i)=0. See Venture model. 
    if gtruth_counts[(y,d,i)] == 0 and observe_counts[(y,d,i)] > 0:
      assert False, 'gtruth_counts[%s] == 0 and observe_counts[%s] > 0' % str((y,d,i) )

    # Check that *get_bird_locations* matches results from using *predict* directly
    # to get ground-truth bird counts      
    assert int( gtruth_counts[(y,d,i)] )==int( bird_locs[y][d][i] )
  


  # PICKLE AND STORE COUNTS

  # Generate filenames based on *unit.parameters.long_name*.
  # Generate bird locations images (FIXME possibly remove).
  params = unit.get_params()
  date = '21_08_14' ## FIXME ADD DATE
  full_directory = '%s/%s/' % (synthetic_directory,date)
  ensure(full_directory)
  store_dict_filename = full_directory + params['long_name'] + '.dat'
  draw_bird_filename =  full_directory + params['long_name'] + '.png'

  title = 'fig for: unit.short_name'
  fig_ax = plot_save_bird_locations(unit,
                                    title,
                                    observe_range['years_list'],
                                    observe_range['days_list'],
                                    plot=True,
                                    save=True,
                                    order='F',
                                    verbose=True,
                                    directory_filename = (full_directory,
                                                          draw_bird_filename) )


  # Build dict of parameters, *observe_range* and counts,
  # along with groundtruth *bird_locs*.
  # Pickle this to file.

  store_dict = {'generate_data_params':params,
                'observe_counts':observe_counts,
                'observe_range':observe_range.copy_dict_only(), ## store() FIXME
                'bird_locs':bird_locs}
                #'bird_locs_fig_ax':fig_ax} ## FIXME serialize figure!

  with open(store_dict_filename,'w') as f:
    pickle.dump(store_dict,f)
  print 'Stored observes in %s.'% store_dict_filename

  ## return filename for use running inference on synthetic data.
  return store_dict_filename, draw_bird_filename 





def load_observes(unit, load_observe_range,
                  use_range_defaults, store_dict_filename):

  unit.ensure_assumes()
  
  with open(store_dict_filename,'r') as f:
     store_dict = pickle.load(f)
    
  observe_counts = store_dict['observe_counts']
  observe_range = Observe_range(**store_dict['observe_range']) ## since we pickle dict, not instance

  if use_range_defaults:
    load_observe_range = observe_range
  else:
    load_observe_range.assert_is_observe_sub_range(observe_range)


  def unit_observe(unit, y, d, i, count_i):
    unit.ripl.observe('(observe_birds %i %i %i)'%(y,d,i), count_i )

  year_day_cell_triples = product( *map( lambda k:load_observe_range[k],
                                         ('years_list','days_list','cells_list') ) )

  for y,d,i in year_day_cell_triples:
    unit_observe(unit,y,d,i,observe_counts[(y,d,i)])
  
  print 'Loaded all observes'



  


#  do we need to actually ensure uniqueness by adding some numbers to the end of long_name (we could check for duplicate names and add suffixes if necessary. good to have some syste that makes it easy to find all identical-param datasets


def make_params( params_short_name = 'minimal_onestep_diag10' ):
# 'easy_hypers', currently uses 'must move exactly onestep away'
# and 'avoid diagonal', but weigths are [1,0], so diagonal does nothing.
  
  def new_params_from_base( changes, base_params ):
    assert all( [k in base_params.keys() for k in changes.keys()] )
    if changes:
      assert 'short_name' in changes.keys()
    
    params_new = base_params.copy()
    params_new.update( changes )
    
    return params_new
  
  base_params = {
      'short_name': 'minimal_onestep_diag10',
      'years': range(1),
      'days': range(1),
      'height': 2,
      'width': 2,
      'feature_functions_name': 'one_step_and_not_diagonal',
      'num_features': 2,
      'prior_on_hypers': ['(gamma 6 1)'] * 2,
      'hypers': [1,0],
      'learn_hypers': False,
      'num_birds': 1,
      'softmax_beta': 4,
      'observes_loaded_from': None,
      'venture_random_seed': 1,
      'features_loaded_from': None,
      'observes_saved_to': None,
    'max_years': None,
    'max_days': None,
  }

  short_name_to_changes = unit_parameter_dicts.parameter_short_name_to_changes
                            
  params = new_params_from_base( short_name_to_changes[ params_short_name ],
                                 base_params )

  for max_param, param in zip( ('max_days','max_years'), ('days','years') ):
    max_v, lst_v = params[ max_param ], params[ param ]
    if max_v is None:
      params[ max_param ] = max( lst_v )
    else:
      assert max_v <=  max( lst_v )

  
  # Generate features dicts
  if not params['features_loaded_from']:
    args = params['height'], params['width'], params['years'], params['days']
    kwargs = dict( feature_functions_name = params['feature_functions_name'] )
    venture_features_dict, python_features_dict = make_features_dict(*args,**kwargs)

  else:
    params['feature_function_names'] = 'features_loaded_from'
    out = load_features( params['features_loaded_from'], params['years'], params['days'],
                         params['max_years'], params['max_days'] )
    venture_features_dict, python_features_dict = out

  params['features'] = venture_features_dict  
  params['features_as_python_dict'] = python_features_dict

  assert params['num_features'] == len( python_features_dict[(0,0,0,0)] )
  assert len( params['prior_on_hypers'] ) == len( params['hypers'] ) ==  params['num_features']
  


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
               max_days=int,
               max_years=int,
               features=(dict,str),
               num_features=int,
               hypers=list,
               learn_hypers=bool,
               prior_on_hypers=list,
               softmax_beta=int,
               venture_random_seed=int,
    features_as_python_dict=dict,)
  
  for k,v in types.items():
    assert isinstance(params[k],v)


  # Check subtypes
  subtypes = dict( hypers = (int,float),
                   prior_on_hypers = str,
                   years = int,
                   days = int )
  
  for k,v in subtypes.items():
    for el in params[k]:
      assert isinstance( el, v )

  # Check types if not None
  types_not_none = {'observes_loaded_from':str,
                    'observes_saved_to':str,
                    'features_loaded_from':str,
                    'learn_hypers':bool }

  for k,v in types_not_none.items():
    param = params[ k ]
    if param is not None:
      assert isinstance( param, v)
  

  return params




# UTILS FOR MAKING INFER UNIT OBJECTS BASED ON SAVED OBSERVES



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
                 'long_name': long_name
  }

  infer_params.update(update_dict)
  return infer_params


def make_infer_unit( generate_data_filename, prior_on_hypers, ripl_thunk,
                     model_constructor): #, load_observe_range=False,use_range_defaults=False):
  '''Utility function that takes synthetic data filename, prior_on_hypers, model_type,
     and generates an inference Unit object with same parameters as synthetic data
     Unit but with prior_on_hypers and optionally with Poisson instead of Multinomial.

     If *load_observe_range* is not False, run *load_observes* on the inference unit
     object with observe_range *load_observe_range*'''

  assert model_constructor in (Multinomial, Poisson)

  with open(generate_data_filename,'r') as f:
    store_dict = pickle.load(f)

  generate_data_params = store_dict['generate_data_params']
  infer_params = generate_data_params_to_infer_params(generate_data_params,
                                                      prior_on_hypers,
                                                      generate_data_filename)

  infer_unit = model_constructor( ripl_thunk(), infer_params) 

  return infer_unit



def make_infer_unit_and_observe_defaults( generate_data_filename, prior_on_hypers, ripl_thunk,
                                          model_constructor):
  
  infer_unit =  make_infer_unit( generate_data_filename,
                                 prior_on_hypers,
                                 ripl_thunk,
                                 model_constructor)
  load_observe_range = None
  use_range_defaults = True

  load_observes(infer_unit, load_observe_range, use_range_defaults, generate_data_filename)
  
  return infer_unit


def _test_inference(generate_data_unit, observe_range, ripl_thunk, model_constructor):
  
  assert isinstance(generate_data_unit, (Multinomial,Poisson))
  if observe_range is not None:
    assert isinstance(observe_range, Observe_range)
  
  generate_data_params = generate_data_unit.params
  
  generate_data_store_dict_filename, _ = store_observes(generate_data_unit, observe_range)

  ## NOTE THIS PRIOR MAKE HYPERS [1,0] easy to learn
  prior_on_hypers = ['(gamma 1 1)'] * generate_data_params['num_features']
  infer_unit = make_infer_unit_and_observe_defaults( generate_data_store_dict_filename,
                                                     prior_on_hypers,
                                                     ripl_thunk,
                                                     model_constructor)
                                
  return observe_range, generate_data_unit, generate_data_store_dict_filename, infer_unit
  

def _default_test_inference():
  model_constructor = Multinomial
  ripl_thunk = mk_p_ripl
  params_short_name = 'bigger_onestep_diag105'
  generate_data_unit = model_constructor( ripl_thunk(), make_params( params_short_name) )

  observe_range = None ## for max range
  _,_,_, infer_unit = _test_inference(generate_data_unit, observe_range, ripl_thunk, model_constructor)
  
  for _ in range(4):
    infer_unit.ripl.infer(20)
    print 'hypers,logsscores,mse:  ', compare_hypers(generate_data_unit, infer_unit)
  return infer_unit


def compare_hypers(gtruth_unit,inferred_unit):
  'Compare hypers across two different Birds unit objects'
  def mse(hypers1,hypers2): return np.mean((hypers1-hypers2)**2)

  def get_hypers(ripl,num_features):
    return np.array([ripl.sample('hypers%i'%i) for i in range(num_features)])
  
  hypers = []
  logscores = []
  
  for r in (gtruth_unit.ripl, inferred_unit.ripl):
    hypers.append( get_hypers(r, gtruth_unit.num_features))
    logscores.append( r.get_global_logscore() )
  mse_pair = mse(*hypers)

  return hypers,logscores,mse_pair

  
  

## CELL NAMES (GRID REFS)
# Multinomial Venture prog just has integer cell indices
# We only convert to ij for Python stuff that displays
# (We use ij form for synthetic data generation also
# and so that has to be converted to an index before conditioning)
  

class Multinomial(object):
  
  def __init__(self, ripl, params, delay_load_assumes=False):

    self.ripl = ripl
    print '\n\nBirds Unit Instance created with %s ripl\n\n----\n' % self.ripl.backend()
    
    self.params = params
    for k,v in self.params.iteritems():
      setattr(self,k,v)

    self.cells = self.width * self.height
    self.ripl.set_seed( self.venture_random_seed );

    
    if self.ripl.list_directives() != []:
      self.assumes_loaded = True # If ripl pre-loaded, don't load assumes
    elif delay_load_assumes:
      self.assumes_loaded = False
    else:
      self.load_assumes()



## UTILITIES FOR SAVING, LOADING, GENERATING SYNTHETIC DATA
  def save(self, directory):
    ## FIXME random_directory should be ripl hash
    random_directory_name = np.random.randint(10**9) 
    filename = directory + '/%s/%s' % ( self.long_name, str(random_directory_name) )
    ensure(filename)
    
    with open(filename + 'params.dat','w') as f:
      pickle.dump(self.params,f)

    self.ripl.save( filename + 'ripl.dat')
    print 'Saved to %s' % filename
    return filename


  def make_saved_model(self,filename, backend=None):
    # Currently defaults to same backend
    if backend is None:
      backend = self.ripl.backend()
    
    with open(filename + 'params.dat','r') as f:
      params = pickle.load(f)
      
    ripl = backend_to_ripl_thunk( backend )()
    ripl.load( filename + 'ripl.dat')
  
    return self.__class__( ripl, params)
    

  def get_params(self):
    self.params['ripl_directives'] = self.ripl.list_directives()
    return self.params


  def get_max_observe_range(self):
    params =  {'days_list': [d for d in self.days if d <= self.max_days],
               'years_list': [y for y in self.years if  y<= self.max_years],
               'cells_list': range(self.cells)}
    return Observe_range(**params)


  def ensure_assumes(self):
    if self.assumes_loaded:
      pass
    else:
      self.load_assumes()


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

## TODO add option to vary scale
    if self.learn_hypers:
      ##ripl.assume('scale', '(scope_include (quote hypers) (quote scale) (gamma 1 1))')
      for k in range(self.num_features):
        ##ripl.assume('hypers%d' % k, '(scope_include (quote hypers) %f (* scale (normal 0 5) ))' % k)
        ripl.assume('hypers%d' % k, '(scope_include (quote hypers) %f %s)' %
                    (k, self.prior_on_hypers[k] ))

    else:
      for k, value_k in enumerate(self.hypers):
        ripl.assume('hypers%d' % k, '(scope_include (quote hypers) 0 %f)'%value_k)

    
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

    
    ripl.assume('sum_phi',
      '(mem (lambda (y d i) ' +
                fold( '+', '(phi y d i j)', 'j', self.cells) +
                '))' )


    ripl.assume('get_bird_move_dist',
      '(mem (lambda (y d i) ' +
        fold('simplex', '(/ (phi y d i j) (sum_phi y d i))', 'j', self.cells) +
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

    ripl.assume('single_observe_birds', '(lambda (y d i) (poisson (+ (single_count_birds y d i) 0.00001)))')


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
## we memoize this, so that observes are fixed for a given run of the model

    ripl.assume('observe_birds', '(mem (lambda (y d i) (poisson (+ (count_birds_v2 y d i) 0.00001))))')
    
    self.assumes_loaded = True

  



  def get_hist(self, bird_to_pos_list):
    # How np.histogram works:
    # np.histogram([0,1,2],bins=np.arange(0,3)) == ar[1,2], ar[0,1,2]
    # np.histogram([0,1,2],bins=np.arange(0,4)) == ar[1,1,1], ar[0,1,2,3]
    hist,_ = np.histogram(bird_to_pos_list,bins=range(self.cells+1))
    assert len(hist)==self.cells
    assert np.sum(hist) == self.num_birds
    return hist


  def bird_to_pos( self, year, day, hist=False):
    'Return list [cell_index for bird_i], or optionally hist, for given day'
    bird_to_pos_list=[]
    for bird_id in self.ripl.sample('bird_ids'):
      args = bird_id, year, day
      bird_to_pos_list.append(self.ripl.predict('(get_bird_pos %i %i %i)'%args))
                                                           
    all_bird_to_pos_list = self.ripl.predict('(all_bird_pos %i %i)'%(year,day))
    assert all( np.array(all_bird_to_pos_list)==np.array(bird_to_pos_list) )
    ## Check that get_bird_pos and all_bird_pos agree (Could turn off 
    # for speed)

    return bird_to_post_list if not hist else self.get_hist(bird_to_pos_list)




  def get_bird_locations(self, years=None, days=None):
    '''Returns dict { y: { d:histogram of bird positions on y,d}  }
       for y,d in product(years,days) '''
    if years is None: years = self.get_max_observe_range()['years_list']
    if days is None: days = self.get_max_observe_range()['days_list']

    
    all_days = product(self.get_max_observe_range()['years_list'],
                       self.get_max_observe_range()['days_list'] )
    
    assert all( [ (y,d) in all_days for (y,d) in product(years,days) ] )
    
    bird_locations = {}
    for y in years:
      bird_locations[y] = {}
      for d in days:
        bird_locations[y][d] = self.bird_to_pos(y,d,hist=True)
    
    return bird_locations

    

    

class Poisson(Multinomial):
  
  def load_assumes(self):

    ripl = self.ripl    
    print "Loading assumes"
    
    ripl.assume('num_birds', self.num_birds)
    ripl.assume('cells', self.cells)

    ## TODO add option to vary scale
    if self.learn_hypers:
      ##ripl.assume('scale', '(scope_include (quote hypers) (quote scale) (gamma 1 1))')
      for k in range(self.num_features):
        ##ripl.assume('hypers%d' % k, '(scope_include (quote hypers) %f (* scale (normal 0 5) ))' % k)
        ripl.assume('hypers%d' % k, '(scope_include (quote hypers) %f %s)' %
                    (k, self.prior_on_hypers[k] ))

    else:
      for k, value_k in enumerate(self.hypers):
        ripl.assume('hypers%d' % k, '(scope_include (quote hypers) 0 %f)'%value_k)


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

## not used anywhere. presumably a multinomial (vs. poisson)
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

    self.assumes_loaded = True


  def bird_to_pos( self, year, day, hist=False):
    'Return list [cell_index for bird_i], or optionally hist, for given day'
    bird_to_pos_list = []
    r = self.ripl
    
    for i in range(self.cells):
      bird_to_pos_list.append(r.predict('(count_birds %d %d %d)' % (year, day, i))) 
      
    return bird_to_pos_list if not hist else self.get_hist(bird_to_pos_list)



class PoissonOld(object):

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
  

## FIXME predict
  def getBirdLocations(self, years=None, days=None, predict=True):
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













