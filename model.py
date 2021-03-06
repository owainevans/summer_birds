from utils import *
from features_utils import make_features_dict, load_features
import unit_parameter_dicts

from venture.shortcuts import make_lite_church_prime_ripl as mk_l_ripl
from venture.shortcuts import make_puma_church_prime_ripl as mk_p_ripl

from itertools import product
import cPickle as pickle
import numpy as np


### PLAN NOTES


## IMMEDIATE TODO
# get rid of load_observes and fully replace with get/load functions

# GO OVER THE 'get bird locs' in tests_infer and unit methods and make
# sure we forget all the predicts after adding them



# SMC:

# --- note, we should probably implement the thing that integrates out moves. you compute birds at cell i by doing one poisson of some over all source cells.
# --- note: for poisson, doing inference by hand should be fairly easy. should implement that using numpy/scipy. good exercise in understanding stuff here. 

# Issue that no random choice for birds moving from i to j if none at sources of j. Worry is that you have some patch of the grid where there are no birds. Then there will always be no birds. So maybe need something that spawns birds with some probability. Problem is that this gives you lots of new random variables unless you control it in some way. (Note that if we are freezing older variables, then we'll have a large but fixed number of variables over time. So this should be ok). 

# KEY INFERENCE TESTS:
# 1. Good MSE performance on onebird and dataset 2 reconstruction
# 2. Compare Poisson model against numerical simulation (for small version). Use this 
# As part of unit testing. (Plus, mb, thing that integrates out moves). 

# GENERAL POINTS ABOUT TESTING
# check coverage and improve (esp. type checking)
# be more careful about filenames for tests that save/load
# should have fast test that generates data once, then does 
# everything with same infer unit, and does so on a random small model. 


# keep in mind:
# tasks: reconstruction of latent states, prediction, param inference (onebird / poisson)
# params for inference: (depends on space of inference progs)


## NOTE we are using PREDICT for getting locations. Careful of predicts piling up. 
## EXP RUNNER AND BEING ABLE TO INTEGRATE WITH THAT


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

# 4. Add more realistic features. Things like wind-direction

#  5. play with experiment runner and see if there's any issues with integrating. 
#     some functionality that it doesn't have (scaling/timing info for particular
#     elements of inference, e.g. loading vs. single step of MH, etc.) might be 
#     necessary to write into birds using the timeit module. 



# TODO
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



def new_load_observes(unit, observations, load_observe_sub_range, use_range_defaults):
  
  # *observations* dict has form { (y,d,cell) : count }
  first_key = observations.keys()[0]
  first_value = observations[first_key]
  assert isinstance(first_key, tuple)
  assert len(first_key) == 3
  assert isinstance(first_value, (int,float))
  
  # can only *observe* after all *assumes* are loaded
  unit.ensure_assumes()

  ## check that *load_observe_sub_range* is within the max range
  ## (if using *range_defaults*, over-write whatever is *load_observe_sub_range*)
  default_observe_range = unit.get_max_observe_range()

  if use_range_defaults:
    load_observe_sub_range = default_observe_range
  else:
    load_observe_sub_range.assert_is_observe_sub_range(default_observe_range)

  for y,d,i in load_observe_sub_range.get_year_day_cell_product():
    count_i = observations[(y,d,i)]
    unit.ripl.observe('(observe_birds %i %i %i)'%(y,d,i), count_i )
  
  years = load_observe_sub_range['years_list']
  days = load_observe_sub_range['days_list']
  cells = load_observe_sub_range['cells_list']

  print 'Loaded years: %s, days: %s, cells: %s' %(years,days,cells)



def dataset_load_observes(unit, dataset, name, load_observe_sub_range, use_range_defaults):
  cleaned_observations = dataset_get_observes(dataset, name,
                                              load_observe_sub_range)
  new_load_observes(unit, cleaned_observations,
                    load_observe_sub_range, use_range_defaults)



def dataset_get_observes(dataset, name, load_observe_sub_range):
  observations_file = "data/input/dataset%d/%s-observations.csv" % (dataset, name)
  observations = read_observations(observations_file)
  
  # need to convert *observations* from form { y: { d: [count_celli] } }
  # to form { (y,d,cell_i): count_celli }
  # NOTE: also remove any observation not in *load_observe_sub_range*
  cleaned_observations = {}

  years = load_observe_sub_range['years_list']
  days = load_observe_sub_range['days_list']

  for y in years:
    for (d, bird_counts_list) in observations[y]:
      if d not in days: continue
      for cell_index, bird_count in enumerate(bird_counts_list):
        cleaned_observations[(y,d,cell_index)] = bird_count

  return cleaned_observations
        

def synthetic_get_observes(store_dict_filename, load_observe_sub_range):

  with open(store_dict_filename,'r') as f:
    store_dict = pickle.load(f)
    observations = store_dict['observe_counts']
  
  filtered_observations = {}

  for y,d,i in load_observe_sub_range.get_year_day_cell_product():
    filtered_observations[(y,d,i)] = observations[(y,d,i)]

  return filtered_observations


def synthetic_load_observes(unit, load_observe_sub_range, use_range_defaults, store_dict_filename):

  filtered_observations = synthetic_get_observes(store_dict_filename, load_observe_sub_range)
  
  new_load_observes(unit, filtered_observations, load_observe_sub_range, use_range_defaults)

        
 

def dataset_load_observations(unit, dataset, name, load_observe_sub_range, use_range_defaults):
  'Load observations from Birds dataset'
  
  unit.ensure_assumes()

  observations_file = "data/input/dataset%d/%s-observations.csv" % (dataset, name)
  observations = read_observations(observations_file)

  default_observe_range = unit.get_max_observe_range()

  if use_range_defaults:
    load_observe_sub_range = default_observe_range
  else:
    load_observe_sub_range.assert_is_observe_sub_range(default_observe_range)

  years = load_observe_sub_range['years_list']
  days = load_observe_sub_range['days_list']

  for y in years:
    for (d, bird_counts_list) in observations[y]:
      if d not in days: continue
      for cell_index, bird_count in enumerate(bird_counts_list):
        unit.ripl.observe('(observe_birds %d %d %d)'%(y,d,cell_index), bird_count)




def computeScoreDay(d):
  bird_moves = self.ripl.sample('(get_birds_moving3 %d)' % d)
  score = 0
    
  for y in self.years:
    for i in range(self.cells):
      for j in range(self.cells):
        score += (bird_moves[y][i][j] - self.ground[(y, d, i, j)]) ** 2
    
      return score
  

def computeScore():
  infer_bird_moves = self.getBirdMoves()
  score = 0
  for key in infer_bird_moves:
    score += (infer_bird_moves[key] - self.ground[key]) ** 2

  return score



## Multinomial & Poisson functions for saving synthetic Observes and 
## loading and running unit.ripl.observe(loaded_observe)



# Store observes in file:
# filename determined by *long_name* unit attribute
# *observe_range* defaults to max range
def store_observes(unit, observe_range=None, synthetic_directory = 'synthetic'):
  
  # assumes need to be loaded to get observes
  unit.ensure_assumes()  
 
  # check *observe_range* and fill in *None* values
  max_observe_range = unit.get_max_observe_range();
  if observe_range is None:
    observe_range = max_observe_range
  else:
    assert isinstance(observe_range,Observe_range)
    observe_range.replace_none_with_super_range(max_observe_range)
    observe_range.assert_is_observe_sub_range(max_observe_range)
  

  # GET OBSERVE VALUES FROM MODEL

  # Observed (noisy) and ground-truth bird counts
  observe_counts={}
  gtruth_counts={}
  ripl = unit.ripl

  for y,d,i in observe_range.get_year_day_cell_product():
    gtruth_counts[(y,d,i)] = ripl.predict('(count_birds %i %i %i)'%(y,d,i))
    observe_counts[(y,d,i)] = ripl.predict('(observe_birds %i %i %i)'%(y,d,i))
    
    # (observe_birds y d i) is (poisson epsilon) if count for (y,d,i)=0.
    # (See Venture programs)
    if gtruth_counts[(y,d,i)] == 0 and observe_counts[(y,d,i)] > 0:
      assert False, 'gtruth_counts[%s] == 0 and observe_counts[%s] > 0' % str((y,d,i) )

  
  # PICKLE AND STORE COUNTS

  # Generate filenames based on *unit.parameters.long_name*.
  # Generate bird locations images (FIXME possibly remove).
  params = unit.get_params()
  date = '21_08_14' ## FIXME ADD DATE
  full_directory = '%s/%s/' % (synthetic_directory,date)
  ensure(full_directory)
  store_dict_filename = full_directory + params['long_name'] + '.dat'
  draw_bird_filename =  full_directory + params['long_name'] + '.png'

  years = observe_range['years_list']
  days = observe_range['days_list']
  directory_filename = (full_directory, draw_bird_filename)
  fig_ax = plot_save_bird_locations(unit, years, days, plot=True, save=True,
                                    directory_filename = directory_filename)
                                    
  # Build dict of parameters, *observe_range* and counts,
  # along with groundtruth *bird_locs* and pickle to file.

  store_dict = {'generate_data_params':params,
                'generate_data_backend': unit.ripl.backend(),
                'observe_counts':observe_counts,
                'observe_range':observe_range.copy_dict_only(), 
                'bird_locs': gtruth_counts}
                #'bird_locs_fig_ax':fig_ax} ## FIXME serialize figure!

  with open(store_dict_filename,'w') as f:
    pickle.dump(store_dict,f)
  print 'Stored observes in %s.'% store_dict_filename

  ## return filename for use running inference on synthetic data.
  return store_dict_filename, draw_bird_filename 



## TODO get rid of this entirely. remove from test_infer.py
def load_observes(unit, load_observe_sub_range,
                  use_range_defaults, store_dict_filename, observe_counts=None):

  unit.ensure_assumes()
 
  if observe_counts is None:
    with open(store_dict_filename,'r') as f:
      store_dict = pickle.load(f)
    
      observe_counts = store_dict['observe_counts']
      default_observe_range = Observe_range(**store_dict['observe_range']) ## since we pickle dict, not instance
  else:
    default_observe_range = unit.get_max_observe_range()

  if use_range_defaults:
    load_observe_sub_range = default_observe_range
  else:
    load_observe_sub_range.assert_is_observe_sub_range(default_observe_range)

  for y,d,i in load_observe_sub_range.get_year_day_cell_product():
    count_i = observe_counts[(y,d,i)]
    unit.ripl.observe('(observe_birds %i %i %i)'%(y,d,i), count_i )
  
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
      'phi_constant_beta': 4,
      'observes_loaded_from': None,
      'venture_random_seed': 1,
      'features_loaded_from': None,
      'observes_saved_to': None,
  'max_years_for_experiment': None,
  'max_days_for_experiment': None,
  }

  short_name_to_changes = unit_parameter_dicts.parameter_short_name_to_changes
                            
  params = new_params_from_base( short_name_to_changes[ params_short_name ],
                                 base_params )

  ## max(param) vs. max_param_for_experiment
  # Some datasets have a large number of days/years. We might 
  # want to only load some of the days/years (to avoid a huge
  # features dict). We select how much will be loaded with 
  # *max_params_for_experiment*. 
  for max_param, param in zip( ('max_days_for_experiment',
                                'max_years_for_experiment'),
                               ('days','years') ):
    max_v, lst_v = params[ max_param ], params[ param ]
    if max_v is None:
      params[ max_param ] = max( lst_v )
    else:
      assert max_v <=  max( lst_v )

  
  # Generate features dicts
  if not params['features_loaded_from']:
    ## Set whether we create Venture stack-dict (*'dict'*) or
    # Venture string expression (*'string'*) for loading features
    dict_string = 'dict'
    
    args = params['height'], params['width'], params['years'], params['days'], dict_string
    kwargs = dict( feature_functions_name = params['feature_functions_name'] )
    venture_features_dict, python_features_dict = make_features_dict(*args,**kwargs)

  else:
    params['feature_function_names'] = 'features_loaded_from'
    out = load_features( params['features_loaded_from'], params['years'], params['days'],
                         params['max_years_for_experiment'], params['max_days_for_experiment'] )
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
            'height','width','num_birds','phi_constant_beta',
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
               max_days_for_experiment=int,
               max_years_for_experiment=int,
               features=(dict,str),
               num_features=int,
               hypers=list,
               prior_on_hypers=list,
               phi_constant_beta=int,
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


def make_infer_unit(generate_data_filename, prior_on_hypers, ripl_thunk, model_constructor):
  '''Utility function that takes synthetic data filename, prior_on_hypers, model_type,
     and generates an inference Unit object with same parameters as synthetic data
     Unit but with prior_on_hypers and either Poisson/Multinomial model.'''

  assert model_constructor in (Multinomial, Poisson)
  
  with open(generate_data_filename,'r') as f:
    store_dict = pickle.load(f)

  generate_data_params = store_dict['generate_data_params']
  infer_params = generate_data_params_to_infer_params(generate_data_params,
                                                      prior_on_hypers,
                                                      generate_data_filename)

  if ripl_thunk is None:
    ripl_thunk = backend_to_ripl_thunk(store_dict['generate_data_backend'])
    
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


# def _test_inference(generate_data_unit, observe_range, ripl_thunk, model_constructor):
  
#   assert isinstance(generate_data_unit, (Multinomial,Poisson))
#   if observe_range is not None:
#     assert isinstance(observe_range, Observe_range)
  
#   generate_data_store_dict_filename, _ = store_observes(generate_data_unit, observe_range)

#   ## NOTE THIS PRIOR MAKES HYPERS [1,0] easy to learn
#   prior_on_hypers = ['(gamma 1 1)'] * generate_data_unit.params['num_features']
#   infer_unit = make_infer_unit_and_observe_defaults( generate_data_store_dict_filename,
#                                                      prior_on_hypers,
#                                                      ripl_thunk,
#                                                      model_constructor)
                                
#   return observe_range, generate_data_unit, generate_data_store_dict_filename, infer_unit
  

# def _test_inference_bigger_onestep():
#   observe_range = None ## for max range
#   _,_,_, infer_unit = _test_inference(generate_data_unit, observe_range, ripl_thunk, model_constructor)
  
#   for _ in range(4):
#     infer_unit.ripl.infer(20)
#     print 'hypers,logsscores,mse:  ', compare_hypers(generate_data_unit, infer_unit)
#   return infer_unit







class Multinomial(object):
  
  def __init__(self, ripl, params, delay_load_assumes=False):

    self.ripl = ripl
    self.params = params

    # Multinomial Venture model has integer-indexed cells.
    # For plotting and synthetic-data generation we use matrix
    # row-column indices (with 'C' order by default
    self.cells = self.params['width'] * self.params['height']

    ## TODO THINK ABOUT SEED
    #self.ripl.set_seed( self.params['venture_random_seed'] );

    if self.ripl.list_directives() != []:
      self.assumes_loaded = True # If ripl pre-loaded, don't load assumes
    elif delay_load_assumes:
      self.assumes_loaded = False
    else:
      self.load_assumes()

## UTILITIES FOR SAVING, LOADING, GENERATING SYNTHETIC DATA
  def get_model_name(self):  return 'Multinomial'

  def save(self, directory):
    ## FIXME random_directory should be ripl hash
    random_directory_name = np.random.randint(10**9) 
    filename = directory + '/%s/%s' % ( self.params['long_name'], str(random_directory_name) )
    ensure(filename)
    
    with open(filename + 'params.dat','w') as f:
      pickle.dump(self.params,f)

    self.ripl.save( filename + 'ripl.dat')
    print 'Saved to %s' % filename
    return filename


  def make_saved_model(self,filename, backend=None):
    # Defaults to same backend
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
    days = self.params['days']
    years = self.params['years']
    max_days = self.params['max_days_for_experiment']
    max_years = self.params['max_years_for_experiment']
    
    params =  {'days_list': [d for d in days if d <= max_days],
               'years_list': [y for y in years if y <= max_years],
               'cells_list': range(self.cells) }

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
    if self.params['learn_hypers']:
      ##ripl.assume('scale', '(scope_include (quote hypers) (quote scale) (gamma 1 1))')
      for k in range(self.params['num_features']):
        ##ripl.assume('hypers%d' % k, '(scope_include (quote hypers) %f (* scale (normal 0 5) ))' % k)
        ripl.assume('hypers%d' % k, '(scope_include (quote hypers) %f %s)' %
                    (k, self.params['prior_on_hypers'][k] ))

    else:
      for k, value_k in enumerate(self.params['hypers']):
        ripl.assume('hypers%d' % k, '(scope_include (quote hypers) 0 %f)'%value_k)

    
    ripl.assume('features', self.params['features'])
    ripl.assume('num_birds', self.params['num_birds'])
    
    bird_ids = ' '.join(map(str,range(self.params['num_birds']))) # multibird only
    ripl.assume('bird_ids','(list %s)'%bird_ids) # multibird only

    ripl.assume('phi_constant_beta',self.params['phi_constant_beta'])

    ripl.assume('phi', """
      (mem (lambda (y d i j)
        (let ((fs (lookup features (array y d i j))))
          (exp (* phi_constant_beta %s)))))"""
                % fold('+', '(* hypers_k_ (lookup fs _k_))', '_k_', self.params['num_features']))

    
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
          (scope_include (quote d) bird_id
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

    ripl.assume('count_birds_no_map', """
      (lambda (y d i)
        (size (filter 
                 (lambda (bird_id) (= i (get_bird_pos bird_id y d)) )
                  bird_ids) ) ) """)

# alternative version of count_birds
    ripl.assume('count_birds', """
      (mem (lambda (y d i)
        (size (filter
                (lambda (x) (= x i)) (all_bird_pos y d)))))""" )

  
## note that count_birds_v2 seems to work faster. haven't looked at whether it harms inference.
## we memoize this, so that observes are fixed for a given run of the model

    ripl.assume('observe_birds', '(mem (lambda (y d i) (poisson (+ (count_birds y d i) 0.00001))))')
    
    self.assumes_loaded = True

  

  def get_hist(self, bird_locations):
    # How np.histogram works:
    # np.histogram([0,1,2],bins=np.arange(0,3)) == ar[1,2], ar[0,1,2]
    # np.histogram([0,1,2],bins=np.arange(0,4)) == ar[1,1,1], ar[0,1,2,3]
    hist,_ = np.histogram(bird_locations,bins=range(self.cells+1))
    assert len(hist)==self.cells
    assert np.sum(hist) == self.params['num_birds']
    return hist

  def year_day_to_bird_locations( self, year, day, hist=False):
    'Return list [cell_index for bird_i], or optionally histogram over cells, for given day'
    bird_id_to_location=[]
    for bird_id in self.ripl.sample('bird_ids'):
      args = bird_id, year, day
      bird_id_to_location.append(self.ripl.predict('(get_bird_pos %i %i %i)'%args))
                                                           
    all_bird_id_to_location = self.ripl.predict('(all_bird_pos %i %i)'%(year,day))
    assert all( np.array(all_bird_id_to_location)==np.array(bird_id_to_location) )
    ## Check that get_bird_pos and all_bird_pos agree (Could turn off 
    # for speed)

    return bird_id_to_location if not hist else self.get_hist(bird_id_to_location)


  def days_list_to_bird_locations(self, years=None, days=None):
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
        bird_locations[y][d] = self.year_day_to_bird_locations(y,d,hist=True)
    
    return bird_locations

    

class Poisson(Multinomial):
  
  def get_model_name(self):
    return 'Poisson'

  def load_assumes(self):

    ripl = self.ripl
    print "Loading assumes"
    
    ## TODO add option to vary scale
    if self.params['learn_hypers']:
      ##ripl.assume('scale', '(scope_include (quote hypers) (quote scale) (gamma 1 1))')
      for k in range(self.params['num_features']):
        ##ripl.assume('hypers%d' % k, '(scope_include (quote hypers) %f (* scale (normal 0 5) ))' % k)
        ripl.assume('hypers%d' % k, '(scope_include (quote hypers) %f %s)' %
                    (k, self.params['prior_on_hypers'][k] ))

    else:
      for k, value_k in enumerate(self.params['hypers']):
        ripl.assume('hypers%d' % k, '(scope_include (quote hypers) 0 %f)'%value_k)

    ripl.assume('num_birds', self.params['num_birds']) ## used in *count_birds* (below)
    ripl.assume('cells', self.cells)

    ripl.assume('features', self.params['features'])
    ripl.assume('phi_constant_beta', self.params['phi_constant_beta'])

    ripl.assume('width', self.params['width'])
    ripl.assume('height', self.params['height'])
    ripl.assume('max_dist_squared', '18.')

## Distances computed with 'C' order. If (height,width)=(3,4), then
# array([[ 0,  3,  6,  9],
#        [ 1,  4,  7, 10],
#        [ 2,  5,  8, 11]])
# So x (column) given by cell_ind / height, row given by cell_ind % height

    def make_cell_row_column_string():
      my_dict = {}
      for cell_ind in range(self.cells):
        # make the key a tuple for compatibility with *python_features_to_venture_exp*
        my_dict[ (cell_ind,) ] =  ( cell_ind % self.params['height'], cell_ind / self.params['height'])
      return python_features_to_venture_exp( my_dict )

    ripl.assume('cell_to_row_column_dict', make_cell_row_column_string())
    ripl.assume('cell_to_row_column', '(lambda (cell) (lookup cell_to_row_column_dict (array cell)))')

    #ripl.assume('cell_to_x', '(lambda (cell) (int_div cell height))')
    #ripl.assume('cell_to_y', '(lambda (cell) (int_mod cell height))')
    ripl.assume('cell_to_row', '(lambda (cell) (lookup (cell_to_row_column cell) 0))')
    ripl.assume('cell_to_column', '(lambda (cell) (lookup (cell_to_row_column cell) 1))')

    #ripl.assume('cell2P', '(lambda (cell) (make_pair (cell2X cell) (cell2Y cell)))')
    ripl.assume('row_column_to_cell', '(lambda (row column) (+ (* height column) row))')

    ripl.assume('square', '(lambda (x) (* x x))')

    ripl.assume('dist_squared', """
      (lambda (x1 y1 x2 y2)
        (+ (square (- x1 x2)) (square (- y1 y2))))""")

    ripl.assume('cell_dist_squared', """
      (lambda (i j)
        (dist_squared
          (cell_to_row i) (cell_to_column i)
          (cell_to_row j) (cell_to_column j)))""")
    
    # phi is the unnormalized probability of a bird moving from cell i to cell j on day d
    ripl.assume('phi', """
      (mem (lambda (y d i j)
        (if (> (cell_dist_squared i j) max_dist_squared) 0
          (let ((fs (lookup features (array y d i j))))
            (exp (* phi_constant_beta %s) )))))"""
                % fold('+', '(* hypers__k (lookup fs __k))', '__k', self.params['num_features']))
    
    ripl.assume('sum_phi',
      '(mem (lambda (y d i) ' +
                fold( '+', '(phi y d i j)', 'j', self.cells) +
                '))' )

    ripl.assume('get_bird_move_prob',
                '(mem (lambda (y d i j) (/ (phi y d i j) (sum_phi y d i))) )')
                
    ripl.assume('get_bird_move_dist',
      '(mem (lambda (y d i) ' +
         fold('simplex', '(get_bird_move_prob y d i j)', 'j', self.cells) +
           '))')
    
    
## TODO count_birds assumes all birds at cell 0 at start, abstract this out
    ripl.assume('count_birds', """
      (mem (lambda (y d i)
        (if (= d 0) (if (= i 0) num_birds 0)""" +
          fold('+', '(get_birds_moving y (- d 1) __j i)', '__j', self.cells) + ")))")
    

    # if no birds at cell i, no movement to any j from i
    # n = birdcount_i * normed_prob(i,j)
    # return: (lambda (j) (poisson n) )

# what do scopes day here? if there are no birds at i, then no birds can move to j.
# (note an asymmetry in model, where a bird can disappear but can't appear. martingale thing.)
# moreover, if the prob of a bird moving from i to j is 0, the no birds move from i to j. 

# these are deterministic functions, and so they won't be subject to inference. we don't
# put them in the scope_include, but it wouldn't matter if they were in it. an alternate
# model would make these latents stochastic, say with low prob of creating a bird. this 
# would make the number of latents fixed over time, but would also make it much larger. 

# however, note that a variable (bird_movements_loc y d i j) could become random if we change
# whether bird is at i on this day. so we'd want this to shift in and out of the scope. would this work?
# suppose we start with no bird at cell 1 on day 1, then we have 1 bird there. so initially we
# have (bird_m_loc 0 1 1 j)=0 for all j. then we want this to be a poisson random var.

# filtering: we observe birds on day 0, with count (poi num_birds) at cell 0 and (poi .0001) 
# elsewhere. our model has uncertainty over hypers. but still assumes all birds at 0 
# on first day. so we don't get any info here (we could get info about amount of noise
# in observations --- if that was learnable in model). on day 0, the only random vars
# should be (0 0 0 j) for j where (phi 0 j) is non-zero. here's the issue: on day 0, 
# i don't observe anything that depends on (bird_moves_loc 0 0 0 j), so the 
# array (0 0 0 j) won't be in the scope 0. instead it will be in scope 1. suppose
# we then do infer on d-1 after observing for d. we can't do this on day 0 (but
# needn't do any inference then anyway - as nothing can be learned on that day)

# what about doing cell-wise incremental inference? if we observe count at i from
# day d, that gives us info about (bird_moves_loc y d-1 source i) for each source.
# in principle, observing one cell count for day d could activate every latent for
# the previous day. but in our practical cases, most of those latents will be deterministically
# zero. suppose i do inference on day d-1 by looping over cells in order. some cells
# won't be in the scope, so we'll have an error. (vlad's point that when something 
# is not in the block, you probably want it to do nothing). 

    ##if ripl.backend() == 'puma':
    ripl.assume('bird_movements_loc', """
    (mem (lambda (y d i)
      (if (= (count_birds y d i) 0)
        (lambda (j) 0)
          (mem (lambda (j)
            (if (= (phi y d i j) 0) 0
              (let ((n (* (count_birds y d i) (get_bird_move_prob y d i j))))
                (scope_include d (array y d i j)
      (poisson n)))))))))""")
  
    # else:   ## FIXME HACK THAT SHOULD REPLACE WHEN LITE TAKES ARRAYS
    #   ripl.assume('bird_movements_loc', """
    #   (mem (lambda (y d i)
    #     (if (= (count_birds y d i) 0)
    #       (lambda (j) 0)
    #       (mem (lambda (j)
    #         (if (= (phi y d i j) 0) 0
    #           (let ((n (* (count_birds y d i) (get_bird_move_prob y d i j))))
    #             (scope_include d (+ (* 1000 y) (+ (* 100 d) (+ (* 10 i)  j)) )
    #   (poisson n)))))))))""")
    
    
    ripl.assume('observe_birds', '(mem (lambda (y d i) (poisson (+ (count_birds y d i) 0.00001))))')

    # returns number birds from i,j (we want to force this value)
    ripl.assume('get_birds_moving', """
      (lambda (y d i j)
        ((bird_movements_loc y d i) j))""")
    
    ripl.assume('get_birds_moving1',
                '(lambda (y d i) %s)' % fold('array', '(get_birds_moving y d i __j)', '__j', self.cells))
    ripl.assume('get_birds_moving2',
                '(lambda (y d) %s)' % fold('array', '(get_birds_moving1 y d __i)', '__i', self.cells))
    ripl.assume('get_birds_moving3',
                '(lambda (d) %s)' % fold('array', '(get_birds_moving2 __y d)', '__y', len(self.params['years'])))
    ripl.assume('get_birds_moving4',
                '(lambda () %s)' % fold('array', '(get_birds_moving3 __d)', '__d', len(self.params['days'])-1))

    self.assumes_loaded = True


  def year_day_to_bird_locations( self, year, day, hist=True):
    'Return list [cell_index for bird_i], or optionally hist, for given day'
    assert hist, 'For Poisson model birds are not identified and so output is always hist'

    bird_locations = []
    r = self.ripl
    
    for i in range(self.cells):
      bird_locations.append(r.predict('(count_birds %d %d %d)' % (year, day, i)))
      
    return np.array( bird_locations )













