import numpy as np
from itertools import product
from nose.tools import eq_, assert_almost_equal

from venture.shortcuts import make_puma_church_prime_ripl as mk_p_ripl
from venture.shortcuts import make_lite_church_prime_ripl as mk_l_ripl
from utils import  Observe_range

from model import *
from mytest_utils import *

import sys


## SCORE FUNCTIONS


def compare_hypers(gtruth_unit,infer_unit):
  'Compare hypers across two different Birds unit objects'
  hypers = []
  logscores = []
  
  for unit in (gtruth_unit, infer_unit):
    hypers.append( get_hypers(unit) )
    logscores.append( unit.ripl.get_global_logscore() )

  return dict(hypers=hypers, logscores=logscores, mse=mse(*hypers))


def mse(hypers1,hypers2): return np.mean((hypers1-hypers2)**2)

def get_hypers(unit):
  ripl = unit.ripl
  num_features = unit.params['num_features']
  return np.array([ripl.sample('hypers%i'%i) for i in range(num_features)])


def compare_latents(gtruth_unit, infer_unit, observe_range):
  
  years = observe_range['years_list']
  days = observe_range['days_list']
  cells = observe_range['cells_list']

  model = gtruth_unit.get_model_name()


  def poisson_latents():
    ## NB latents are num birds moving from i to j
    year_day_i_j = product(years,days,cells,cells)
    moves_ij = {}   
    all_mses = []

    for year,day,i,j in year_day_i_j:
      args = year,day,i,j
      moves_ij[args] = []

      for unit in (gtruth_unit, infer_unit):
        r = unit.ripl
        moves_ij[args].append(r.predict('(get_birds_moving %i %i %i %i)'%args ))

      all_mses.append( mse(*moves_ij[args]) )

    return np.mean(all_mses)


  def multinomial_latents():
    ## NB latents are counts for birds at each position
    gtruth_locs = gtruth_unit.days_list_to_bird_locations(years, days)
    infer_locs =  infer_unit.days_list_to_bird_locations(years, days)

    mse_part = []
    for year,day in product(years,days):
      mse_part.append( mse( gtruth_locs[year][day], infer_locs[year][day]) )
    return np.mean(mse_part)
    
  mse_latents = poisson_latents() if model=='Poisson' else multinomial_latents()

  print '\n\n mse latents, range, val', observe_range, ' mse ' , mse_latents
  return mse_latents



## INFERENCE PROGRAMS
def transitions_to_mh_default( transitions = 50 ):
  return lambda unit, year, day, cell, gtruth_unit: unit.ripl.infer( transitions )

  
def transitions_to_cycle_mh( transitions_latents=10, transitions_hypers=5,
                             number_of_cycles=1):
  
  def cycle_filter_on_days( unit, year, day, cell, gtruth_unit):

    if day > 0:
      args = transitions_latents,
      hypers = '(mh hypers all %i)' % transitions_latents
      latents = '(mh %i one %i)' % (day-1, transitions_hypers)
      s = '(cycle ( %s %s ) %i )' % (hypers, latents, number_of_cycles)
      unit.ripl.infer(s)

  return cycle_filter_on_days



## INTERLEAVE OBSERVE AND INFERENCE PROGRAM
def incremental_observe_infer( unit, observes_filename, observe_range,
                               inference_prog, infer_every_cell=False, gtruth_unit = None,
                               score_function = None, observations=None):
  
  if observes_filename is None:
    load_observes_function = dataset_load_observes
  else:
    load_observes_function = synthetic_load_observes

  scores = {}
  scores['before'] = score_function(unit)
    

  cells_list = observe_range['cells_list']
  use_range_defaults = False

  for year in observe_range['years_list']:
    for day in observe_range['days_list']:

      if not infer_every_cell:
        observe_sub_range = Observe_range(years_list=[year], days_list=[day], cells_list=cells_list)
        new_load_observes(unit, observations, observe_sub_range, use_range_defaults)
        inference_prog( unit, year, day, 'all', gtruth_unit)  ## FIXME dubious 'all'

      else:
        for cell in observe_range['cells_list']:
          observe_sub_range = Observe_range(years_list=[year], days_list=[day], cells_list=[cell])
          new_load_observes( unit, observations, observe_sub_range, use_range_defaults)
          inference_prog( unit, year, day, cell, gtruth_unit)

  scores['after'] = score_function(unit)
  return scores


def onebird():
  ## Get infer_unit for dataset1
  params = make_params( 'dataset1' )
  ripl_thunk = mk_p_ripl
  prior_on_hypers = ['(gamma 5 1)'] * 4
  generate_data_filename = None
  infer_params = generate_data_params_to_infer_params(params,
                                                      prior_on_hypers,
                                                      generate_data_filename)
  infer_unit = Multinomial( ripl_thunk(), infer_params)


  ## Params for dataset_get_observes
  ## ----------------------------
  
  dataset = 1
  name = 'onebird'

  if len(sys.argv) > 3:
    days_list = range(int(sys.argv[2]))
    transitions = sys.argv[3]
  else:
    days_list = range(4)
    transitions = 100

  years_list = range(1)  

  load_observe_sub_range = Observe_range(days_list, years_list, range(infer_unit.cells) )
  use_range_defaults = False

  observations = dataset_get_observes(dataset, name, load_observe_sub_range,
                                      use_range_defaults)
  ## we should have an observations object that contains an observe_range


  ##  Params for Inference
  ## ----------------------------
  def mse_onebird_hypers(unit):    
    hypers = get_hypers(unit)
    return {'hyper_mse': mse(hypers,np.array([5,10,10,10])),
            'hypers':hypers}

  inference_prog = transitions_to_mh_default(transitions=100)
  score_function = mse_onebird_hypers
  observes_filename = None

  scores = incremental_observe_infer( infer_unit, observes_filename, load_observe_sub_range,
                                      inference_prog, infer_every_cell=False, gtruth_unit = None,
                                      score_function = score_function,
                                      observations = observations)
  ## Print key params
  ## ----------------------------
  print '\n\n PARAMS: '
  print dict(dataset=dataset, name=name,
             days_list=days_list, mh_transitions=transitions)
             
  return scores
  
# ## TEMP TESTING OF ONEBIRD, TODO NOSE-IFY
# if __name__ == '__main__':
#   print 'sys args:  name, None, days_list, mh_transitions, time.sleep(arg)\n'
#   print 'sys.argv', sys.argv, '\n\n'
  
#   import time
#   time.sleep(int(sys.argv[4]))
#   print '\n\n SCORES \n', onebird()

  

def generate_unit_to_incremental_infer( generate_data_unit, load_observe_range,
                                        prior_on_hypers, inference_prog, infer_every_cell,
                                        score_function = None):

  # Check arguments
  full_observe_range = generate_data_unit.get_max_observe_range()
  load_observe_range.assert_is_observe_sub_range(full_observe_range)
  if load_observe_range is None:
      load_observe_range = full_observe_range
  
  assert len(prior_on_hypers) == generate_data_unit.params['num_features']

  # Save observes from data-generating model
  generate_data_filename, _ = store_observes(generate_data_unit, load_observe_range)

  # Get infer_unit
  model_constructor = generate_data_unit.__class__
  infer_unit = make_infer_unit(generate_data_filename,
                               prior_on_hypers,
                               None,   # default to same ripl thunk
                               model_constructor)

  observations = synthetic_get_observes(generate_data_filename, load_observe_range)
  
  scores = incremental_observe_infer(infer_unit, generate_data_filename,
                                     load_observe_range, inference_prog, infer_every_cell,
                                     score_function = score_function,
                                     observations = observations)

  return infer_unit, scores




def _test_incremental_infer( generate_data_unit, load_observe_range, prior_on_hypers, inference_prog, infer_every_cell, score_function):

  infer_unit, scores = generate_unit_to_incremental_infer( generate_data_unit,
                                                           load_observe_range,
                                                           prior_on_hypers,
                                                           inference_prog,
                                                           infer_every_cell,
                                                           score_function)
  
  print  '\n\n _test_incremental_infer: short_name', generate_data_unit.params['short_name']
  print 'score_function', score_function.__doc__
  print '\n score_before_inference', scores['before']
  print '\n score_after_inference', scores['after']
  
  for score_key, score_before_value in scores['before'].items():
    assert score_before_value > scores['after'][score_key]
  



def get_input_for_incremental_infer( ):

  def gtruth_unit_to_mse_latents(gtruth_unit,observe_range):
    def score_function(infer_unit):
      'mse latents'
      return dict(latents=compare_latents(gtruth_unit, infer_unit, observe_range))
    return score_function


  def gtruth_unit_to_mse_hypers(gtruth_unit):
    def score_function(infer_unit):
      'mse hypers'
      return dict(hypers=compare_hypers(gtruth_unit, infer_unit)['mse'])
    return score_function


  def gtruth_unit_to_mse_both(gtruth_unit,observe_range):
    def score_function(infer_unit):
      'mse latents, hypers'
      return dict(latents=compare_latents(gtruth_unit, infer_unit, observe_range),
                  hypers = compare_hypers(gtruth_unit, infer_unit)['mse'])

    return score_function

        

  def make_multinomial_size33(ripl_thunk, load_observe_range, prior_string, mh_transitions, infer_every_cell,
                              score_function_constructor=None):
      
    def thunk():
      params_short_name = 'multinomial_onestep_diag105_size33'
      generate_data_unit = Multinomial( ripl_thunk(), make_params( params_short_name) )
      
      load_observe_range = Observe_range(years_list=range(1), days_list=range(3),
                                         cells_list=range(generate_data_unit.cells))

      prior_on_hypers = [prior_string] * generate_data_unit.params['num_features']
      inference_prog = transitions_to_mh_default(transitions=mh_transitions)

      if score_function_constructor is None:
        score_function = gtruth_unit_to_mse_both(generate_data_unit, load_observe_range)

      return generate_data_unit, load_observe_range, prior_on_hypers, inference_prog, infer_every_cell, score_function

    return thunk

  
  thunk0 = make_multinomial_size33(mk_p_ripl, None, '(uniform_continuous 0.01 20)', 10, True)


  def thunk1():
    params_short_name = 'poisson_onestep_diag105_size33'
    model_constructor = Poisson
    ripl_thunk = mk_p_ripl
    generate_data_unit = model_constructor( ripl_thunk(), make_params( params_short_name) )
    load_observe_range = Observe_range(years_list=range(1), days_list=range(3),
                                       cells_list=range(generate_data_unit.cells))
    num_features = generate_data_unit.params['num_features']
    prior_on_hypers = ['(uniform_continuous 0.01 20)'] * num_features
    inference_prog = transitions_to_cycle_mh( transitions_latents=10, transitions_hypers=10, number_of_cycles=1)

    infer_every_cell = False
    score_function = gtruth_unit_to_mse_both(generate_data_unit, load_observe_range)
    return generate_data_unit, load_observe_range, prior_on_hypers, inference_prog, infer_every_cell, score_function


  def thunk2():
    params_short_name = 'dataset1'
    model_constructor = Multinomial
    ripl_thunk = mk_p_ripl
    generate_data_unit = model_constructor( ripl_thunk(), make_params( params_short_name) )
    load_observe_range = Observe_range(years_list=range(1), days_list=range(4),
                                       cells_list=range(generate_data_unit.cells))
    num_features = generate_data_unit.params['num_features']
    prior_on_hypers = ['(uniform_continuous 1 20)'] * num_features
    inference_prog = transitions_to_mh_default(transitions=200)
    infer_every_cell = False
    score_function = gtruth_unit_to_mse_hypers(generate_data_unit)
    return generate_data_unit, load_observe_range, prior_on_hypers, inference_prog, infer_every_cell, score_function

## FIXME
  return [thunk0, thunk1, thunk2]

  

def test_all_incremental_infer():
  thunks = get_input_for_incremental_infer()
  for t in thunks:
    generate_data_unit, load_observe_range, prior_on_hypers, inference_prog, infer_every_cell, score_function = t()

    yield  _test_incremental_infer,  generate_data_unit, load_observe_range, prior_on_hypers, inference_prog, infer_every_cell, score_function

def _test_one_incremental_infer( index ):
  thunk = get_input_for_incremental_infer()[index]
  _test_incremental_infer( *thunk() )



def run_all():
  regular_tests = ()
  generative_tests = ( lambda: test_all_incremental_infer(), )

  return run_regular_and_generative_nosetests( regular_tests, generative_tests)
    
  










  
