import numpy as np
from itertools import product
from nose.tools import eq_, assert_almost_equal

from venture.shortcuts import make_puma_church_prime_ripl as mk_p_ripl
from venture.shortcuts import make_lite_church_prime_ripl as mk_l_ripl
from utils import  Observe_range

from model import *
from mytest_utils import *


## SCORE FUNCTIONS


def compare_hypers(gtruth_unit,infer_unit):
  'Compare hypers across two different Birds unit objects'

  def get_hypers(unit):
    ripl = unit.ripl
    num_features = unit.params['num_features']
    return np.array([ripl.sample('hypers%i'%i) for i in range(num_features)])

  hypers = []
  logscores = []
  
  for unit in (gtruth_unit, infer_unit):
    hypers.append( get_hypers(unit) )
    logscores.append( unit.ripl.get_global_logscore() )

  return dict(hypers=hypers, logscores=logscores, mse=mse(*hypers))


def mse(hypers1,hypers2): return np.mean((hypers1-hypers2)**2)


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
                               inference_prog, infer_every_cell=False, gtruth_unit = None):
  
  cells_list = observe_range['cells_list']
  
  for year in observe_range['years_list']:
    for day in observe_range['days_list']:

      if not infer_every_cell:
        observe_sub_range = Observe_range(years_list=[year], days_list=[day], cells_list=cells_list)
        # slow because reading file every time
        load_observes( unit, observe_sub_range, False, observes_filename) 

        inference_prog( unit, year, day, 'all', gtruth_unit)

      else:
        for cell in observe_range['cells_list']:
          observe_sub_range = Observe_range(years_list=[year], days_list=[day], cells_list=[cell])
          load_observes( unit, observe_sub_range, False, observes_filename)
          inference_prog( unit, year, day, cell, gtruth_unit)



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
  
  score_before_inference = score_function(infer_unit)

  incremental_observe_infer(infer_unit, generate_data_filename, load_observe_range, inference_prog, infer_every_cell)
  score_after_inference = score_function(infer_unit)

  return infer_unit, score_before_inference, score_after_inference



def _test_incremental_infer( generate_data_unit, load_observe_range, prior_on_hypers, inference_prog, infer_every_cell, score_function):

  out = generate_unit_to_incremental_infer( generate_data_unit,
                                            load_observe_range,
                                            prior_on_hypers,
                                            inference_prog,
                                            infer_every_cell,
                                            score_function)

  infer_unit, score_before_inference, score_after_inference = out
  
  print  '\n\n _test_incremental_infer: short_name', generate_data_unit.params['short_name']
  print 'score_function', score_function.__doc__
  print '\n score_before_inference', score_before_inference
  print '\n score_after_inference', score_after_inference
  
  for k,v in score_before_inference.items():
    assert v < score_after_inference[k]
  


def get_input_for_incremental_infer( ):

  def gtruth_unit_to_mse_latents(gtruth_unit,observe_range):
    def score_function(infer_unit):
      'mse latents'
      return compare_latents(gtruth_unit, infer_unit, observe_range)
    return score_function

  def gtruth_unit_to_mse_hypers(gtruth_unit):
    def score_function(infer_unit):
      'mse hypers'
      return compare_hypers(gtruth_unit, infer_unit)['mse']
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

  
  thunk0 = make_multinomial_size33(mk_p_ripl, None, '(uniform_continuous 0.01 20)', 3, True)


  def thunk1():
    params_short_name = 'poisson_onestep_diag105_size33'
    model_constructor = Poisson
    ripl_thunk = mk_p_ripl
    generate_data_unit = model_constructor( ripl_thunk(), make_params( params_short_name) )
    load_observe_range = Observe_range(years_list=range(1), days_list=range(3),
                                       cells_list=range(generate_data_unit.cells))
    num_features = generate_data_unit.params['num_features']
    prior_on_hypers = ['(uniform_continuous 0.01 10)'] * num_features
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


  return [thunk0, thunk1] #, thunk2]
    
  

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

  return run_regular_and_generative_tests( regular_tests, generative_tests)
    
  










  
