import numpy as np
import os, subprocess
from venture.shortcuts import make_puma_church_prime_ripl as mk_p_ripl
from venture.shortcuts import make_lite_church_prime_ripl as mk_l_ripl
from nose.tools import eq_, assert_almost_equal
import cPickle as pickle

from itertools import product
from utils import make_grid
from features_utils import ind_to_ij, make_features_dict, cell_to_prob_dist
from synthetic import get_multinomial_params
from model import *

def test_make_grid():
  mk_array = lambda ar: np.array(ar,np.int32)

  def ar_eq_(ar1,ar2):
    'Asssert equality of two arrays'
    for pair in zip( ar1.flatten(), ar2.flatten() ):
      eq_(*pair)
  
  pairs = ( ( ([0,3],[1,4],[2,5]), 
              make_grid(3,2, top0=True, lst = range(6), order='F') ),
            
            ( ([1,3],[0,2]),
              make_grid( 2, 2,top0=False, lst = range(4), order='F') ),
            
            ( ([0,1],[2,3]),
              make_grid( 2, 2, top0=True, lst = range(4), order='C') ) )
  
  for ar,grid in pairs:
    ar_eq_( mk_array(ar), grid)

  
def test_ind_to_ij():
  height, width = 3,2
  grid = make_grid(height,width,lst=range(height*width),order='F')
  for ind in range(height*width):
    ij = tuple( ind_to_ij(height,width,ind,'F') )
    eq_( grid[ij], ind )

def test_make_features_dict():
  height, width = 3,2
  years,days = range(2), range(2)
  args = (height, width, years, days)
  name = 'one_step_and_not_diagonal'
  venture_dict, python_dict = make_features_dict(*args, feature_functions_name=name )
  eq_( len(python_dict), (height*width)**2 * ( len(years)*len(days) ) )
  assert isinstance( python_dict[ (0,0,0,0) ], (list,tuple) )
  eq_( venture_dict['type'], 'dict' )
  assert isinstance(venture_dict['value'],dict)


def test_features_functions():
  args_names = ( ('height',2), ('width',3), ('years',range(1)), ('days',range(1)), )
  _, args = zip(*args_names)
  num_cells = args[0] + args[1]

  feature_functions_names = ('uniform', 'distance', 'not_diagonal')
  #  only one feature per set of functions, but we'll have singleton list of features
  
  feature_dicts = {}
  for name in feature_functions_names:
    feature_dicts[name] = make_features_dict( *args, feature_functions_name = name)[1]

  is_constant = lambda seq: len( np.unique( seq) ) == 1
  assert is_constant( [v[0] for v in feature_dicts['uniform'].values()] )

  assert feature_dicts['not_diagonal'][(0,0,0,0)][0] == 0  # first cell is on diagonal

  ## FIXME get this working, something funky with indices?
  # def distances_from_i(i):
  #   distance_dict = feature_dicts['distance']
  #   return [distance_dict[(0,0,i,j)][0] for j in range(num_cells)]

  # sum_distances = [sum(distances_from_i(i)) for i in (0, num_cells-1) ]  
  # eq_( *sum_distances )

  
def make_multinomial_unit( params_short_name = 'minimal_onestepdiag10', ripl_thunk = None):
  if ripl_thunk is None:
    ripl_thunk = mk_p_ripl
  
  return Multinomial( ripl_thunk(), make_params( params_short_name) )


def example_make_infer(generate_data_unit, observe_range = None ):
  generate_data_params = generate_data_unit.params

  if observe_range is None:
    observe_range = dict(  years_list = range(1),
                           days_list= range(1),
                           cells_list = None )
  
  out = generate_data_unit.store_observes(observe_range)
  generate_data_store_dict_filename, generate_data_draw_bird_filename = out
  ## FIXME, we don't need draw_bird_filename and so can ignore this

  prior_on_hypers = ['(gamma 1 1)'] * generate_data_params['num_features']
  infer_unit = make_infer_unit( generate_data_store_dict_filename, prior_on_hypers, True)

  return observe_range, generate_data_unit, generate_data_store_dict_filename, infer_unit
  
  
def test_cell_to_prob_dist( unit ):
  height, width, ripl = unit.height, unit.width, unit.ripl
  cells = height * width
  for cell in range(cells):
    grid = cell_to_prob_dist( height, width, ripl, cell, 0, 0, order='F' )
    assert_almost_equal( np.sum(grid), 1)

    
def test_model_multinomial( unit ):
  
  simplex = unit.ripl.sample('(get_bird_move_dist 0 0 0)',type=True)
  eq_( simplex['type'], 'simplex')
  eq_( len( simplex['value'] ), unit.cells)

  # bird with bird_id=0 is at pos_day1 on day 1, so total birds
  # at cell is >= 1
  pos_day1 = unit.ripl.predict('(get_bird_pos 0 0 1)')
  count_pos_day1 = unit.ripl.predict('(count_birds 0 1 %i)'%pos_day1)
  assert 1 <= count_pos_day1 <= unit.num_birds

  # assuming all birds start at zero on d=0
  eq_( unit.ripl.predict('(move 0 0 0 0)'), pos_day1 )


  # observe and infer should change position of bird0
  # - observe no bird at pos_day1
  unit.ripl.observe('(observe_birds 0 1 %i)'%pos_day1,'0.')
  total_transitions = 0
  transitions_chunk = 50
  while total_transitions < 500:
    unit.ripl.infer(transitions_chunk)
    new_pos_day1 = unit.ripl.predict('(get_bird_pos 0 0 1)')
    if new_pos_day1 != pos_day1:
      break
    total_transitions += transitions_chunk
    if total_transitions >= 500:
      assert False,'Did total_transitions without changing bird_pos'
    
  
def test_make_infer( generate_data_unit ):
  out = example_make_infer( generate_data_unit )
  _, generate_data_unit, generate_data_filename, infer_unit = out
  generate_data_params = generate_data_unit.get_params()
  infer_params = infer_unit.get_params()

  # is infer_params mostly same as generate_data_params?
  for k,v in generate_data_params.items():
    if k not in ('ripl_directives','prior_on_hypers',
                 'learn_hypers','observes_loaded_from'):
      eq_( v, infer_params[k] )

  infer_unit.ensure_assumes()

  # do constants agree for generate_data_unit and infer_unit?
  expressions = ('features', 'num_birds', '(phi 0 0 0 0)')
  for exp in expressions:
    eq_( generate_data_unit.ripl.sample(exp), infer_unit.ripl.sample(exp) )



def test_memoization_observe( unit ):
  
  num_tries = 100
  
  for _ in range(num_tries):
    r = unit.ripl

    pred_val = r.predict('(observe_birds 0 0 0)')
    eq_( pred_val, r.predict('(observe_birds 0 0 0)') )
    
    obs_val = 5
    if obs_val != pred_val:
      r.observe('(observe_birds 0 0 0)', obs_val)
      eq_( pred_val, r.predict('(observe_birds 0 0 0)') )

      r.infer(1)
      eq_( obs_val, r.predict('(observe_birds 0 0 0)') )

      r.infer(100)
      eq_( obs_val, r.predict('(observe_birds 0 0 0)') )

      break
  

 
def compare_observes( first_unit, second_unit, triples ):
  'Pass asserts if unit.ripls agree on all triples'

  # one infer on each ripl to ensure observes are 'registered'
  [unit.ripl.infer(1) for unit in (first_unit, second_unit) ]

  def predict_observe( unit,y,d,i):
    return unit.ripl.predict('(observe_birds %i %i %i)'%(y,d,i))
  
  for y,d,i in triples:
    print 'cf:',predict_observe( first_unit, y,d,i), predict_observe( second_unit, y,d,i)

    eq_( predict_observe( first_unit, y,d,i),
         predict_observe( second_unit, y,d,i), )

    
def make_triples( observe_range ):
  return product(observe_range['years_list'],
                 observe_range['days_list'],
                 observe_range['cells_list'] )



def test_load_observations( generate_data_unit ):
  
  out = example_make_infer( generate_data_unit )
  observe_range, generate_data_unit, store_dict_filename, infer_unit = out

  infer_unit.load_observes(observe_range, store_dict_filename)

  if observe_range['cells_list'] is None:
    observe_range['cells_list'] = range( infer.unit.cells )
    
  ydi = make_triples( observe_range )
  # do values for *observe_birds* agree for generate_data_unit
  # and infer_unit?
  compare_observes( generate_data_unit, infer_unit, ydi )


    
def test_incremental_load_observations( generate_data_unit):

  observe_range, generate_data_unit, store_dict_filename, infer_unit = example_make_infer( generate_data_unit)

  for cell in range(infer_unit.cells):
    updated_observe_range = observe_range.copy()
    updated_observe_range.update( dict(cells_list = [cell] ) )
    print updated_observe_range
    infer_unit.load_observes( updated_observe_range, store_dict_filename)

    ydi = make_triples(updated_observe_range)
    print '\n',ydi
    compare_observes( generate_data_unit, infer_unit, ydi)
            
    infer_unit.ripl.infer(10)
    
     
def test_save_images(unit, del_images=True):
  years = range(1)
  days = range(1)
  directory = 'tmp_test_bird_moves_/'
  filename = directory + 'temp_test_save.png'
  #os.chdir('~/summer_birds')
  unit.draw_bird_locations(years, days, 'test', save=True, order='F', print_features_info=True,
                           directory_filename = (directory, filename) )
  assert os.path.exists( directory)
  if del_images: subprocess.call(['rm','-r',directory])
  


def test_save_load_multinomial( ripl_thunk, make_params_thunk ):

  
  def equality_multinomial(u1, u2):
    'Equality for Multinomial objects with predict'
    test_lambdas = (lambda u: u.params,
                    lambda u: u.ripl.list_directives())

    bools = [ f(u1)==f(u2) for f in test_lambdas ]
    return all(bools)

  def print_random_draws(u1, u2):
    print 'compare beta(1 1)',
    print map( lambda u: u.ripl.sample('(beta 1 1)'), (u1,u2) )


  def make_unit_with_predict():
    unit = Multinomial( ripl_thunk(), make_params_thunk() )
    predicts = ( '(observe_birds 0 0 0)',
                 '(observe_birds 0 1 0)',
                 '(observe_birds 0 1 1)',
                 '(observe_birds 0 1 2)',
                 '(observe_birds 0 2 0)', )
    [unit.ripl.predict(exp) for exp in predicts]
    return unit
    
  original_unit = make_unit_with_predict()
  original_unit.ripl.infer(20)
  original_filename = original_unit.save('temp_test')
    
  copy_unit = make_unit_with_predict().make_saved_model(original_filename)
  
  # loaded copy equals original
  assert equality_multinomial( original_unit, original_unit)
  assert equality_multinomial( original_unit, copy_unit)
  print_random_draws( original_unit, copy_unit)

  # do more inference on original unit. save and load. assert unequal.
  original_unit.ripl.infer(20)
  filename_more_infer = original_unit.save('temp_test_more_infer')
  copy_unit_more_infer = make_unit_with_predict().make_saved_model(filename_more_infer)

  # updated original unit equals loaded copy of it
  assert equality_multinomial( original_unit, copy_unit_more_infer)
  print_random_draws( original_unit, copy_unit_more_infer)

  # copy of updated original unit not equal to copy of non-updated
  assert not equality_multinomial( copy_unit, copy_unit_more_infer)
  print_random_draws( copy_unit, copy_unit_more_infer)
  
  # but they do have same params and all but predicts
  copies = ( copy_unit, copy_unit_more_infer )
  assert copies[0].params == copies[1].params
  
  directives = [u.ripl.list_directives() for u in copies]
                                                  
  for d1,d2 in zip(*directives):
    if d1['instruction']!='predict':
      assert d1 == d2




def test_all_multinomial_unit_params( puma = None):
  tests =  (test_model_multinomial,
            test_cell_to_prob_dist,
            test_make_infer,
            test_memoization_observe,
            test_load_observations,
            test_incremental_load_observations,
            test_save_images, )
  
  ripl_thunks = (mk_p_ripl, mk_l_ripl)
  if puma: ripl_thunks = (mk_p_ripl,)
  
  params_short_names = ('minimal_onestepdiag10', 'bigger_onestep_diag105')

  for test, params_short_name, ripl_thunk in product(tests,params_short_names, ripl_thunks):
    test( make_multinomial_unit( params_short_name, ripl_thunk) )

  make_params_thunks = [ lambda:make_params( name ) for name in params_shot_names ]
  
  for ripl_thunk, make_params_thunk in product( ripl_thunks, make_params_thunk):
    test_save_load_multinomial( ripl_thunk, make_params_thunk )



def all_tests():
 
  test_make_features_dict()
  test_features_functions()
  test_ind_to_ij()
  test_make_grid()
 
  test_all_multinomial_unit_params()

  print 'passed all tests'
  





