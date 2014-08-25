import numpy as np
import os, subprocess
from venture.shortcuts import make_puma_church_prime_ripl as mk_p_ripl
from nose.tools import eq_, assert_almost_equal
import cPickle as pickle

from itertools import product
from utils import make_grid
from features_utils import ind_to_ij, make_features_dict, cell_to_prob_dist
from synthetic import get_multinomial_params
from model import Multinomial, Poisson, make_params

def test_make_grid():
  mk_array = lambda ar: np.array(ar,np.int32)

  def ar_eq_(ar1,ar2):
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


  
def make_multinomial_unit():
  params = get_multinomial_params(params_name = 'easy_hypers' )
  unit =  Multinomial(mk_p_ripl(),params)
  
  return unit

  
def test_cell_to_prob_dist():
  unit = make_multinomial_unit()
  height, width, ripl = unit.height, unit.width, unit.ripl
  cells = height * width
  for cell in range(cells):
    grid = cell_to_prob_dist( height, width, ripl, cell, 0, 0, order='F' )
    assert_almost_equal( np.sum(grid), 1)

    
def test_model_multinomial():
  unit = make_multinomial_unit()
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
    


# UTILS FOR MAKING INFER UNIT OBJECTS BASED ON SAVED OBSERVES
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


def compare_observes( first_unit, second_unit, triples ):
  'Pass asserts if unit.ripls agree on all triples'
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

## FIXME, failing, incremental is also
def test_load_observations():
  
  observe_range, generate_data_unit, path_filename, infer_unit = example_make_infer()

  infer_unit.load_observes(observe_range, path_filename)

  if observe_range['cells_list'] is None:
    observe_range['cells_list'] = range( infer.unit.cells )
    
  ydi = make_triples( observe_range )
  # do values for *observe_birds* agree for generate_data_unit
  # and infer_unit?
  compare_observes( generate_data_unit, infer_unit, ydi )


    
def test_incremental_load_observations():
  ## FIXME
  observe_range, generate_data_unit, path_filename, infer_unit = example_make_infer()

  for cell in range(infer_unit.cells):
    updated_observe_range = observe_range.copy()
    updated_observe_range.update( dict(cells_list = [cell] ) )
    print updated_observe_range

    ydi = make_triples(updated_observe_range)
    print '\n',ydi
    compare_observes( generate_data_unit, infer_unit, ydi)
            
    infer_unit.ripl.infer(10)
    
     



def test_save_images(del_images=True):
  unit = make_multinomial_unit()
  years = range(1)
  days = range(1)
  name = 'temp_test_save'
  path = 'bird_moves_' + name
  #os.chdir('~/summer_birds')
  unit.draw_bird_locations(years, days, name, save=True, order='F', print_features_info=True)
  assert os.path.exists( path )
  if del_images: subprocess.call(['rm','-r',path])
  

def all_tests():
  test_cell_to_prob_dist()
  test_make_features_dict()
  test_features_functions()
  test_ind_to_ij()
  test_make_grid()
  test_model_multinomial()
  test_save_images()
  





