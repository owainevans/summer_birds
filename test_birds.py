import numpy as np
import os, subprocess
from venture.shortcuts import make_puma_church_prime_ripl as mk_p_ripl
from nose.tools import eq_, assert_almost_equal

from utils import make_grid
from features_utils import ind_to_ij, make_features_dict, cell_to_prob_dist
from synthetic import get_multinomial_params
from model import Multinomial, Poisson

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
  unit.load_assumes()
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
  





