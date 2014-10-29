import numpy as np
import os, subprocess, sys
import cPickle as pickle
from timeit import timeit
from itertools import product
from nose.tools import eq_, assert_almost_equal

from venture.shortcuts import make_puma_church_prime_ripl as mk_p_ripl
from venture.shortcuts import make_lite_church_prime_ripl as mk_l_ripl
from utils import make_grid, Observe_range
from features_utils import ind_to_ij, make_features_dict, cell_to_prob_dist
from synthetic import get_multinomial_params
from model import *


# venture value dicts (python/lib

def test_make_grid():
  'Makes 2D grid from list using F or C order'
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
  '''Consistency of make_grid (which maps list to 2D grid) and ind_to_ij
     which maps list index to ij position on grid'''
  height, width = 3,2
  grid = make_grid(height,width,lst=range(height*width),order='F')
  for ind in range(height*width):
    ij = tuple( ind_to_ij(height,width,ind,'F') )
    eq_( grid[ij], ind )

def _test_make_features_dict(ripl_thunk):
  '''Type and basic sanity checks for making one_step_not_diagonal
  features'''
  height, width = 3,2
  years,days = range(2), range(2)
  args = (height, width, years, days)
  name = 'one_step_and_not_diagonal'

  # Create venture stack dict (*venture_dict*) and venture expression (*venture_exp*)
  venture_dict, python_dict = make_features_dict(*args, feature_functions_name=name, dict_string='dict' )
  venture_exp, _ = make_features_dict(*args, feature_functions_name=name, dict_string='string')

  eq_( len(python_dict), (height*width)**2 * ( len(years)*len(days) ) )
  assert isinstance( python_dict[ (0,0,0,0) ], (list,tuple) )
  eq_( venture_dict['type'], 'dict' )

  ripl = ripl_thunk()
  ripl.assume('feature_exp', venture_exp)
  ripl.assume('feature_stack_dict', venture_dict)
  
  cell00 = []
  cell01 = []

  for var_name in ('feature_exp','feature_stack_dict'):
    cell00.append( ripl.sample('(lookup %s (array 0 0 0 0))' % var_name)[0] )
    cell01.append( ripl.sample('(lookup %s (array 0 0 0 1))' % var_name)[0] )
    
  eq_(*cell00)
  eq_(*cell01)

  
def test_features_functions():
  'Do *uniform*, *not_diagonal* behave as expected? Uses *make_features_dict*'
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


  
def make_unit_instance( model_constructor, ripl_thunk, params_short_name = 'minimal_onestep_diag10'):

  return model_constructor( ripl_thunk(), make_params( params_short_name) )



def example_make_infer(generate_data_unit, observe_range, ripl_thunk ):
  generate_data_params = generate_data_unit.params
  
  generate_data_store_dict_filename,_ = store_observes(generate_data_unit,
                                                       observe_range)

  prior_on_hypers = ['(gamma 1 1)'] * generate_data_params['num_features']
  if multinomial_or_poisson( generate_data_unit) == 'Multinomial':
    model_constructor = Multinomial
  else:
    model_constructor = Poisson
    
  infer_unit = make_infer_unit( generate_data_store_dict_filename,
                                prior_on_hypers,
                                ripl_thunk,
                                model_constructor )

  return observe_range, generate_data_unit, generate_data_store_dict_filename, infer_unit
  
  
def _test_cell_to_prob_dist( unit ):
  height, width, ripl = unit.params['height'], unit.params['width'], unit.ripl
  cells = height * width
  for cell in range(cells):
    grid = cell_to_prob_dist( height, width, ripl, cell, 0, 0, order='F' )
    assert_almost_equal( np.sum(grid), 1)


def _test_model( unit ):  
  
  simplex = unit.ripl.sample('(get_bird_move_dist 0 0 0)',type=True)
  eq_( simplex['type'], 'simplex')
  eq_( len( simplex['value'] ), unit.cells)
  
  sum_phi = unit.ripl.sample('(sum_phi 0 0 0 )')
  phi0 = simplex['value'][0] * sum_phi
  assert_almost_equal( phi0, unit.ripl.sample('(phi 0 0 0 0)') )

  # bird with bird_id=0 is at pos_day1 on day 1, so total birds
  # at cell is >= 1
  if not isinstance(unit,Poisson):
    pos_day1 = unit.ripl.predict('(get_bird_pos 0 0 1)')
    count_pos_day1 = unit.ripl.predict('(count_birds 0 1 %i)'%pos_day1)
    assert 1 <= count_pos_day1 <= unit.params['num_birds']

    # assuming all birds start at zero on d=0
    eq_( unit.ripl.predict('(move 0 0 0 0)'), pos_day1 )


    # observe and infer should change position of bird0
    # - observe no bird at pos_day1
    unit.ripl.observe('(observe_birds 0 1 %i)'%pos_day1,'0.')
    total_transitions = 0
    transitions_chunk = 10
    while total_transitions < 100:
      unit.ripl.infer(transitions_chunk)
      new_pos_day1 = unit.ripl.predict('(get_bird_pos 0 0 1)')
      if new_pos_day1 != pos_day1:
        break
        total_transitions += transitions_chunk
        if total_transitions >= 500:
          assert False, 'Performed %i without changing bird_pos' % total_transitions

  else: # Poisson model
    # no birds outside first cell on day 0
    num_birds = unit.ripl.predict('(count_birds 0 0 0)')
    eq_(num_birds, unit.ripl.sample('num_birds'))
    eq_(unit.ripl.predict('(count_birds 0 0 1)'), 0)

    # inference works
    unit.ripl.infer(10)
    
    
    
  
def _test_make_infer( generate_data_unit, ripl_thunk ):
  observe_range = generate_data_unit.get_max_observe_range()
  
  out = example_make_infer( generate_data_unit, observe_range, ripl_thunk)
  _, generate_data_unit, generate_data_filename, infer_unit = out
  generate_data_params = generate_data_unit.get_params()
  infer_params = infer_unit.get_params()

  # is infer_params mostly same as generate_data_params?
  for k,v in generate_data_params.items():
    if k not in ('ripl_directives','prior_on_hypers',
                 'learn_hypers','observes_loaded_from',
                 'short_name', 'long_name'):
      eq_( v, infer_params[k] )

  infer_unit.ensure_assumes()

  # do constants agree for generate_data_unit and infer_unit?
  expressions = ('features', 'num_birds')
  for exp in expressions:
    eq_( generate_data_unit.ripl.sample(exp), infer_unit.ripl.sample(exp) )



def _test_memoization_observe( unit ):
  r = unit.ripl
  
  pred_val = r.predict('(observe_birds 0 0 0)')
  eq_( pred_val, r.predict('(observe_birds 0 0 0)') )

  obs_val = pred_val + 1

  r.observe('(observe_birds 0 0 0)', obs_val)
  r.infer(1)
  eq_( obs_val, r.predict('(observe_birds 0 0 0)') )

  

 
def compare_observes( first_unit, second_unit, triples ):
  'Pass asserts if unit.ripls agree on all triples'

  def predict_observe( unit, year_day_cell):
    return unit.ripl.predict('(observe_birds %i %i %i)'% tuple(year_day_cell))
  
  print '\n compare_observes:'
  

  for year_day_cell in triples:
    print '\n triple %s \n cf.'%str( year_day_cell )
    print map( lambda u: predict_observe(u, year_day_cell),
               (first_unit, second_unit) )

    eq_( predict_observe( first_unit, year_day_cell),
         predict_observe( second_unit, year_day_cell) )

    
def make_triples( observe_range ):
  return product(observe_range['years_list'],
                 observe_range['days_list'],
                 observe_range['cells_list'] )


def load_observations_vars(generate_data_unit, ripl_thunk):

  observe_range = generate_data_unit.get_max_observe_range()

  out = example_make_infer( generate_data_unit, observe_range, ripl_thunk)
  observe_range, _, store_dict_filename, infer_unit = out

  return observe_range, store_dict_filename, infer_unit


def register_observes( ripl ):
  ripl.infer(1)


def _test_load_observations( generate_data_unit, ripl_thunk ):
                                                                            
  observe_range, store_dict_filename, infer_unit = load_observations_vars( generate_data_unit,
                                                                           ripl_thunk )

  use_defaults = False
  load_observes(infer_unit, observe_range, use_defaults, store_dict_filename)
  register_observes( infer_unit.ripl )
    
  year_day_cells = make_triples( observe_range )
  print '-----------'
  for el in year_day_cells:
    print '\nel:', el

  year_day_cell_iter = make_triples( observe_range )

  # do values for *observe_birds* agree for generate_data_unit
  # and infer_unit?
  compare_observes( generate_data_unit, infer_unit, year_day_cell_iter )


    
def _test_incremental_load_observations( generate_data_unit, ripl_thunk):

  observe_range, store_dict_filename, infer_unit = load_observations_vars( generate_data_unit,
                                                                           ripl_thunk )

  # add observes to infer_unit cell by cell
  cells = range(  min( 5, infer_unit.cells) )
  for cell in cells:
    updated_observe_range = observe_range.copy_observe_range()
    updated_observe_range.update( dict(cells_list = [cell] ) )
    
    use_defaults = False
    load_observes( infer_unit, updated_observe_range, use_defaults, store_dict_filename)
    register_observes( infer_unit.ripl )

    year_day_cell_iter = make_triples(updated_observe_range)
    compare_observes( generate_data_unit, infer_unit, year_day_cell_iter)
            
    

def _test_dataset_load_observations(ripl_thunk):

  def load_check_directives(unit, dataset_number, name, observe_sub_range, use_range_defaults):
    dataset_load_observations(unit, dataset_number, name, observe_sub_range, use_range_defaults)
    register_observes( unit.ripl)
    last_directive = unit.ripl.list_directives()[-1]
    eq_(last_directive['instruction'], 'observe')
    assert isinstance(last_directive['value'], (float,int))

  ## Dataset 1: Multinomial model
  dataset_short_name = 'dataset1'
  unit = Multinomial( ripl_thunk(), make_params(dataset_short_name))
  name = 'onebird'
  observe_sub_range = None
  use_range_defaults = True
  load_check_directives(unit, 1, name, observe_sub_range,
                        use_range_defaults)

 # FIXME TOO SLOW FOR NOW
 # Dataset 2/3: Poisson model
  # for dataset_short_name in ('dataset2',):
  #   dataset_number = int(dataset_short_name[-1])
  #   unit = Poisson( ripl_thunk(), make_params(dataset_short_name) )
  #   name = '10x10x1000-train'
  #   observe_sub_range = Observe_range(years_list=range(1), days_list=range(1),
  #                                     cells_list=range(unit.cells))
  #   use_range_defaults = False
  #   load_check_directives(unit, dataset_number, name, observe_sub_range, use_range_defaults)


     
def _test_save_images(unit, del_images=True):
  years = range(1)
  days = range(1)
  directory = 'tmp_test_bird_moves_/'
  filename = directory + 'temp_test_save.png'
  #os.chdir('~/summer_birds')
  plot_save_bird_locations(unit, years, days, title='test', save=True, order='F', verbose=True,
                           directory_filename = (directory, filename) )
  assert os.path.exists( directory)
  if del_images: subprocess.call(['rm','-r',directory])
  


def _test_save_load_model( model_constructor, ripl_thunk, make_params_thunk, verbose=False):
  'Save and Load Methods for unit instance. Test object equality.'
  
  def equality_unit(u1, u2, verbose):
    'Equality for Unit objects with predict'

    def test_equality(u1,u2):
      test = [ u1.params == u2.params,
               u1.ripl.list_directives() == u2.ripl.list_directives()]
      return all(test)

    def find_unequal_directive(u1,u2):
      zip_directives = zip(u1.ripl.list_directives(), u2.ripl.list_directives() )
      for directive1, directive2 in zip_directives:
        if directive1 != directive2:
          print 'Failed equality', directive1, directive2
          return False
    
    if verbose: print_random_draws(u1,u2)
    return True if test_equality(u1,u2) else find_unequal_directive(u1,u2)


  def print_random_draws(u1, u2):
    print '\n Compare units on beta(1 1)'
    print map(lambda u: u.ripl.sample('(beta 1 1)'), (u1,u2) )


  def make_unit_with_predict():
    unit = model_constructor( ripl_thunk(), make_params_thunk() )
    triples = product( range(2), range(2), range(2) )
    for y,d,i in triples:
      if (y,d,i,0) in unit.params['features_as_python_dict']:
        unit.ripl.predict('(observe_birds %i %i %i)'%(y,d,i) )
    return unit
    
  # Create unit and initialize with infer(1)
  original_unit = make_unit_with_predict()
  original_unit.ripl.infer(1)
  original_filename = original_unit.save('temp_test')
  copy_unit = make_unit_with_predict().make_saved_model(original_filename)
  
  assert equality_unit( original_unit, original_unit, verbose), 'orginal != original'
  assert equality_unit( original_unit, copy_unit, verbose), 'loaded copy != original'

  # More inference on original unit. 
  original_unit.ripl.infer(10)
  filename_more_infer = original_unit.save('temp_test_more_infer')
  copy_unit_more_infer = make_unit_with_predict().make_saved_model(filename_more_infer)

  assert equality_unit( original_unit, copy_unit_more_infer, verbose), 'updated original!=loaded copy of updated original'
  ## inference will often (not always) produce diverging copies
  ## (we don't assert this, but calling equality_unit allows us
  ## to find different directives if they exist)
  equality_unit( copy_unit, copy_unit_more_infer, verbose)
  
    

def test_all_unit_params( backends=('puma','lite'), random_or_exhaustive='random', small_model = True):

  random_mode = True if random_or_exhaustive=='random' else False

  # tests that take a ripl_thunk
  tests_ripl_thunk = (_test_dataset_load_observations,
                      _test_make_features_dict)

  # tests that take unit object (with ripl) as input
  tests_one_ripl =  (_test_model,
                     _test_cell_to_prob_dist,
                     _test_memoization_observe,
                     _test_save_images, )
  
  # tests that take unit object and a separate ripl_thunk
  # e.g. for loading saved observes onto a new ripl
  # -- allows for mixing up backends (which we don't test currently)
  tests_two_ripls = (_test_load_observations,
                     _test_incremental_load_observations,
                     _test_make_infer)

  models = (Multinomial, Poisson)
  
  ripl_thunks = []
  for backend in backends:
    thunk = mk_p_ripl if backend=='puma' else mk_l_ripl
    ripl_thunks.append( thunk )

  if small_model:
    params_short_names = ('minimal_onestep_diag10',) 
  else:
    params_short_names = ('minimal_onestep_diag10', 'dataset1', 'test_medium_onestep_diag105')

  make_params_thunks = [ lambda:make_params( name ) for name in params_short_names ]

  rand_draw = lambda seq: seq[ np.random.randint(len(seq)) ]


  def get_test_unit_args(tests, models, params_short_names, ripl_thunks):
    all_unit_args = [e for e in product(models, params_short_names, ripl_thunks)]
    test_unit_args = []

    for test in tests:
      unit_args = [rand_draw(all_unit_args)] if random_mode else all_unit_args      
      for model, params, ripl in unit_args:
          test_unit_args.append((test, model, params, ripl))
          
    return test_unit_args
                    

  ## tests that take ripl_thunk
  for test in tests_ripl_thunk:
    local_ripl_thunks = [rand_draw(ripl_thunks)] if random_mode else ripl_thunks
    for ripl_thunk in local_ripl_thunks:
      yield test, ripl_thunk

  ## run *tests_one_ripl*
  test_unit_args = get_test_unit_args(tests_one_ripl, models, params_short_names, ripl_thunks)
  for test,model,params,ripl in test_unit_args:
    yield test, make_unit_instance(model, ripl, params)

  ## run *tests_two_ripls*
  test_unit_args = get_test_unit_args(tests_two_ripls, models, params_short_names, ripl_thunks)
  for test,model,params,ripl in test_unit_args:
    yield test, make_unit_instance(model, ripl, params), ripl
  

  ## FIXME: TOO SLOW, NOT RUNNING YET
  # run Poisson-only (big num_birds) datasets  
  # many_birds_short_names = ('poisson_onestep_diag105', 'dataset2')
  # tests = (_test_incremental_load_observations,)
  # models = (Poisson,)
  # test_unit_args = get_test_unit_args(tests, models, many_birds_short_names, ripl_thunks)
  # for test,model,params,ripl in test_unit_args:
  #   yield test, make_unit_instance(model, ripl, params), ripl
  


  ## special case test that takes ripl_thunk and make_params_thunk
  args = [el for el in product( models, ripl_thunks, make_params_thunks)]
  if random_mode:
    args = [ rand_draw(args) ]
  for model, ripl_thunk, make_params_thunk in args:
      yield _test_save_load_model, model, ripl_thunk, make_params_thunk 


  
  

def _test_load_features_multinomial( ):
  # verify that make_features generates equivalent strings and dicts
  # TODO currently only true for Puma. Need to add this for Lite. 
  # make_params doesn't have the option to do this, and this will have to be added

  pass
 #  units = []
#   for dict_string in ('string','dict'):
    
#     params = make_params(  params_short_name = 'bigger_onestep_diag105' )
    
                          
#     units.append( Multinomial( mk_p_ripl(), params ) )

  
#   unit = Multinomial( mk_p_ripl(), make_params(
#   cells_list = range( unit.cells )
#   test_keys = product( unit.years, unit.days, cells_list, cells_list )
#   key_to_string = lambda k: '%i %i %i %i'%k
  
# #  for u in units:
# #   print u.features

#   for k in test_keys:

#     test_string = '(lookup features (array %s))' % key_to_string( k )
#     values = [u.ripl.sample(test_string) for u in units]
#     eq_( *values )





def display_timeit_run( test_func, test_lambda ):
  bar = '----------\n'
  print '%s TEST: %s \n DOC: %s %s'% (test_func.__name__,
                                      test_func.__doc__,
                                      bar, bar)

  return timeit( test_lambda, number=1 )



def run_nose_generative( test ):

  test_times = {}

  for yield_args in test():
    test_func, args = yield_args[0], yield_args[1:]
    test_lambda =  lambda: test_func( *args )

    test_time = display_timeit_run( test_func, test_lambda )

    test_times[ test_func.__name__ + '__%s'%str(args) ] = test_time

  return test_times


      
def run_all(kwargs = None):
  
  regular_tests =  ( test_features_functions,
                     test_ind_to_ij,
                     test_make_grid, )
  
  test_times = {}
  
  for t in regular_tests:
    
    test_time = display_timeit_run( t, t )
    
    test_times[ t.__name__ ] = test_time
  
    generative_tests = ( lambda: test_all_unit_params( ), )

  for t in generative_tests:
    test_times.update( run_nose_generative( t ) )


  print '\n\n\n-----------------------\n passed all tests'

  return test_times









