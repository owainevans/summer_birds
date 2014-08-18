import numpy as np
from venture.shortcuts import make_puma_church_prime_ripl as mk_p_ripl
from nose.tools import eq_, assert_almost_equal

from utils import make_grid
from features_utils import ind_to_ij, make_features_dict, from_cell_dist
from synthetic import get_onebird_params
from model import OneBird, Poisson

def test_make_grid():
  cf = []
  cf.append(  np.array( ([0,3],[1,4],[2,5]) ) == make_grid(3,2,True,None,'F') )
  cf.append(  np.array( ([1,3],[0,2]) ) == make_grid(2,2,False,None,'F') )
  cf.append(  np.array( ([0,1],[2,3]) ) == make_grid(2,2,True,None,'C') )
  cf.append(  np.array( ([0,0,0],[0,0,0]) ) == make_grid(2,3,True,np.ones(4),'F') )  
  assert all( [ ar.all() for ar in cf ] )  
  
def test_ind_to_ij():
  height, width = 3,2
  grid = make_grid(height,width,True,range(height*width),'F')
  for ind in range(height*width):
    ij = tuple( ind_to_ij(height,width,ind,'F') )
    eq_( grid[ij], ind )

def test_make_features_dict():
  height, width = 3,2
  years,days = range(2), range(2)
  venture_dict, python_dict = make_features_dict(height,width,years,days,order='F',functions='easy')
  eq_( len(python_dict), (height*width)**2 * ( len(years)*len(days) ) )
  assert isinstance( python_dict[ (0,0,0,0) ], (list,tuple) )
  eq_( venture_dict['type'], 'dict' )
  assert isinstance(venture_dict['value'],dict)

def make_onebird_unit():
  params = get_onebird_params(params_name = 'easy_hypers' )
  unit =  OneBird(mk_p_ripl(),params)
  unit.loadAssumes()
  return unit

  
def test_from_cell_dist():
  unit = make_onebird_unit()
  height, width, ripl = unit.height, unit.width, unit.ripl
  for cell in [0,1,2]:
    _,grid,_,_ = from_cell_dist( height, width, ripl, cell, 0, 0, order='F' )
    assert_almost_equal( sum(grid), 1)
    









