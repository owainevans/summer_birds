from itertools import product
from utils import *

import numpy as np
import matplotlib.pylab as plt

## Utils and functions on pairs of cells (for feature generation)

# Prob( move(i,j) ) = exp( hypers0*feat0(i,j) + hypers1*feat1(i,j) ) / exp( ... for all j's)

# NB: if hypers0=-5, then -ve feature value has positive effect
# due to exponential: any negative vals for hypers_i*feat_i will add nothing.

# if we make all feature vals +ve, then -ve hypers will be like 0 hypers.
# if not, then for a feature_i to not have impact, we need to have
# hyper_i = 0. 


### Functions for generating features
def l2(cell1_ij,cell2_ij ):
    return ((cell1_ij[0] - cell2_ij[0])**2 + (cell1_ij[1] - cell2_ij[1])**2)**.5

def within_d(cell1_ij, cell2_ij, d=2):
  return 1 if l2(cell1_ij, cell2_ij) <= d else 0

def one_step(cell1_ij, cell2_ij):
  return 1 if l2(cell1_ij, cell2_ij) == 1 else 0

def distance(cell1_ij, cell2_ij):
  d = l2(cell1_ij, cell2_ij)
  return 1. if d==0 else d**(-1)  # 1 could be set differently

def avoid_cells(cell1_ij, cell2_ij, avoid_cell):
  'Avoided cells get 0 and so are neutral over all. Other cells are "goal" cells.'
  assert isinstance( cell2_ij, (tuple,list) )

  return 0 if avoid_cell( cell2_ij ) else 1

diagonal = lambda ij: ij[0]==ij[1]
color_diag = lambda c1,c2: avoid_cells(c1,c2,diagonal)


def goal_direction(cell1_ij, cell2_ij, goal_direction=np.pi/4):
  dx = cell2_ij[1]-cell1_ij[1] ## TODO get this working 
  dy = cell2_ij[0]-cell1_ij[0] 
  if dx==0: angle = 0
  else:
      angle=np.arctan( float(dy) / dx )
  return angle - goal_direction     # find angle between cells

  
### Store named sets of feature_functions
name_to_feature_functions = {'uniform':( lambda cell1_ij, cell2_ij: 0, ),
                             'one_step_and_not_diagonal': (one_step,color_diag),
                             'distance': (distance,),
                             'not_diagonal': (color_diag,) }
assert all( [isinstance(v,tuple) for k,v in name_to_feature_functions.items()] )


### Generate Python and Venture dict of features from *functions*                        
def ind_to_ij(height, width, index, order='F'):
  'Convert flat index to (i,j), depending on order'
  grid = make_grid(height,width=width,order=order)
  return map(int,np.where(grid==index))

  
def make_features_dict(height, width, years, days, feature_functions_name='distance', dict_string='string'):
    
  cells = height * width
  latents = product(years,days,range(cells),range(cells))

  feature_functions = name_to_feature_functions[feature_functions_name]
  feature_dict = {}

  for (y,d,cell1,cell2) in latents:
    feature_dict[(y,d,cell1,cell2)] = []
    # assume all feature functions work on F order arrays
    cell1_ij,cell2_ij = map(lambda index:ind_to_ij(height, width, index, order='F'),
                            (cell1,cell2))
    
    for f in feature_functions:
      feature_value = f(cell1_ij, cell2_ij)
      assert isinstance(feature_value,(int,float))
      feature_dict[(y,d,cell1,cell2)].append( feature_value )

  # turn features dict into a string or directory into Venture value
  if dict_string=='string':
      venture_exp = python_features_to_venture_exp( feature_dict)
  else:
      venture_exp = toVenture( feature_dict)
  return venture_exp, feature_dict





def cell_to_feature(height, width, state, python_features_dict, feature_ind):
  'Given state(y,d,i) and feature index, return feature from features_dict'
  cells = height * width
  y,d,i = state
  l=[ python_features_dict[(y,d,i,j)][feature_ind] for j in range(cells)]
  return make_grid(height, width, top0=True, lst=l)
  


## function for loading saved features from a file (currently saved as CSV)
#current we delete stuff but not clear why we should do this. two ways to cut 
#down the list, one is from readFeatures, another from load_features. might
#be easiest to have deletions here, or at least make it optional

def load_features(features_file, years_list, days_list, max_year=None, max_day=None):
  'Load features from Birds datasets and convert to Venture dict'

  print "Loading features from %s" % features_file  

  ## FIXME needs be careful about 0 vs 1 indexing for the year
  ## NOTE: looks 
  features = readFeatures(features_file, max_year, max_day)
  
  for (y, d, i, j) in features.keys():
    if y not in years_list or d not in days_list:
      del features[(y, d, i, j)]
  
  return toVenture(features), features









def cell_to_prob_dist(height, width, ripl, source_cell, year, day,order='F'):
  '''Given *cell_i* get normalized grid for probability
     of moving to each cell from *cell_i*'''
  simplex = ripl.sample('(get_bird_move_dist %i %i %i)'%(year,day,source_cell))

  phi= lambda j: ripl.sample('(phi %s %s %s %s)' % (year,day,source_cell,j) )
  simplex_from_phi = [ phi(j) for j in range(height*width) ]

  assert (np.array(simplex) == np.array(simplex_from_phi)).all()
                                       
  p_dist = simplex / np.sum(simplex)
  grid = make_grid( height, width, lst=p_dist, order=order )
  return grid



def plot_cell_to_prob_dist(height, width, ripl, source_cells, year=0, day=0, order='F', name=''):

  assert isinstance( source_cells, (list,tuple) )
  assert isinstance( source_cells[0], int)

  
  fig,ax = plt.subplots(len(source_cells), 1, figsize=(5,2.5*len(source_cells)))
  
  for count, cell in enumerate(source_cells):

    grid_cell_to_prob_dist = cell_to_prob_dist(height, width, ripl, cell,year,day,order)

    im= ax[count].imshow(grid_cell_to_prob_dist, cmap='hot', vmin=0, vmax=1,
                         interpolation='none', extent=[0,width,height,0]) 
    ax[count].set_title('%s  P(i,j) for Cell i=%i, day:%i'%(name,cell,day))
    ax[count].set_xticks(range(width+1))
    ax[count].set_yticks(range(height+1))

    ij_cell = ind_to_ij( height, width, cell, order=order)
    ij_cell = ij_cell[1]+.1, ij_cell[0]+.5 # switch order for annotation
    ax[count].annotate('Cell i', xy = ij_cell, xytext = ij_cell, color='c')

  fig.tight_layout()  
  fig.subplots_adjust(right=0.8)
  cbar_ax = fig.add_axes([0.75, 0.7, 0.05, 0.2])
  fig.colorbar(im, cax=cbar_ax)
  



