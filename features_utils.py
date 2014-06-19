from itertools import product
from utils import toVenture, make_grid
import numpy as np
import matplotlib.pylab as plt

## Utils and functions on pairs of cells (for feature generation)

# Prob( move(i,j) ) = exp( hypers0*feat0(i,j) + hypers1*feat1(i,j) ) / exp( ... for all j's)

# NB: if hypers0=-5, then -ve feature value has positive effect
# due to exponential: any negative vals for hypers_i*feat_i will add nothing.

# if we make all feature vals +ve, then -ve hypers will be like 0 hypers.
# if not, then for a feature_i to not have impact, we need to have
# hyper_i = 0. 

def l2(cell1_ij,cell2_ij ):
    return ((cell1_ij[0] - cell2_ij[0])**2 + (cell1_ij[1] - cell2_ij[1])**2)**.5

def within_d(cell1_ij, cell2_ij, d=2):
  return 1 if l2(cell1_ij, cell2_ij) <= d else 0

def one_step(cell1_ij, cell2_ij):
  return 1 if l2(cell1_ij, cell2_ij) == 1 else 0
  
def uniform_feature(cell1_ij, cell2_ij): return 1

def distance(cell1_ij, cell2_ij):
  d=l2(cell1_ij, cell2_ij)**(.5)
  return d**-1 if d!=0 else 2  # 2 could be set differently

def avoid_cells(cell1_ij, cell2_ij, avoided_cells):
  'Avoided cells get 0 and so are neutral over all. Other cells are "goal" cells.'
  return 0 if list(cell2_ij) in map(list,avoided_cells) else 1

def goal_direction(cell1_ij, cell2_ij, goal_direction=np.pi/4):
  dx = cell2_ij[1]-cell1_ij[1] ## TODO get this working proper
  dy = cell2_ij[0]-cell1_ij[0] 
  if dx==0: angle = 0
  else:
      angle=np.arctan( float(dy) / dx )
  return angle - goal_direction 
    # find angle between cells

# also try a det function for reproducibility

#date_wind = mem(lambda y,d: np.random.vonmises(0,4))
# def wind(cell1_ij,cell2_ij,year=0,day=0):
#     return goal_direction( cell1_ij, cell2_ij, date_wind(year,day))



## Other Utils for generating features
def ind_to_ij(height,width,index,order='F'):
  'Convert flat index to (i,j), depending on order'
  grid = make_grid(height,width=width,order=order)
  return map(int,np.where(grid==index))



# Generate Python and Venture dict of features from *functions*                        
def genFeatures(height,width,years,days,order='F',functions='easy'):
    
  cells = height * width
  latents = product(years,days,range(cells),range(cells))
  
  diagonal = [(i,i) for i in range(min(height,width))]
  color_diag = lambda c1,c2: avoid_cells(c1,c2,diagonal)

  # diagonal2 = [(1,0),(2,0) ]
  # color_diag2 = lambda c1,c2: avoid_cells(c1,c2,diagonal2)
  

  if functions=='easy':
      feature_functions =  (one_step,color_diag)#,lambda c1,c2: within_d( c1,c2, d=.33), within_d, color_diag, uniform_feature )
  else:
      feature_functions = (distance,color_diag)

  feature_dict = {}
  for (y,d,cell1,cell2) in latents:
    feature_dict[(y,d,cell1,cell2)] = []
    cell1_ij,cell2_ij = map(lambda index:ind_to_ij(height,width,index, order),
                            (cell1,cell2))
    
    # feature func should take years/days also for wind
    for f in feature_functions:
    
      feature_dict[(y,d,cell1,cell2)].append( f(cell1_ij, cell2_ij) )

  return toVenture(feature_dict),feature_dict




def cell_to_feature(height, width, state, python_features_dict, feature_ind):
  'Given state(y,d,i) and feature index, return feature from features_dict'
  cells = height * width
  y,d,i = state
  l=[ python_features_dict[(y,d,i,j)][feature_ind] for j in range(cells)]
  return make_grid(height, width, top0=True, lst=l)
  


def from_cell_dist(height,width,ripl,cell_i,year,day,order='F'):
  'Given ripl, (year,day,cell_i), get simplex (unnormed) and grid with normed dist'
  simplex =ripl.sample('(get_bird_move_dist %i %i %i)'%(year,day,cell_i))
  p_dist = simplex / np.sum(simplex)
  grid = make_grid(height,width,lst=p_dist,order=order)
  return simplex,grid


def plot_from_cell_dist(params,ripl,cells,year=0,day=0,order='F',horizontal=True):

  height,width =params['height'],params['width']

  fig,ax = plt.subplots(len(cells),1,figsize=(5,2.5*len(cells)))
  
  for count,cell in enumerate(cells):
    simplex, grid_from_cell_dist = from_cell_dist( height,width,ripl,cell,year,day,order=order)
    im= ax[count].imshow(grid_from_cell_dist, cmap='copper',interpolation='none') 
    ax[count].set_title('P(i,j),i=%i'%cell)
    #cbar = plt.colorbar(im)
  fig.tight_layout()  
  fig.subplots_adjust(right=0.8)
  cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
  fig.colorbar(im, cax=cbar_ax)
  
## ALT that might work better
# for ax in axes.flat:
#     im = ax.imshow(np.random.random((10,10)), vmin=0, vmax=1)

# cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
# plt.colorbar(im, cax=cax, **kw)
  



