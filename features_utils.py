from itertools import product
from utils import toVenture, make_grid
from venture.venturemagics.ip_parallel import *
import numpy as np
import matplotlib.pylab as plt


def l2(cell1_ij,cell2_ij ):
    return ((cell1_ij[0] - cell2_ij[0])**2 + (cell1_ij[1] - cell2_ij[1])**2)**.5

def within_d(cell1_ij, cell2_ij, d=2):
  return 1 if l2(cell1_ij, cell2_ij) <= d else 0

def one_step(cell1_ij, cell2_ij):
  return 1 if l2(cell1_ij, cell2_ij) == 1 else 0
  
def uniform_feature(cell1_ij, cell2_ij): return 1

def distance(cell1_ij, cell2_ij):
  d=l2(cell1_ij, cell2_ij)**(.5)
  return d**-1 if d!=0 else 2

def avoid_cells(cell1_ij, cell2_ij, avoided_cells):
  'Avoided cells get -1. Other cells are "goal" cells.'
  return 0 if list(cell2_ij) in map(list,avoided_cells) else 1

def goal_direction(cell1_ij, cell2_ij, goal_direction=np.pi/4):
  dx = cell2_ij[1]-cell1_ij[1] # flip this round
  dy = cell2_ij[0]-cell1_ij[0] # flip this round
  if dx==0: angle = 0
  else:
      angle=np.arctan( float(dy) / dx )
  return angle - goal_direction 
    # find angle between cells

# also try a det function for reproducibility
#date_wind = mem(lambda y,d: np.random.vonmises(0,4))
def wind(cell1_ij,cell2_ij,year=0,day=0):
    return goal_direction( cell1_ij, cell2_ij, date_wind(year,day))

                        
def genFeatures(height,width,years,days,order='F',functions='easy'):
  cells = height * width
  latents = product(years,days,range(cells),range(cells))
  
  diagonal = [(i,i) for i in range(min(height,width))]
  color_diag = lambda c1,c2: avoid_cells(c1,c2,diagonal)

  diagonal2 = [(1,0),(2,0) ]
  color_diag2 = lambda c1,c2: avoid_cells(c1,c2,diagonal2)
  
  #feature_functions = (goal_direction,within_d,color_diag)
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



def ind_to_ij(height,width,index,order='F'):
  grid = make_grid(height,width=width,order=order)
  return map(int,np.where(grid==index))


def cell_to_feature(height, width, state, features, feature_ind):
  cells = height * width
  y,d,i = state
  l=[ features[(y,d,i,j)][feature_ind] for j in range(cells)]
  return make_grid(height, width, top0=True, lst=l)
  

def from_cell_dist(height,width,ripl,i,year,day,order='F'):
  
  simplex =ripl.sample('(get_bird_move_dist %i %i %i)'%(year,day,i))
  p_dist = simplex / np.sum(simplex)
  grid = make_grid(height,width,lst=p_dist,order=order)
  return simplex,grid


def plot_from_cell_dist(params,ripl,cells,year=0,day=0,order='F',horizontal=True):

  height,width =params['height'],params['width']

  fig,ax = plt.subplots(len(cells),1,figsize=(5,2.5*len(cells)))
  ims = []
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
  



