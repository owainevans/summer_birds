from itertools import product
from utils import *
from model import OneBird,Poisson
from venture.venturemagics.ip_parallel import *
import matplotlib.pylab as plt

def l2(cell1_ij,cell2_ij ):
    return ((cell1_ij[0] - cell2_ij[0])**2 + (cell1_ij[1] - cell2_ij[1])**2)**.5

def within_d(cell1_ij, cell2_ij, d=2):
  return 1 if l2(cell1_ij, cell2_ij) <= d else -1

def avoid_cells(cell1_ij, cell2_ij, avoided_cells):
  'Avoided cells get -1. Other cells are "goal" cells.'
  return -1 if list(cell2_ij) in map(list,avoided_cells) else 1

def goal_direction(cell1_ij, cell2_ij, goal_direction=np.pi/4):
  dy = cell2_ij[1]-cell1_ij[1] # flip this round
  if dy==0:
      angle = 0
  else:
      dx = cell2_ij[0]-cell1_ij[0]
      angle=np.arctan( float(dx) / dy )
  return angle - goal_direction 
    # find angle between cells

# also try a det function for reproducibility
#date_wind = mem(lambda y,d: np.random.vonmises(0,4))
def wind(cell1_ij,cell2_ij,year=0,day=0):
    return goal_direction( cell1_ij, cell2_ij, date_wind(year,day))

    
def genFeatures(height,width,years,days,order='F'):#feature_functions=None):
  cells = height * width
  latents = product(years,days,range(cells),range(cells))
  
  diagonal = [(i,i) for i in range(min(height,width))]
  color_diag = lambda c1,c2: avoid_cells(c1,c2,diagonal)
  
  #feature_functions = (goal_direction,within_d,color_diag)
  feature_functions = (lambda c1,c2:within_d(c1,c2,d=1),color_diag)

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

def make_grid(height,width=None,top0=True,lst=None,order='F'):
  width = height if width is None else width
  l = np.array(range(width*height)) if lst is None else np.array(lst)
  grid = l.reshape( (height, width), order=order)
  if top0:
    return grid
  else:
    grid_mat = np.zeros( shape=(height,width),dtype=int )
    for i in range(width):
      grid_mat[:,i] = grid[:,i][::-1]
    return grid_mat
      
def cell_to_feature(height, width, state, features, feature_ind):
  cells = height * width
  y,d,i = state
  l=[ features[(y,d,i,j)][feature_ind] for j in range(cells)]
  return make_grid(height, width, top0=True, lst=l)
  

def from_cell_dist(height,width,ripl,i,year,day):
  simplex =ripl.sample('(get_bird_move_dist %i %i %i)'%(year,day,i))
  p_dist = simplex / np.sum(simplex)
  grid = make_grid(height,width,lst=p_dist,order='F')
  return simplex,grid
    

Y, D = 1, 20
years,days = range(Y),range(D)
height,width = 5,5
features,features_dict = genFeatures(height,width,years=years,days=days,order='F')
num_features = len( features_dict[(0,0,0,0)] )
hypers = [1,1]


params = dict(name='w2',
              height = height,
              width = width,
              years = years,
              days = days,
              features = features,
              num_features = num_features,
              learnHypers=False,
              hypers = hypers,
              num_birds = 20,
              softmax_beta = 5,
              load_observes_file=False)

r = mk_p_ripl()
uni = OneBird(r,params)
uni.loadAssumes()
ana = uni.getAnalytics()
h,rfc = ana.runFromConditional(5,runs=1)

assert not(r is rfc)
assert r.sample('hypers0')==hypers[0] and rfc.sample('hypers0')==hypers[0]







# compare from-i and from-cell-dist
plt.close('all')
cells=(0,5,15)
fig,ax = plt.subplots(len(cells),2)


for count,cell in enumerate(cells):
  state = (0,0,cell)
  year,day,_ = state
  grid_from_i = { hyper: cell_to_feature(height,width, state, features_dict,hyper) for hyper in range(num_features) }
  simple, grid_from_cell_dist = from_cell_dist( height,width,r,cell,year,day )
  ax[count,0].imshow(grid_from_i[0], cmap='copper',interpolation='none')
  ax[count,0].set_title('From_i: %i, feat0'%cell)
  ax[count,1].imshow(grid_from_cell_dist, cmap='copper',interpolation='none')
  ax[count,1].set_title('f_cell_dist: %i'%cell)

fig.tight_layout()
plt.show()

# ripl tests
