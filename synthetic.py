from itertools import product
from utils import *
from model import OneBird,Poisson
from venture.venturemagics.ip_parallel import *
import matplotlib.pylab as plt
import sys,time

def l2(cell1_ij,cell2_ij ):
    return ((cell1_ij[0] - cell2_ij[0])**2 + (cell1_ij[1] - cell2_ij[1])**2)**.5

def within_d(cell1_ij, cell2_ij, d=2):
  return 1 if l2(cell1_ij, cell2_ij) <= d else -1

def distance(cell1_ij, cell2_ij):
  d=l2(cell1_ij, cell2_ij)**(.5)
  return d**-1 if d!=0 else 2

def avoid_cells(cell1_ij, cell2_ij, avoided_cells):
  'Avoided cells get -1. Other cells are "goal" cells.'
  return -1 if list(cell2_ij) in map(list,avoided_cells) else 1

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

                        
def genFeatures(height,width,years,days,order='F'):#feature_functions=None):
  cells = height * width
  latents = product(years,days,range(cells),range(cells))
  
  diagonal = [(i,i) for i in range(min(height,width))]
  color_diag = lambda c1,c2: avoid_cells(c1,c2,diagonal)
  
  #feature_functions = (goal_direction,within_d,color_diag)
  feature_functions = (goal_direction,lambda c1,c2:within_d(c1,c2,d=1),color_diag)
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

params_keys = ['height','width','years','days',
              'features','num_features','num_birds',
              'learn_hypers','hypers',
              'softmax_beta','load_observes_file']

def test_unit_analytics_loading(params):
  r = mk_p_ripl()
  uni = OneBird(r,params)
  uni.loadAssumes()
  ana = uni.getAnalytics()
  ana.updateQueryExps(['(move2 0 0 0 0)'])
  h,rfc = ana.runFromConditional(100,runs=1)
  assert r is uni.ripl
  assert not(r is rfc)
  assert h.averageValue('(move2 0 0 0 0)') > 0 
  assert len(ana.parameters) == len(uni.parameters)

  for ripl in (r,rfc):
    assert ripl.sample('hypers0')==params['hypers'][0]
    assert ripl.sample('(size cell_array)')==params['width']*params['height']
    assert ana.parameters


def onebird_synthetic_infer(gtruth_params,infer_params,infer_prog,steps_iterations,
                            save=False,plot=True):
    
  years,days = gtruth_params['years'],gtruth_params['days']
  
  def locs_fig(unit,name):      
      locs = unit.getBirdLocations(years,days)
      fig = unit.draw_bird_locations(years,days,name=name,plot=plot,save=save)
      return locs,fig
    
  uni = OneBird(mk_p_ripl(),gtruth_params)
  uni.loadAssumes()
  gtruth_locs,gtruth_fig = locs_fig(uni,gtruth_params['name'])
  filename = uni.store_observes(years,days)

  # make inference model
  uni_inf = OneBird(mk_p_ripl(),infer_params)
  uni_inf.loadAssumes()
  prior_locs,prior_fig = locs_fig(uni_inf,infer_params['name']+'_prior')

  # observe and infer (currently we pass in the whole unit object)
  start = time.time()
  infer_prog(uni_inf, steps_iterations, filename)
  print 'Obs and Inf: %s, elapsed: %s'%(infer_prog,time.time() - start)

  # posterior info
  posterior_locs,posterior_fig = locs_fig(uni_inf,infer_params['name']+'_post')

  # make a fresh ripl to measure impact of inference
  uni_fresh = OneBird(mk_p_ripl(),infer_params)
  uni_fresh.loadAssumes()
  unit_objects = uni,uni_fresh,uni_inf
  
  all_locs = gtruth_locs, prior_locs, posterior_locs
  figs = gtruth_fig, prior_fig, posterior_fig

  return unit_objects,all_locs, figs

  

def mse(locs1,locs2,years,days):
  all_days = product(years,days)
  all_error = [ (locs1[y][d]-locs2[y][d])**2 for (y,d) in all_days]
  return np.mean(all_error)

def get_hypers(ripl,num_features):
  return np.array([ripl.sample('hypers%i'%i) for i in range(num_features)])

def compare_hypers(gtruth_unit,inferred_unit):
  def mse(hypers1,hypers2):
    return np.mean((hypers1-hypers2)**2)
    
  get_hypers_par = lambda r: get_hypers(r, gtruth_unit.num_features)
  
  return mse( *map(get_hypers_par, (gtruth_unit.ripl, inferred_unit.ripl) ) )




def filter_inf(unit,steps_iterations,filename=None,record_prog=None):
  steps,iterations = steps_iterations  
  args = unit.name, steps, iterations
  print 'filter_inf. Name: %s, Steps:%i, iterations:%i'%args
                                                         
  def basic_inf(ripl,year,day):
    for iteration in range(iterations):
      latents = '(mh move2 %i %i)'%( day, steps)
      ripl.infer('(mh hypers one 10)')
      ripl.infer(latents)
      print 'iter: %i, inf_str:%s'%(iteration,latents)
  
  def record(unit):  return get_hypers(unit.ripl, unit.num_features)

  records = {}
  for y in unit.years:
    for d in unit.days:
      unit.observe_from_file([y],[d],filename)

      if d>0:
        basic_inf(unit.ripl, y, d-1)
      
      if record_prog:
        records[(y,d)] = record_prog(unit)
      else:
        records[(y,d)] = record(unit)
  return unit


def smooth_inf(unit,steps_iterations,filename=None):
  steps,iterations = steps_iterations  
  args = unit.name, steps, iterations
  print 'smooth_inf. Name: %s, Steps:%i, iterations:%i'%args

  # observe all data
  unit.observe_from_file(unit.years, unit.days, filename=filename)
    
  for iteration in range(iterations):
    unit.ripl.infer('(mh hypers one 10)')
    unit.ripl.infer('(mh move2 one %i)'%steps)      
  return unit


    
def test_recon(steps_iterations,test_hypers=False,plot=True,use_mh_filter=False):
  # model params
  Y, D = 1, 4
  years,days = range(Y),range(D)
  height,width = 4,4
  features,features_dict = genFeatures(height, width, years, days, order='F')
  num_features = len( features_dict[(0,0,0,0)] )
  learn_hypers, hypers = False,(1,1)
  num_birds = 15
  softmax_beta = 2
  load_observes_file=False

  params = dict(years=years, days = days, height=height, width=width,
                features=features, num_features = num_features,
                learn_hypers=learn_hypers, hypers = hypers,
                num_birds = num_birds, softmax_beta = softmax_beta,
                load_observes_file=load_observes_file)
  assert set(params.keys()).issubset(params_keys)

  # copy and specialize params for gtruth and inference
  gtruth_params  = params.copy()
  infer_params = params.copy()
  gtruth_params['name'] = 'gtruth'
  infer_params['name'] = 'infer'

  # define inference program
  if test_hypers:
    infer_params['learn_hypers'] = True
    
  infer_prog = filter_inf if use_mh_filter else smooth_inf
  
  # do inference using OneBird class                                                  
  unit_objects, all_locs, all_figs = onebird_synthetic_infer(gtruth_params, infer_params, infer_prog,
                                                             steps_iterations, plot=plot)

                                                  
  # unpack results                                                  
  gtruth_unit, fresh_unit, inf_unit = unit_objects
  gt_locs,prior_locs,post_locs = all_locs
  gt,pri,post = all_figs

  mse_gt = lambda l: mse(gt_locs,l,gtruth_params['years'],gtruth_params['days'])
  print 'prior,post mses: %.2f %.2f'%(mse_gt(prior_locs),mse_gt(post_locs))

  if test_hypers:
    mse_hypers_gt = lambda unit_obj: compare_hypers(gtruth_unit,unit_obj)
    print 'prior,post hypers mses: %.2f %.2f'%(mse_hypers_gt(fresh_unit),mse_hypers_gt(inf_unit))

    
  return unit_objects,params, all_locs, all_figs



# Y, D = 1, 8
# years,days = range(Y),range(D)
# height,width = 5,5
# features,features_dict = genFeatures(height, width,years,days, order='F')
# num_features = len( features_dict[(0,0,0,0)] )
# learn_hypers, hypers = False,(1,1)
# num_birds = 14
# softmax_beta = 3
# load_observes_file=False

# params = dict(years=years, days = days, height=height, width=width,
#                 features=features, num_features = num_features,
#                 learn_hypers=learn_hypers, hypers = hypers,
#                 num_birds = num_birds, softmax_beta = softmax_beta,
#                 load_observes_file=load_observes_file)


# r=mk_p_ripl()
# uni = OneBird(r,params)
# cells = height * width
# # compare from-i and from-cell-dist
# if int(sys.argv[1])==1:
#   cells=(0,5,15)
#   fig,ax = plt.subplots(len(cells),2)


def plot_from_cell_dist(params,ripl,cells,year=0,day=0,order='F'):

  height,width =params['height'],params['width']
  fig,ax = plt.subplots(len(cells),1,figsize=(5,2.5*len(cells)))

  for count,cell in enumerate(cells):
    simplex, grid_from_cell_dist = from_cell_dist( height,width,ripl,cell,year,day,order=order)
    ax[count].imshow(grid_from_cell_dist, cmap='copper',interpolation='none')
    ax[count].set_title('f_cell_dist: %i'%cell)
  fig.tight_layout()

# def plot_all_features_from_cell(params,ripl,cells,year=0,day=0,order='F'):
#   height,width,num_features =params['height'],params['width'],params['num_features']
#   fig,ax = plt.subplots(num_features,1,figsize=(5,2.5*num_features)
#   for feature_ind in range(num_features):
  
#     state = (0,0,cell)
#     year,day,_ = state
#     grid_from_i = { hyper: cell_to_feature(height,width, state, features_dict,hyper) 

