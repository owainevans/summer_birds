from itertools import product
from features_utils import genFeatures,from_cell_dist
from model import OneBird,Poisson
from venture.venturemagics.ip_parallel import mk_p_ripl,MRipl
from venture.unit import Analytics,display_directives
import matplotlib.pylab as plt
import numpy as np
import sys,time

params_keys = ['height','width','years','days',
              'features','num_features','num_birds',
              'learn_hypers','hypers',
              'softmax_beta','load_observes_file']

def mse(locs1,locs2,years,days):
  'MSE for bird_locations, output of onebird method'
  all_days = product(years,days)
  all_error = [ (locs1[y][d]-locs2[y][d])**2 for (y,d) in all_days]
  return np.mean(all_error)


def get_hypers(ripl,num_features):
  return np.array([ripl.sample('hypers%i'%i) for i in range(num_features)])

def compare_hypers(gtruth_unit,inferred_unit):
  def mse(hypers1,hypers2): return np.mean((hypers1-hypers2)**2)
    
  get_hypers_par = lambda r: get_hypers(r, gtruth_unit.num_features)
  
  return mse( *map(get_hypers_par, (gtruth_unit.ripl, inferred_unit.ripl) ) )




def onebird_synthetic_infer(gtruth_params,infer_params,infer_prog,steps_iterations,
                            save=False,plot=True,analytics_infer=False):
  '''Generate OneBird data from prior, save to file, do inference on data'''
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
  if analytics_infer:
    ana = uni_inf.getAnalytics(mripl=None)
    
  
  prior_locs,prior_fig = locs_fig(uni_inf,infer_params['name']+'_prior')

  # observe and infer (currently we pass in the whole unit object)
  start = time.time()
  if analytics_infer:
    ana_filter_inf(uni_inf,ana,steps_iterations,filename)
  else:
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

  if analytics_infer:
    unit_objects = list(unit_objects[:]) + ana 
  return unit_objects,all_locs, figs

  












def filter_inf(unit, steps_iterations, filename=None, record_prog=None, verbose=False):
  steps,iterations = steps_iterations  
  args = unit.name, steps, iterations
  print 'filter_inf. Name: %s, Steps:%i, iterations:%i'%args
                                                         
  def basic_inf(ripl,year,day):
    for iteration in range(iterations):
      latents = '(mh move2 %i %i)'%( day, steps)
      ripl.infer('(mh hypers one 10)')
      ripl.infer(latents)
      if verbose: print 'iter: %i, inf_str:%s'%(iteration,latents)
  
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

def basic_ana_test(mripl=False):
    v = MRipl(2,local_mode=True) if mripl else mk_p_ripl()
    v.assume('x','(normal 0 100)')
    observes = [('(normal x 10)',val) for val in [10,10,10,10,10,30,20,30,30,40,34,50,50,45,50,50] ]+[('(normal x .1)',30)]

    ana = Analytics(v,mutateRipl=True )
    hists = []
    vv = ana.ripl if not mripl else ana.mripl

    for i,obs in enumerate(observes):
        ana.updateObserves( [ obs ] )
        h,_ = ana.runFromConditional( 40, runs=1)        
        print i,' x: ', vv.sample('x')
        hists.append( h )
    
    return vv,ana,hists



def ana_filter_inf(unit, ana, steps_iterations, filename=None, query_exps=None, verbose=False):
  '''Incremental inference on Analytics object. General pattern: loop over observes,
     add a set of them to Unit and Analytics objects [also add query expressions],
     then run Analytics inference.
     Here we specialize the observes and infers to Birds model.'''
  
  steps,iterations = steps_iterations  
  args = unit.name, steps, iterations
  print 'ana filter_inf. Name: %s, Steps:%i, iterations:%i'%args
                                                         

  def analytics_infer(ripl,year,day):

    hists = []
    
    for iteration in range(iterations):
      # block:day-1, as observe on day calls *move2* for day-1
      latents = '(mh move2 %i %i)'%( day-1, steps)
      hypers = '(mh hypers one 10)'
      inf_prog = '(cycle ( %s %s) 1)'%(latents,hypers)

      runs = 1 if not ana.mripl else ana.mripl.no_ripls
      h,_ = ana.runFromConditional(1, runs=runs, infer = inf_prog )
      hists.append(h)
      if verbose: print 'iter: %i, inf_str:%s'%(iteration,latents)

    return hists


  all_hists = []
  for y in unit.years:
    for d in unit.days:
      observes_yd = unit.observe_from_file([y],[d],filename)
      ana.updateObserves( observes_yd )
      
      if verbose: print 'ana.observes[-1]:', ana.observes[-1]
      
      #[ana.updateQueryExps( [exp] ) for exp in moves_exp(y,d)]
      if d>0:  all_hists.extend( analytics_infer(ana,y,d)  )
    
  return ana, all_hists


def get_onebird_params(params_name='easy_hypers'):

  if params_name == 'easy_hypers':
    name = 'easy_hypers'
    Y, D = 1, 6
    years,days = range(Y),range(D)
    height,width = 4,4
    functions = 'easy'
    features,features_dict = genFeatures(height, width, years, days,
                                         order='F',functions=functions)
    num_features = len( features_dict[(0,0,0,0)] )
    learn_hypers, hypers = False, (5, -3, -3, -3)
    num_birds = 8
    softmax_beta = 1
    load_observes_file=False

    params = dict(name = name,
                  years=years, days = days, height=height, width=width,
                  features=features, num_features = num_features,
                  learn_hypers=learn_hypers, hypers = hypers,
                  num_birds = num_birds, softmax_beta = softmax_beta,
                  load_observes_file=load_observes_file)

  return params


def test_easy_hypers_onebird():
  easy_params = get_params('easy')
  out = test_recon( (10,4), test_hypers=True, plot=True, use_mh_filter = True)
  unit_objects, params, all_locs, all_figs, mses = out

  gtruth_unit =  unit_objects[0]
  assert isinstance(gtruth_unit,OneBird)
  assert not gtruth_unit.learn_hypers
  assert gtruth_unit.hypers == easy_params['hypers']
  
  latent_mse_prior, latent_mse_post = mses[0]
  assert k  > latent_mse_prior > latent_mse_post
  assert .1 < latent_mse_post < 0.6

  hypers_mse_prior, hypers_mse_post = mses[1]
  assert latent_mse_prior > latent_mse_post




    
def test_recon(steps_iterations,test_hypers=False,plot=True,use_mh_filter=False):
  params = get_onebird_params()
  assert set(params_keys).issubset( set(params.keys()) )

  # copy and specialize params for gtruth and inference
  gtruth_params  = params.copy()
  infer_params = params.copy()
  gtruth_params['name'] = 'gtruth'
  infer_params['name'] = 'infer'

  # define inference program
  if test_hypers: infer_params['learn_hypers'] = True
  
  infer_prog = filter_inf if use_mh_filter else smooth_inf
  
  # do inference using OneBird class                                                  
  unit_objects, all_locs, all_figs = onebird_synthetic_infer(gtruth_params, infer_params, infer_prog,
                                                             steps_iterations, plot=plot)

                                                  
  # unpack results                                                  
  gtruth_unit, fresh_unit, inf_unit = unit_objects
  gt_locs,prior_locs,post_locs = all_locs
  gt,pri,post = all_figs

  mse_gt = lambda l: mse(gt_locs,l,gtruth_params['years'],gtruth_params['days'])
  mses = [ (mse_gt(prior_locs),mse_gt(post_locs) ) ]
  print 'prior,post mses: %.2f %.2f'%mses[0]

  if test_hypers:
    mse_hypers_gt = lambda unit_obj: compare_hypers(gtruth_unit,unit_obj)
    mses_hypers = mse_hypers_gt(fresh_unit),mse_hypers_gt(inf_unit)
    mses.append(mses_hypers)
    
    print '\n---\n gt_hypers, fresh_prior_hypers, post_hypers:'
    all_hypers= [get_hypers(unit.ripl,params['num_features']) for unit in unit_objects]
    print all_hypers

    print '\nprior,post hypers mses: %.2f %.2f'%(mses[-1])
  else:
    all_hypers = None
    
  return unit_objects,params, all_locs, all_figs, mses, all_hypers



def test_ana_inf(mripl=False):
  'Series of tests for inference on analytics object for OneBird'
  params = get_onebird_params()
  params['learn_hypers'] = True
  
  unit = OneBird(mk_p_ripl(),params)
  unit.loadAssumes()

  ripl_mripl = MRipl(2,local_mode=True) if mripl else mk_p_ripl()
  ana = unit.getAnalytics(ripl_mripl,mutateRipl=True)
  
  ana_ripl = ana.ripl if not mripl else ana.mripl
  assert len(unit.assumes) == len(ana.assumes)
  hypers0_prior = ana_ripl.sample('hypers0')

  # test RFC with no observes
  runs = 1 if not mripl else 2
  h,_ = ana.runFromConditional(10,runs=runs)
  assert 'hypers0' in h.nameToSeries.keys()
  assert hypers0_prior != ana_ripl.sample('hypers0')

  # test RFC with one observe
  ana.updateObserves([ ('(normal hypers0 1)','1') ] )
  assert len(ana.observes) == 1
  assert ana_ripl.list_directives()[-1]['instruction'] == 'observe'
  h,_ = ana.runFromConditional(40,runs=runs)
  hypers0_1 =  ana_ripl.sample('hypers0')
  assert hypers0_prior != hypers0_1
  print 'should be close to 1:,', ana_ripl.sample('hypers0')

  # test RFC with second observe
  ana.updateObserves([ ('(normal hypers0 .5)','1') ] )
  assert len(ana.observes) == 2
  assert ana_ripl.list_directives()[-1]['instruction'] == 'observe'
  h,_ = ana.runFromConditional(40,runs=runs)
  assert hypers0_1 != ana_ripl.sample('hypers0')
  print 'should be closer to 1:,', ana_ripl.sample('hypers0')

  # create gtruth object and run incremental inference
  gt_params = params.copy()
  gt_params['learn_hypers'] = False
  gt_unit = OneBird(mk_p_ripl(),gt_params)
  gt_unit.loadAssumes()
  filename = gt_unit.store_observes(gt_unit.years,gt_unit.days)

  inf_params = params.copy()
  inf_unit = OneBird(mk_p_ripl(),inf_params)
  inf_unit.loadAssumes()
  inf_ana = inf_unit.getAnalytics(ripl_mripl,mutateRipl=True)

  # prior for inference doesn't know hypers
  h,inf_ana_ripl = inf_ana.runFromConditional(1,runs=1)
  hypers0_prior= h.nameToSeries['hypers0'][0].values[-1]
  hypers0_gt = gt_unit.hypers[0]
  assert np.abs(hypers0_prior - hypers0_gt) > .2

  # analytics will only record every entire run on inf_prog
  # so we do lots of iteraions with small number of mh transitions
  steps_iterations = (2,4)
  _,hists = ana_filter_inf(inf_unit, inf_ana, steps_iterations, filename, verbose=True)
  
  return unit,ana, inf_unit, inf_ana, hists




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

