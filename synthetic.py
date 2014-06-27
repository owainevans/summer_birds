from itertools import product
from features_utils import genFeatures,from_cell_dist,plot_from_cell_dist
from model import OneBird,Poisson
from venture.venturemagics.ip_parallel import mk_p_ripl,MRipl, display_directives
from venture.unit import Analytics
import matplotlib.pylab as plt
import numpy as np
import sys,time

params_keys = ['height','width','years','days',
              'features','num_features','num_birds',
              'learn_hypers','hypers',
              'softmax_beta','load_observes_file']

## TODO PLAN FOR BIRDS

# 1. I tried to find an example where hypers can be learned
# in small amount of time. Current 'easy' functions params
# have one_step and color_diag, with hypers=[1,0]. if a feature
# can be positive and negative, then it should be possible
# to learn a zero hyper. if the features values are always
# positive, then it's trickier, because a zero weight
# will look similar to a negative weight.

# Once we find an example that can be easily learned, we 
# can vary inf programs and compare performance. We
# can get an integration test of Analytics filtering inference.

# 2. Need to add store_observes and observes_from_file methods
# to poisson and then generalization testing/inference functions
# below to work with Poisson object. Test Poisson param inference
# on our basic example. 





#### Utils for testing performance of inference
def mse(locs1,locs2,years,days):
  'MSE for output of unit.getBirdLocations'
  all_days = product(years,days)
  all_error = [ (locs1[y][d]-locs2[y][d])**2 for (y,d) in all_days]
  return np.mean(all_error)

def get_hypers(ripl,num_features):
  return np.array([ripl.sample('hypers%i'%i) for i in range(num_features)])


def compare_hypers(gtruth_unit,inferred_unit):
  'Compare hypers across two different Birds unit objects'
  def mse(hypers1,hypers2): return np.mean((hypers1-hypers2)**2)
    
  get_hypers_par = lambda r: get_hypers(r, gtruth_unit.num_features)
  
  return mse( *map(get_hypers_par, (gtruth_unit.ripl, inferred_unit.ripl) ) )



### Basic procedure for simulating from prior, saving, doing inference.
def onebird_synthetic_infer(gtruth_params,infer_params,infer_prog,steps_iterations,
                            save=False,plot=True):
  '''Generate OneBird data from prior, save to file, do inference on data.
     Needs full set of OneBird params: one set for generating, another
     for inference.
     *save/plot* is whether to save/plot bird locations images.'''

  # years and days are common to gtruth and infer unit objects
  years,days = gtruth_params['years'],gtruth_params['days']
  
  def locs_fig(unit,name):
    'Call getBirdLocations and draw_bird locations using global *years,days*.'  
    locs = unit.getBirdLocations(years,days)
    fig = unit.draw_bird_locations(years,days,name=name,plot=plot,save=save)
    return locs,fig
    
  # Create gtruth_unit object with Puma ripl  
  uni = OneBird(mk_p_ripl(),gtruth_params)
  uni.loadAssumes()
  gtruth_locs,gtruth_fig = locs_fig(uni, gtruth_params['name'])
  filename = uni.store_observes(years,days)  # filename will use uni.name and random string
  
  # make inference model
  uni_inf = OneBird(mk_p_ripl(),infer_params)
  uni_inf.loadAssumes()
  prior_locs,prior_fig = locs_fig(uni_inf,infer_params['name']+'_prior')

  # observe and infer (mutating the ripl in the Unit object)
  start = time.time()
  infer_prog(uni_inf, steps_iterations, filename)
  print 'Obs and Inf: %s, elapsed: %s'%(infer_prog,time.time() - start)

  # posterior info (after having mutated uni_inf.ripl)
  posterior_locs,posterior_fig = locs_fig(uni_inf,infer_params['name']+'_post')

  # make a fresh ripl to measure impact of inference
  uni_fresh = OneBird(mk_p_ripl(),infer_params)
  uni_fresh.loadAssumes()
  unit_objects = uni,uni_fresh,uni_inf
  
  all_locs = gtruth_locs, prior_locs, posterior_locs
  figs = gtruth_fig, prior_fig, posterior_fig

  return unit_objects,all_locs, figs

  

### Inference procedures for *infer_prog* arg in *onebird_synthetic_infer*

def filter_inf(unit, steps_iterations, filename=None, record_prog=None, verbose=False):
  """Loop over days, add all of a day's observes to birds unit.ripl. Then do multiple loops (iterations)
     of inference on move2(i,j) for the previous day and on the hypers. Optionally
     take a function that records different 'queryExps' in sense of Analytics."""

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
  '''Like *filter_inf* but observes all days at once and does inference on
   everything in moves2(i,j)'''
  steps,iterations = steps_iterations  
  args = unit.name, steps, iterations
  print 'smooth_inf. Name: %s, Steps:%i, iterations:%i'%args

  # observe all data
  unit.observe_from_file(unit.years, unit.days, filename=filename)
    
  for iteration in range(iterations):
    unit.ripl.infer('(mh hypers one 10)')
    unit.ripl.infer('(mh move2 one %i)'%steps)      
  return unit


# Inference procedure for analytics: could be integrated with *synthetic_bird_infer*.
# Note that the unit object is given incremental updates simultaneously with
# ana object and so gets  same observes at same time (though no inference is done on unit.ripl)
def ana_filter_inf(unit, ana, steps_iterations, filename=None, query_exps=None, verbose=False):
  '''Incremental inference on Analytics object. General pattern: loop over observes,
     add a set of them to Unit and Analytics objects [also add query expressions],
     then run Analytics inference.
     Here we specialize the observes and infers to Birds model.'''
  
  steps,iterations = steps_iterations  
  args = unit.name, steps, iterations
  print 'ana filter_inf. Name: %s, Steps:%i, iterations:%i'%args
                                                         

  def analytics_infer(ripl,year,day):
    '''Analog of *basic_inf* in *filter_inf*. Here inference is done
    via Analytics and we store history. Analytics only records after
    a full run of an inf_prog given as a string. Hence we run Analytics
    after every iteration (where we should do many iterations).'''
    
    hists = []
    
    for iteration in range(iterations):
      # inference program specifies 'block':(day-1)
      # as observe on *day* calls *move2* for (day-1)
      latents = '(mh move2 %i %i)'%( day-1, steps)
      hypers = '(mh hypers one 10)'
      inf_prog = '(cycle ( %s %s) 1)'%(latents,hypers)

      runs = 1 if not ana.mripl else ana.mripl.no_ripls # TODO sort out runs
      h,_ = ana.runFromConditional(1, runs=runs, infer = inf_prog )
      hists.append(h)
      if verbose: print 'iter: %i, inf_str:%s'%(iteration,latents)

    return hists


  # Loop over days, updating observes for *unit* and *ana* objects. 
  all_hists = []
  for y in unit.years:
    for d in unit.days:
      observes_yd = unit.observe_from_file([y],[d],filename)
      ana.updateObserves( observes_yd )
      if verbose: print 'ana.observes[-1]:', ana.observes[-1]
      # TODO support for QueryExps
      #[ana.updateQueryExps( [exp] ) for exp in moves_exp(y,d)]
      if d>0:  all_hists.extend( analytics_infer(ana,y,d)  )
    
  return ana, all_hists




####### TESTS FOR INFERENCE ON BIRDS UNIT/ANALYTICS


# Test for Persistent ripls in Analytics (no Birds stuff)
def test_persistent_ripl_analytics(mripl=False):
    v = MRipl(2,local_mode=True) if mripl else mk_p_ripl()
    v.assume('x','(normal 0 100)')
    data = [10]*5 + [25]*5 + [50]*5
    observes = [('(normal x 10)',val) for val in data]+ [('(normal x .1)',30)]

    ana = Analytics(v,mutateRipl=True )
    hists = []
    vv = ana.ripl if not mripl else ana.mripl

    for i,obs in enumerate(observes):
        ana.updateObserves( [ obs ] )
        h,_ = ana.runFromConditional( 40, runs=1)        
        print i,' x: ', vv.sample('x')
        hists.append( h )
    
    return vv,ana,hists


# Produce a params dict for testing inference
# (be wary of mutating entries without copying first)
def get_onebird_params(params_name='easy_hypers'):
  'Function for producing params for OneBird Unit object'
  if params_name == 'easy_hypers':
    name = 'easy_hypers'
    Y, D = 1, 6
    years,days = range(Y),range(D)
    height,width = 4,4
    functions = 'easy'
    features,features_dict = genFeatures(height, width, years, days,
                                         order='F',functions=functions)
    num_features = len( features_dict[(0,0,0,0)] )
    learn_hypers = False
    hypers = [1,0,0,0][:num_features]
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

 

# Once some params settings have been finalized, this should be a 
# test that hypers inference has worked correctly.
# General version would generate some hypers and then test
# the learning of them. But working with fixed params is ok also.

def test_easy_hypers_onebird():
  easy_params = get_params('easy')
  out = test_onebird_reconstruction( (10,4), test_hypers=True, plot=True, use_mh_filter = True)
  unit_objects, params, all_locs, all_figs, mses = out

  gtruth_unit =  unit_objects[0]
  assert isinstance(gtruth_unit,OneBird)
  assert not gtruth_unit.learn_hypers
  assert gtruth_unit.hypers == easy_params['hypers']
  
  latent_mse_prior, latent_mse_post = mses[0]
  assert UPPERBOUND  > latent_mse_prior > latent_mse_post
  assert LOWERBOUND < latent_mse_post < 0.6

  hypers_mse_prior, hypers_mse_post = mses[1]
  assert latent_mse_prior > latent_mse_post
  return None

    

  


## Test non-Analytics reconstruction AND hypers inference. MH-Filter is *filter_inf* vs. *smooth_inf*.
## Testing involves computing mse for latents and hypers.
## Gets params from *get_onebird_params*
def test_onebird_reconstruction(steps_iterations, test_hypers=False, plot=True,use_mh_filter=False):
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
  unit_objects,all_locs,all_figs = onebird_synthetic_infer(gtruth_params,infer_params,infer_prog,steps_iterations, plot=plot)

  # unpack results                                                  
  gtruth_unit, fresh_unit, inf_unit = unit_objects
  gt_locs,prior_locs,post_locs = all_locs
  gt,pri,post = all_figs

  # View the normalized dist on (0,0) for a few cells. Compare to
  # gtruth moves plots as check
  if plot:
    check_cells = tuple(range(5))
    plot_from_cell_dist(gtruth_params, gtruth_unit.ripl,
                        check_cells,year=0,day=0,order='F')


  # compute test statistics
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


###  Tests for Analytics OneBird Incremental Inference
# 1. Series of basic unit tests for OneBird in Analytics
# 2. Filter OneBirds hypers inference using *ana_filter_inf* above.
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

  # create gtruth uni object and run incremental inference
  gt_params = params.copy()
  gt_params['learn_hypers'] = False  ## TODO make params code clearer (safer)
  gt_unit = OneBird(mk_p_ripl(),gt_params)
  gt_unit.loadAssumes()
  filename = gt_unit.store_observes(gt_unit.years,gt_unit.days)

  inf_params = params.copy()  # which must have *learn_hypers*=True
  inf_unit = OneBird(mk_p_ripl(),inf_params)
  inf_unit.loadAssumes()
  inf_ana = inf_unit.getAnalytics(ripl_mripl,mutateRipl=True)

  # check prior for inference doesn't know hypers
  h,inf_ana_ripl = inf_ana.runFromConditional(1,runs=1)
  hypers0_prior= h.nameToSeries['hypers0'][0].values[-1]
  hypers0_gt = gt_unit.hypers[0]
  assert np.abs(hypers0_prior - hypers0_gt) > .2

  # analytics will only record every entire run on inf_prog
  # so we do lots of iteraions with small number of mh transitions
  steps_iterations = (2,4)
  _,hists = ana_filter_inf(inf_unit, inf_ana, steps_iterations, filename, verbose=True)
  
  return unit,ana, inf_unit, inf_ana, hists



def run_all_tests(plot=False):
  for boo in True,False:
    test_persistent_ripl_analytics(mripl=boo)
    test_ana_inf(mripl=boo)

  test_hypers = (True,False)
  use_mh_filter = (True,False)
  steps_iterations = ( (0,0), (1,1) )
  settings = product(steps_iterations, test_hypers, use_mh_filter )
  for steps_iterations, test_hypers, use_mh_filter in settings:
    print 'steps_iterations, test_hypers, use_mh_filter', steps_iterations, test_hypers, use_mh_filter
    test_onebird_reconstruction(  steps_iterations, test_hypers, plot, use_mh_filter)
