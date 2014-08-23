from features_utils import make_features_dict, cell_to_prob_dist, plot_cell_to_prob_dist
from model import Multinomial,Poisson
from venture.venturemagics.ip_parallel import mk_p_ripl, MRipl
from venture.unit import Analytics
import matplotlib.pylab as plt
import numpy as np
import sys,time
from itertools import product



#### Utils for testing performance of inference
def mse(locs1,locs2,years,days):
  'MSE for output of unit.get_bird_locations'
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


def multinomial_synthetic_infer(*args,**kwargs):
  return synthetic_infer('multinomial',*args,**kwargs)


def synthetic_infer( model, gtruth_params, infer_params, infer_prog,
                                    steps_iterations, make_infer_string = None,
                                    save=False, plot=True, use_analytics=False, order='F'):
  '''Generate data from prior, save to file, do inference on data.
     Needs full set of model params: one set for generating, another
     for inference.
     *save/plot* is whether to save/plot bird locations images.'''

  makeUnit = Multinomial if model == 'multinomial' else Poisson
  
  # years and days are common to gtruth and infer unit objects
  years,days = gtruth_params['years'], gtruth_params['days']
  
  def locs_fig(unit,name=None):
    'Call get_bird_locations and draw_bird locations using global *years,days*.'  
    locs = unit.get_bird_locations(years,days)
    fig = unit.draw_bird_locations(years,days,name=name,plot=plot,save=save, order=order)
    return locs,fig
    
  # Create gtruth_unit object with Puma ripl  
  uni = makeUnit(mk_p_ripl(),gtruth_params)
  uni.loadAssumes()
  gtruth_locs,gtruth_fig = locs_fig(uni, gtruth_params['name'])
  filename = uni.store_observes(years,days)  # filename will use uni.name and random string
  
  # make inference model
  uni_inf = makeUnit(mk_p_ripl(),infer_params)
  uni_inf.loadAssumes()
  prior_locs,prior_fig = locs_fig(uni_inf,infer_params['name']+'_prior')

  # observe and infer (mutating the ripl in the Unit object)
  start = time.time()
  if use_analytics:
    analytics_obj_hists = ana_filter_inf(uni_inf, steps_iterations, filename)
  else:
    infer_prog(uni_inf, steps_iterations, filename, make_infer_string = make_infer_string)
    analytics_obj_hists = None
  print 'Obs and Inf: %s, elapsed: %s'%(infer_prog,time.time() - start)

  # posterior info (after having mutated uni_inf.ripl)
  posterior_locs,posterior_fig = locs_fig(uni_inf,infer_params['name']+'_post')

  # make a fresh ripl to measure impact of inference
  uni_fresh = makeUnit(mk_p_ripl(),infer_params)
  uni_fresh.loadAssumes()
  unit_objects = uni,uni_fresh,uni_inf, analytics_obj_hists
  
  all_locs = gtruth_locs, prior_locs, posterior_locs
  figs = gtruth_fig, prior_fig, posterior_fig

  return unit_objects, all_locs, figs

  

### Inference procedures for *infer_prog* arg in *synthetic_infer*

def make_poisson_infer_string( day, steps, day_to_hypers=None):
  if day_to_hypers is None:
    day_to_hypers = lambda day:10
  args = day_to_hypers(day), day, steps
  s='(cycle ((mh hypers all %i) (mh %i one %i)) 1)'%args
  return s

def make_multinomial_infer_string( day, steps, day_to_hypers=None):
  s='(cycle ((mh hypers all 10) (mh move %i %i)) 1)'%(day,steps)
  #s='(cycle ((mh hypers all 10) (mh move one %i)) 1)'%steps
  #s='(mh default one %i)'%steps
  try:
    s=multinomial_string[0]%(day,steps)
  except:
    s=multinomial_string[0]%steps
  return s


def test_inf():
  infer_prog_list = ['(cycle ((mh default one 10) (mh default one %i)) 1)',
                     '(cycle ((mh hypers all 10) (mh move %i %i)) 1)']
  # infer_prog_list = ['(cycle ((mh hypers all 10) (mh move %i %i)) 1)',
  #                    '(cycle ((func_pgibbs hypers all 10 2) (func_pgibbs move %i %i 5)) 1)' ] # note flipped steps and particle
  
  step_size = 10
  steps_prep = [1] + range(step_size,test_inf_limit,step_size)
  steps_list = [(steps,2) for steps in steps_prep]

  mses = {}
  for infer_prog in infer_prog_list:
    multinomial_string[0] = infer_prog
    print '\n-------\n infer_prog: ', infer_prog
    for steps in steps_list:
      print '\n------\n steps', steps
      ou = test_multinomial_reconstruction( steps, True, plot=False)
      mses[(infer_prog,steps)] = ou[-2]

  mses_seq = {'steps_list':steps_list} # recon mse only
  for infer_prog in infer_prog_list:
    mses_seq[infer_prog] = []
    for steps in steps_list: # mses = [ key:(recon[pr,po],hyps) ]
      mses_seq[infer_prog].append( mses[(infer_prog,steps)][0][1] )

  return mses, mses_seq



## GLOBAL VARS
multinomial_string=['(cycle ((mh hypers all 10) (mh move %i %i)) 1)']

test_inf_limit = 20
if len(sys.argv)>1:
  pass
  #test_inf_limit = int( sys.argv[1] )
  
global_order='F'
## test for order: with C order, plots should have counts in C order
## presumably we still want to leave the indices as they are
## (i.e. matrix-style vs. graph style)



def filter_inf(unit, steps_iterations, filename=None, make_infer_string=None, record_prog=None, verbose=False):
  """Loop over days, add all of a day's observes to birds unit.ripl. Then do multiple loops (iterations)
     of inference on move(i,j) for the previous day and on the hypers. Optionally
     take a function that records different 'queryExps' in sense of Analytics."""

  steps,iterations = steps_iterations
  args = str(unit), unit.name, steps, iterations
  print 'filter_inf. Model: %s Name: %s, Steps:%i, iterations:%i'%args
                                                         
  def basic_inf(ripl,year,day):
    'Make inference string given *year,day* and do inference'
    for iteration in range(iterations):
      inf_string = make_infer_string(day, steps)
      # TODO: change verbose setting
      if verbose or True: print 'iter: %i, inf_str:%s'%(iteration,inf_string)
      ripl.infer( inf_string )
      
  
  def record(unit):  return get_hypers(unit.ripl, unit.num_features)

  records = {}
  for y in unit.years:
    for d in unit.days:
      print 'before obs from file'
      ou = unit.observe_from_file([y],[d],filename)
      print ou[1] 

      if d>0:
        basic_inf(unit.ripl, y, d-1)
      
      if record_prog:
        records[(y,d)] = record_prog(unit)
      else:
        records[(y,d)] = record(unit)

  unit.last_inference_records = records



def smooth_inf(unit,steps_iterations,filename=None,**kwargs):
  '''Like *filter_inf* but observes all days at once and does inference on
   everything in moves2(i,j)'''
  steps,iterations = steps_iterations  
  args = unit.name, steps, iterations
  print 'smooth_inf. Name: %s, Steps:%i, iterations:%i'%args

  # observe all data
  unit.observe_from_file(unit.years, unit.days, filename=filename)
    
  for iteration in range(iterations):
    unit.ripl.infer('(mh hypers one 10)')
    unit.ripl.infer('(mh move one %i)'%steps)      
  return unit


# Inference procedure for analytics: could be integrated with *synthetic_bird_infer*.
# Note that the unit object is given incremental updates simultaneously with
# ana object and so gets  same observes at same time (though no inference is done on unit.ripl)
def ana_filter_inf(unit,  steps_iterations, filename=None, query_exps=None, verbose=False):
  '''Incremental inference on Analytics object. General pattern: loop over observes,
     add a set of them to Unit and Analytics objects [also add query expressions],
     then run Analytics inference.
     Here we specialize the observes and infers to Birds model.'''

  ana = unit.getAnalytics(unit.ripl, mutateRipl=True)
  
  steps,iterations = steps_iterations  
  args = unit.name, steps, iterations
  print 'ana filter_inf. Name: %s, Steps:%i, iterations:%i'%args
                                                         

  def analytics_infer(ripl,year,day):
    '''Analog of *basic_inf* in *filter_inf* with inference by
       analytics and history stored. Analytics only records after
       a full run of an inf_prog given as a string. Hence we run Analytics
    after every iteration (where we should do many iterations).'''
    
    hists = []
    
    for iteration in range(iterations):
      # inference program specifies 'block':(day-1)
      # as observe on *day* calls *move* for (day-1)
      latents = '(mh move %i %i)'%( day-1, steps)
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
      observes_yd = unit.observe_from_file([y],[d],filename,no_observe_directives=True)
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
    data_len = 5
    data = [10]*data_len + [25]*data_len + [50]*data_len
    observes = [('(normal x 10)',val) for val in data]+ [('(normal x .1)',30)]

    ana = Analytics(v,mutateRipl=True )
    hists = []
    vv = ana.ripl if not mripl else ana.mripl
    xs = []

    for i,obs in enumerate(observes):
        ana.updateObserves( [ obs ] )
        h,_ = ana.runFromConditional( 40, runs=1)
        x_value = np.mean( vv.sample('x') )
        xs.append(x_value)
        print i,' x: ', x_value
        hists.append( h )
    
    assert np.mean(xs[2:data_len]) < np.mean(xs[data_len:2*data_len])
    assert np.mean(xs[data_len:2*data_len]) < np.mean(xs[2*data_len:])

    return vv,ana,hists


# Produce a params dict for testing inference
# (be wary of mutating entries without copying first)

def get_multinomial_params(params_name='easy_hypers'):
  return get_params( params_name='easy_hypers', model='multinomial')

def get_params(params_name='easy_hypers', model='poisson'):
  'Function for producing params for Multinomial Unit object'
  
# 'easy_hypers', currently uses 'must move exactly onestep away'
# and 'avoid diagonal', but weigths are [1,0], so diagonal does nothing.
  if params_name in ('easy_hypers','easy_d4_s33_bi4_be10'):
    name = 'easy_hypers'
    Y, D = 1, 4
    years,days = range(Y),range(D)
    max_day = D
    height,width = 3,2
    feature_functions_name = 'one_step_and_not_diagonal'
    features,python_features_dict = make_features_dict(height, width, years,
                                                       days,
                                                       feature_functions_name = feature_functions_name)
    num_features = len( python_features_dict[(0,0,0,0)] )
    learn_hypers = False
    hypers = [1,1,0,0][:num_features]
    hypers_prior = ['(gamma 6 1)']*num_features
    num_birds = 20
    softmax_beta = 4
    load_observes_file=False
    venture_random_seed = 1
    dataset = 0
    observed_counts_filename = ''


  elif params_name in ('ds2','ds3'):
    dataset = 2 if params_name=='ds2' else 3
    width,height = 10,10
    num_birds = 1000 if dataset == 2 else 1000000
    name = "%dx%dx%d-train" % (width, height, num_birds)
    Y,D = 1, 4
    years = range(Y)
    days = []
    max_day = D
    hypers = [5, 10, 10, 10] 
    num_features = len(hypers)
    hypers_prior = ['(gamma 6 1)']*num_features
    learn_hypers = False
    features = None
    softmax_beta = None
    load_observes_file = None
    venture_random_seed = 1

  params = dict(name = name, dataset = dataset,
                years=years,  days = days, max_day = max_day,
                height=height, width=width,
                features=features, num_features = num_features,
                learn_hypers=learn_hypers, hypers = hypers, hypers_prior = hypers_prior,
                num_birds = num_birds, softmax_beta = softmax_beta,
                load_observes_file = load_observes_file,
                venture_random_seed = venture_random_seed,
                observed_counts_filename = observed_counts_filename)

  return params

 
def poi(params_name='easy_hypers'):
  params = get_params(params_name=params_name, model='poisson')
  model = 'poisson'
  gtruth_params, infer_params = params.copy(), params.copy()
  out = synthetic_infer(model, gtruth_params, infer_params, smooth_inf, (0,0), False, True)
  
  return out


# Once some params settings have been finalized, this should be a 
# test that hypers inference has worked correctly.
# General version would generate some hypers and then test
# the learning of them. But working with fixed params is ok also.

def test_easy_hypers_multinomial(use_analytics=False, steps_iterations=None):
  steps_iterations = (20,2) if not steps_iterations else steps_iterations
  easy_params = get_multinomial_params('easy_hypers')
  out = test_multinomial_reconstruction( steps_iterations, test_hypers=True,
                                     plot=True, infer_prog = filter_inf, use_analytics=use_analytics)
  unit_objects, params, all_locs, all_figs, mses, all_hypers = out

  gtruth_unit =  unit_objects[0]
  assert isinstance(gtruth_unit,Multinomial)
  assert not gtruth_unit.learn_hypers
  assert gtruth_unit.hypers == easy_params['hypers']
  
  latent_mse_prior, latent_mse_post = mses[0]
  assert 1  > latent_mse_prior > latent_mse_post
  assert .05 < latent_mse_post < 0.8

  hypers_mse_prior, hypers_mse_post = mses[1]
  assert 5 > hypers_mse_prior
  assert 50 > hypers_mse_post
  # mse can be large because scaling of hypers does well (up to point where prior kills it)

  # inferred hypers: hypers0 > hypers1
  assert all_hypers[-1][0] > all_hypers[-1][1]
  assert all_hypers[-1][1] < .5

  try:
    ana,hists = unit_objects[-1]
  except:
    ana,hists = None,None

  if use_analytics:
    from venture.unit.history import historyStitch
    hists_stitched = historyStitch(hists)
    assert 0.1 > abs( hists_stitched.nameToSeries['hypers0'][0].values[-1] - all_hypers[-1][0] )
    assert -2 < hists_stitched.averageValue('hypers0') < 5

  print '\n\n Passed "test_easy_hypers_multinomial"'

  return out,ana,hists

    
## Test non-Analytics reconstruction AND hypers inference. MH-Filter is *filter_inf* vs. *smooth_inf*.
## Testing involves computing mse for latents and hypers.
## Gets params from *get_multinomial_params*
def test_multinomial_reconstruction(steps_iterations, test_hypers=False, plot=True,
                                infer_prog=filter_inf, use_analytics=False):
  return test_reconstruction(steps_iterations, test_hypers, plot,infer_prog,
                             use_analytics, model = 'multinomial')


def test_reconstruction(steps_iterations, test_hypers=False, plot=True,
                        infer_prog=filter_inf, use_analytics=False, model='poisson'):

  plt.close('all')
  params = get_params('easy_hypers', model)
  order = global_order

  # copy and specialize params for gtruth and inference
  gtruth_params  = params.copy()
  infer_params = params.copy()
  gtruth_params['name'] = 'gtruth'
  infer_params['name'] = 'infer'

  # inference string
  if test_hypers: infer_params['learn_hypers'] = True
  
  if model=='poisson':
    make_infer_string = lambda d,s: make_poisson_infer_string(d,s,None)
  else:
    make_infer_string = make_multinomial_infer_string 
    
                              
  # generate synthetic data and do inference                                                 
  inf_out = synthetic_infer(model, gtruth_params, infer_params, infer_prog, steps_iterations,
                            make_infer_string = make_infer_string, plot=plot, use_analytics=use_analytics, order=order)

  # unpack results
  unit_objects,all_locs,all_figs = inf_out
  gtruth_unit, fresh_unit, inf_unit, analytics_obj_hists = unit_objects
  gt_locs,prior_locs,post_locs = all_locs
  gt,pri,post = all_figs

  # View the normalized dist on (0,0) for a few cells. Compare to
  # gtruth moves plots as check
  if plot:
    num_cells = gtruth_params['height']*gtruth_params['width']
    check_cells = range( min( num_cells, 5) )
    plot_cell_to_prob_dist(gtruth_params['height'],gtruth_params['width'], gtruth_unit.ripl,
                        check_cells, year=0, day=0, order= order, name='Gtruth')

  # compute test statistics
  mse_gt = lambda l: mse(gt_locs,l,gtruth_params['years'],gtruth_params['days'])
  mses = [ (mse_gt(prior_locs),mse_gt(post_locs) ) ]
  print 'prior,post mses: %.2f %.2f'%mses[0]

  if test_hypers:
    unit_only = unit_objects[:-1]
    mse_hypers_gt = lambda unit_obj: compare_hypers(gtruth_unit,unit_obj)
    mses_hypers = mse_hypers_gt(fresh_unit),mse_hypers_gt(inf_unit)
    mses.append(mses_hypers)
    
    print '\n---\n gt_hypers, fresh_prior_hypers, post_hypers:'
    all_hypers= [get_hypers(unit.ripl,params['num_features']) for unit in unit_only]
    print all_hypers

    print '\nprior,post hypers mses: %.2f %.2f'%(mses[-1])
  else:
    all_hypers = None
    
  return unit_objects, params, all_locs, all_figs, mses, all_hypers


###  Tests for Analytics Multinomial Incremental Inference
# 1. Series of basic unit tests for Multinomial in Analytics
# 2. Filter Multinomials hypers inference using *ana_filter_inf* above.
def test_ana_inf(mripl=False):
  'Series of tests for inference on analytics object for Multinomial'
  params = get_multinomial_params()
  params['learn_hypers'] = True
  
  unit = Multinomial(mk_p_ripl(),params)
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
  gt_unit = Multinomial(mk_p_ripl(),gt_params)
  gt_unit.loadAssumes()
  filename = gt_unit.store_observes(gt_unit.years,gt_unit.days)

  inf_params = params.copy()  # which must have *learn_hypers*=True
  inf_unit = Multinomial(mk_p_ripl(),inf_params)
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
  _,hists = ana_filter_inf(inf_unit, steps_iterations, filename, verbose=True)
  
  return unit,ana, inf_unit, inf_ana, hists



def run_all_tests(plot=False):
  for boo in True,False:
    test_persistent_ripl_analytics(mripl=boo)
    test_ana_inf(mripl=boo)

  test_hypers = (True,False)
  infer_prog = (filter_inf,smooth_inf)
  steps_iterations = ( (0,0), (1,1) )
  settings = product(steps_iterations, test_hypers, infer_prog )
  for steps_iterations, test_hypers, infer_prog in settings:
    print 'steps_iterations, test_hypers, infer_prog', steps_iterations, test_hypers, infer_prog
    test_multinomial_reconstruction(  steps_iterations, test_hypers, plot, infer_prog)














