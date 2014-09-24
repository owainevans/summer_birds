import cPickle as pickle
from os import listdir
from os.path import isfile, join
import os    
from utils import ensure
from synthetic import *
import matplotlib.pylab as plt

## PLAN (formerly in synthetic.py)

# FEATURES TO ADD
# 
# 
# batch_inf(...) with same args, etc, as birdsUnit
#
# 

# experiments: [ experiment ... ]
# experiment:  { 'type': <shorthand_inference_string_key>,
#                 'logscore': ...,
#                 'runtime': ...,
#                 'other': a dictionary with all keys needed to call either batch_inf or filter_inf
#              }]
#
# load_experiments(directory) => experiments
#   treats all files in the directory as experiment lists
#   loads them all, and appends them together
#
# run_experiment(experiment) => None (but modifies experiment to include logscore and runtime)
#   pefrorms the computation, using the 'type' field to drive make_params, and storing params in the 'other' key of the
#   current experiment, saving the resulting logscore, runtime, etc
#
#   calls batch_inf, filter_inf, etc as appropriate
#  
#   v2: uses a worker pool w/ multiprocessing
# 
# generate_experiment_data([experiments], directory, name, overwrite=False)
#   runs all the given experiments, saves the results to the directory to the file name.dat, and if overwrite is True,
#   actually overwrites the file. 
#
#   
# reduce_by_type([experiments]) => { <type as str> : <vals as list> } 
# for all types <foo> in the experiments:
#   [e['logscore'] for e in experiments if e['type'] = <foo>]
#
# plot( <reduced data>, ground_truth_data, colors = { <type as str> : <color, etc> } )
#   histogram overlay of the reduced data, aggregated by type already, in given colors (placeholder for other layout)
#   vertical line at ground_truth_data value
#
#   colors act as overrides, otherwise trust the default
#
# generate_synthetic_data_<foo>() => <dict that can go in experiment['other'], with raw data values, pickleable>
#
# save_synthetic(synth, dir) # uses pickle to dump synth to a file
# synth = load_synthetic(dir)
#

# ****
# 
# manual code to make the initial list of experiments, after calling the appropriate generate_synthetic_data_<foo>():
#
# <v2: name from command line>
#
# synth = generate_synthetic_data_<foo>()
# save_synthetic(synth, dirname)
# 
# experiments = []
#
# for type in ['asdf' ... ]:
#       experiment = {}
#       experiment['other'] = {}
#
#       <add extra crap to other as appropriate, using your maker procedures>
#
#       experiment['other']['data'] = synth
#       experiments.append(experiment)
# 
# generate_experiment_data(experiments, ..., name = 'packet_of_results_1')
# 
# **** separate script below ***
#
# expts = load_experiments(... <v2: read from cmd line> ...)
# synth = load_synthetic()
#
# reduction = reduce_by_type(expts)
# plot(reduction, synth['ground_truth_logscore'])
#

# write w run_expt producing total random junk w/o a ripl at all, and test end-to-end
#
# refactor make_inference_string to return a list of inference strings, one per day
# refactor batch_inf and filter_inf to take generated inf strings rather than generators so that all the data can be flat in experiment
#
# write run_expt and test manually, comparing to past experience
# incrementally generate some data to end up with a first real 'candidate' graph, albeit on too little data
# add more data and iterate as desired


# PART OF EXPT GENERATION:
#
# get_parameters(<config info>) => <params dict>
# make_inference_string(day, #steps) => <inference prog src>
#
# PART OF EXPT RUNNING:
#
# birdsUnit(ripl, <params dict>)
# birdsUnit.makeAssumes, loadObserves, ...
#
# filter_inf(birdsUnit, logger_proc, amount_of_inference, inference_string_maker, datafilename) => None (but birdsUnit modified, so
# birdsUnit.last_inference_records = {(d,y): logger_proc(ripl)}
# batch_inf(...)
#

# synthetic_infer(<config info + inf config info>, logger(ripl)) => gtUnit, ripl_with_results, 
# 
#  




# we select a dirname (name for the synth data params).
# dirname is used to save synth data and params in 'synth.dat'
# we then use dirname to generate_experiment_data
# every run of this creates a file (or overwrites existing)
# which is a pickled list of experiments. name can't start 
# with syn. 

os.chdir('/home/owainevans/summer_birds')

get_ripl = mk_p_ripl
plot_all = True


def draw_locs_fig(unit,plot=True,name='draw_locs'):
  'Call getBirdLocations and draw_bird locations'
  years,days  = unit.years,unit.days
  locs = unit.getBirdLocations(years,days,predict = True)
  fig = unit.draw_bird_locations( years, days,name=name,plot=plot,save=False)
  return locs,fig


# also saves gtruth params
def generate_save_synthetic_data(params,directory):
  gtruth_unit = OneBird(get_ripl(), params)
  gtruth_unit.loadAssumes()

  years,days  = gtruth_unit.years,gtruth_unit.days
  gtruth_locs,gtruth_fig = draw_locs_fig(gtruth_unit,plot=plot_all)

  # store synthetic data
  filename = directory + 'synthetic_data.dat'
  gtruth_unit.store_observes(years,days,filename)
  
  # store gtruth params
  with open(directory+'synthetic_gtruth_params.dat','w') as f:
    pickle.dump(params,f)

  print 'saved synthetic data and gtruth params'

  return gtruth_unit,gtruth_locs,gtruth_fig




def filter_inf(unit, filename, infer_string_list):
  args = str(unit), unit.name, steps,
  print 'filter_inf. Model: %s Name: %s, Steps:%i '%args
                                                         
  def basic_inf(ripl,year,day):
    inf_string = infer_string_list(day)
    ripl.infer( inf_string )
      
  def record(unit):  return get_hypers(unit.ripl, unit.num_features)

  records = {}
  for y in unit.years:
    for d in unit.days:
      ou = unit.observe_from_file([y],[d],filename)
      print 'first observe: ou[1]' 

      if d>0:
        basic_inf(unit.ripl, y, d-1)
        records[(y,d)] = record(unit)

  unit.last_inference_records = records
  unit.logscore = unit.ripl.get_global_logscore()


def seq_block_pgibbs_make_inf_string(day,steps):
  s='(cycle ((func_pgibbs hypers one 10 2) (func_pgibbs move2 %i %i 2)) 1)'%(day,steps)
  return s

def seq_block_mh_make_inf_string(day,steps):
  s='(cycle ((mh hypers one 10) (mh move2 %i %i)) 1)'%(day,steps)
  return s

def make_inf_list(params,steps, make_string):
    l=[make_string(day,steps) for day in params['days']]
    return l

# specify params for synth data 
# params are also needed to construct inference prog / ripl
params_name = 'easy_d4_s33_bi4_be10'
#featurefunctions__maxDay_size_num_birds_softmaxbeta

params = get_params(params_name, 'onebird')
gtruth_params  = params.copy()
infer_params = params.copy()
gtruth_params['name'] = 'gtruth'
infer_params['name'] = 'infer'

ensure(params_name)
directory = params_name +'/'

generate_save_synthetic_data(gtruth_params,directory)


## specifying an inference prog:
# need list of inference strings or single one (filter/batch)
# for list, need num_days from params

exp_seed1=dict( type = 'seq_block_mh', 
                steps = 50,
                make_inf_string = seq_block_mh_make_inf_string)
exp_seed2 = exp_seed1.copy()
exp_seed3=dict( type = 'seq_block_pgibbs', 
                steps = 50,
                make_inf_string = seq_block_pgibbs_make_inf_string)

exp_seeds = (exp_seed1,exp_seed2,exp_seed3)



# loop over seeds, generate experiments
experiments = []
for seed in exp_seeds:
  experiment = {'type': seed['type'],
                'logscore':[],
                'runtime':[] }
  infer_string_list = make_inf_list(params, seed['steps'],
                                      seed['make_inf_string'])

  other= { 'infer_string_list': infer_string_list,
           'steps': seed['steps'],
           'gtruth_params': gtruth_params,
           'infer_params': infer_params,}
           #'synthetic_data': synthetic_data, }
           ## synthetic data always found in directory (but would be ideal to include it here also)
  
  experiment['other'] = other

  experiments.append( experiment )


def run_experiment(experiment):
  infer_params = experiment['infer_params']
  infer_string_list = experiment['infer_string_list']
  steps = experiment['steps']
  
  infer_unit = OneBird(get_ripl(),infer_params)

  start = time.time()
  filter_inf(infer_unit, filename, infer_string_list)
  elapsed = start - time.time()
  
  post_locs,post_fig = draw_locs_fig(infer_unit,plot=plot_all)
  experiment['logscore'] = [infer_unit.logscore]
  experiment['runtime'] = [elapsed]
  
  


def load_experiments(directory):
  experiments = []
  for f in listdir(directory):
    if isfile(join(directory,f)) and not f.startswith('syn'):
        
      with open(join(directory,f),'r') as experiments_list:
        experiments.extend(pickle.load(experiments_list))
  return experiments



def generate_experiment_data(experiments, directory, name, overwrite = False):
  assert not name.startswith('syn')
  filename = directory + name + '.dat'

  if os.path.isfile(filename) and not overwrite:
    print 'generate_experiment_data avoided overwriting %s'%filename  
    return  # or should we run anyway?
   
  map(run_experiment,experiments)
  
  with open(filename,'w') as f:
    pickle.dump(experiments,f)



def reduce_by_type(experiments,measure='logscore'):
  all_types = set([e['type'] for e in experiments])
  reduced_data = {}
  
  for type in all_types:
    reduced_data[type] = []
    
    for e in experiments:
      if e['type']==type:
        reduced_data[type].extend( e[measure] )
        
  return reduced_data


def plot_reduced(reduced_data,ground_truth_data=None,colors=None):
  fig,ax = plt.subplots()
  for type,vals in reduced_data.items():
    n = len(vals)
    ax.hist(vals, normed=True, label=type+', n=%i'%n)
  ax.legend()
  #ax.set_title('


def test_reduce():
  ## uses global *experiments*
  generate_experiment_data(experiments,directory,'testred2')
  exps = load_experiments(directory)
  red = reduce_by_type(exps,measure='logscore')
  plot_reduced(red)


def test_save_load():
  # should use fresh directory. uses global *experiments*
  my_exps = experiments[:]
  map(run_experiment,my_exps)

  name = 'test_run'
  generate_experiment_data(experiments[:],directory,name)
  out_load_exp = load_experiments(directory)

  assert len(my_exps) == len(out_load_exp)
  for k,v in my_exps[0].items():
    assert k in out_load_exp[0]
    assert v == out_load_exp[0][k]








