import cPickle as pickle
from synthetic import *

def generate_synthetic_data(params):
  return {}

def save_synthetic_data(synthetic_data,dirname):
  'add name that denotes synthetic data'
  pass

def mh_make_inf_string(day,steps):
  s='(cycle ((mh hypers one 10) (mh move2 %i %i)) 1)'%(day,steps)
  return s

def make_inf_list(params,steps, make_string):
    l=[make_string(day,steps) for day in params['days']]
    return l

# specify params for synth data 
# params are also needed to construct inference prog / ripl
params_name = 'easy_hypers'

params = get_params(params_name, 'onebird')
gtruth_params  = params.copy()
infer_params = params.copy()
gtruth_params['name'] = 'gtruth'
infer_params['name'] = 'infer'

dir_name = '/'+params_name
synthetic_data = generate_synthetic_data(gtruth_params)
save_synthetic_data(synthetic_data,dir_name)
# save gtruth params?


## specifying an inference prog:
# need list of inference strings or single one (filter/batch)
# for list, need num_days from params

exp_seed1=dict( type = 'mh_sequential_blocks', 
                steps = 50,
                make_inf_string = mh_make_inf_string )

exp_seeds = (exp_seed1,)

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
           'infer_params': infer_params,
           'synthetic_data': synthetic_data, }
  
  experiment['other'] = other

  experiments.append( experiment )


def run_experiment(experiment):
  experiment['logscore'] = np.random.randint(100)
  experiment['runtime'] = .3*np.random.randint(100)
    
def generate_experiment_data(experiments, directory, name, overwrite = False):
  filename = 
  for experiment in experiments:
    run_experiment[experiment]
    with open(filename,'a') as f:
      pickle.dump(experiment,f)
#generate_experiment_data(experiments, name='pack_of_results')






