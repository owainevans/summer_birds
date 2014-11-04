  

parameter_short_name_to_changes = {'minimal_onestep_diag10':
                           {},

                           'multinomial_onestep_diag105_size32':
                           {'short_name': 'multinomial_onestep_diag105_size32',
                            'years': range(2),
                            'days': range(2),
                            'height': 2,
                            'width': 3,
                            'hypers':[1,0.5],
                            'num_birds': 2 },

                           'multinomial_onestep_diag105_size33':
                           {'short_name': 'multinomial_onestep_diag105_size33',
                            'years': range(2),
                            'days': range(3),
                            'height': 3,
                            'width': 3,
                            'hypers':[1,0.5],
                            'num_birds': 12,
                            'phi_constant_beta': 1},

                            'poisson_onestep_diag105_size44':
                            {'short_name': 'poisson_onestep_diag105_size44',
                            'years': range(2),
                            'days': range(4),
                            'height': 4,
                            'width': 4,
                            'hypers':[1,0.5],
                            'num_birds': 400 },

                           'dataset1':
                           {'short_name':'dataset1',
                            'years': range(30),
                            'days': range(20),
                            'width':4,
                            'height':4,
                            'num_birds': 1, 
                            'num_features': 4,
                            'hypers': [5,10,10,10],
                            'prior_on_hypers': ['(gamma 6 1)'] * 4,
                            'features_loaded_from': "data/input/dataset1/onebird-features.csv",
                            'max_years_for_experiment': 2, 
                            'max_days_for_experiment': 19,
                            'phi_constant_beta': 1},


                          'dataset1_small':
                           {'short_name':'dataset1',
                            'years': range(30),
                            'days': range(20),
                            'width':4,
                            'height':4,
                            'num_birds': 1, 
                            'num_features': 4,
                            'hypers': [5,10,10,10],
                            'prior_on_hypers': ['(gamma 6 1)'] * 4,
                            'features_loaded_from': "data/input/dataset1/onebird-features.csv",
                            'max_years_for_experiment': 2, 
                            'max_days_for_experiment': 5,
                            'phi_constant_beta': 1},

                           'dataset2':
                           {'short_name':'dataset2',
                            'years': range(3),
                            'days': range(20),
                            'width':10,
                            'height':10,
                            'num_birds': 1000, 
                            'num_features': 4,
                            'hypers': [5,10,10,10],
                            'prior_on_hypers': ['(gamma 6 1)'] * 4,
                            'features_loaded_from': "data/input/dataset2/10x10x1000-train-features.csv",
                            'max_years_for_experiment': 2, # FOR NOW WE LIMIT THIS
                            'max_days_for_experiment': 4,
                            'phi_constant_beta': 1}

                         }

for k,v in parameter_short_name_to_changes.iteritems():
  if 'short_name' in v  and  k != v['short_name']:
    parameter_short_name_to_changes[k]['short_name'] = k

  if k.startswith('dataset'):
    assert v['phi_constant_beta'] == 1












