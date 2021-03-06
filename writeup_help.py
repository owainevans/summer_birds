from model import *

unit = Multinomial( mk_p_ripl(), make_params() )

def print_directives(v):
  dirs = []
  for directive in v.list_directives():
    dir_id = int(directive['directive_id'])
    
    dir_val = str(directive['value'])
    dir_type = directive['instruction']
    try:
      dir_text = v._get_raw_text(dir_id)
    except:
      dir_text = 'fail'
    if dir_type == "assume":
      dirs.append( "%s" % dir_text )

      #dirs.append( "%s:\t%s" % (dir_id, dir_text, dir_val) )
    elif dir_type == "observe":
      dirs.append( "%d: %s" % (dir_id, dir_text) )
    elif dir_type == "predict":
      dirs.append( "%d: %s:\t %s" % (dir_id, dir_text, dir_val) )
    else:
      assert False, "Unknown directive type found: %s" % str(directive)
  return dirs
    
## NOTE this is the generating not hypers learning. so we also need to fix the hypers to be gammas
# display

def disp():
  dirs = print_directives(unit.ripl)
  dirs = dirs[2:] # remove map and filter
  ## TODO CHANGE NUMBER BIRDS
  pref = ['[assume features <features_dict>]',
          '[assume num_birds 1]',
          '[assume phi_constant_beta 1]' ]
  
  dirs = pref + dirs[:]
  # filter
  new_dirs = []
  
  def filter_d(d):
    disj=(len(d) > 600,
          'fail' in d,
          'single' in d,
          'no_map' in d,)
    return any(disj)
          
  for d in dirs:
    if filter_d(d):
      pass
    else:
      new_dirs.append(d)

  print '\n\n'.join(new_dirs)
  return new_dirs
    


