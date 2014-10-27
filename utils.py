import venture.shortcuts as s
import venture.value.dicts as venturedicts
from venture.ripl.utils import strip_types_from_dict_values
import numpy as np
from scipy import misc
import os
import matplotlib.pylab as plt
from itertools import product



def backend_to_ripl_thunk(backend_string):
  if backend_string == 'puma':
    return s.make_puma_church_prime_ripl
  elif backend_string == 'lite':
    return s.make_lite_church_prime_ripl
  else:
    assert False, 'Bad backend_string'


def parseLine(line):
  return line.strip().split(',')

def loadCSV(filename):
  with open(filename) as f:
    return map(parseLine, f.readlines())

def update(dict, key, data):
  if key not in dict:
    dict[key] = []
  
  dict[key].append(data)


def read_features(filename, max_year, max_day): 
  csv = loadCSV(filename)
  data = {}
  
  for i,row in enumerate(csv[1:]):
    
    ## TODO: assumes keys are len 4
    ## NOTE converstion from MATLAB 1 indexing to zero indexing
    keys = tuple( int(k)-1 for k in row[:4])
    year, day = keys[:2]
    
    if year > max_year:
        print 'max_year: %i, first table entry missed: \n %s' % (max_year, keys )
        break

    if day > max_day:
      if max_year == year:
        print '(max_year, max_day): (%i,%i), first entry missed: \n %s' % (max_day, max_year, keys)
        break
      else:
        continue
    
    features = map(float, row[4:])
    data[keys] = features
                                                                    
  return data

def read_observations(filename):
  # structure of csv:
  # row: year,day, observed_birds_cell0, observed_birds_cell1, etc.

  csv = loadCSV(filename)
  years = {}

  for row in csv[1:]:
    [year, day] = map(int, row[:2])
    cells = map(float, row[2:])
    update(years, year-1, (day-1, cells)) ## 1 to 0-indexing
  
  return years

def readReconstruction(params):
  filename = "data/ground/dataset%d/10x10x%d-reconstruction-ground.csv" % (params["dataset"], params["num_birds"])
  csv = loadCSV(filename)
  
  bird_moves = {}
  
  for row in csv[1:]:
    bird_moves[tuple(int(k)-1 for k in row[:4])] = float(row[4])
  
  return bird_moves

def writeReconstruction(params, bird_moves):
  filename = "data/output/dataset%d/10x10x%d-reconstruction-ground.csv" % (params["dataset"], params["num_birds"])
  
  with open(filename, 'w') as f:
    for key, value in sorted(bird_moves.items()):
      f.write(','.join(map(str, [k+1 for k  in key] + [value])))
      f.write('\n')


def ensure(path):
  if not os.path.exists(path):
    os.makedirs(path)




##------------------------------------
### MODEL AND SYNTHETIC DATA UTILS
##------------------------------------



class Observe_range(dict):
  def __init__(self, days_list=None, years_list=None, cells_list = None):
    args = (days_list, years_list, cells_list)
    for arg in args:
      if arg is not None:
        assert isinstance(arg,(list,tuple))
        assert all( [isinstance(el,int) for el in arg] )

    self['days_list'] = days_list
    self['years_list'] = years_list
    self['cells_list'] = cells_list
    
  def get_year_day_cell_product(self):
    return product(self['years_list'], self['days_list'],self['cells_list'] )

  def assert_is_observe_sub_range(self, bigger_observe_range):
    for k,v in self.items():
      assert v is not None
      assert set(v).issubset( set(bigger_observe_range[k]))

  def replace_none_with_super_range(self, super_range):
    for k,v in self.items():
      if v is None:
        self[k] = super_range[k]
    
  def copy_observe_range(self):
    return Observe_range(**self.copy())

  def copy_dict_only(self):
    return self.copy()



def make_grid(height,width,top0=True,lst=None,order='F'):
  '''Turn *lst* into np.array, optionally having 0 start at top/bottom. *lst* defaults to range(width*height)'''
  if lst is not None:
    assert isinstance( lst[0] , (int,float) )
    
  l = np.array(range(width*height)) if lst is None else np.array(lst)
  grid = l.reshape( (height, width), order=order)
  if top0:
    return grid
  else:
    grid_mat = np.zeros( shape=(height,width),dtype=int )
    for i in range(width):
      grid_mat[:,i] = grid[:,i][::-1]
    return grid_mat


def plot_save_bird_locations(unit, years, days, title=None, save=True,
                             plot=True, order='F', verbose=False, directory_filename=None):

  assert isinstance(years, list)
  assert isinstance(days, list)
  assert order in ('F','C')
  title = unit.params['short_name'] if title is None else title
  bird_locs = unit.days_list_to_bird_locations(years,days)
  height, width = unit.params['height'], unit.params['width']

  if verbose:
    features_dict = unit.params['features_as_python_dict']
    max_years = max(unit.params['years'])
    max_days = max(unit.params['days'])
    assert len(features_dict) <= (height*width)**2 * (max_years+1) * (max_days+1)

    dash_line = '\n----------\n'
    print dash_line + 'Features dict (up to 10th entry) for year,day = 0,0'
    count = 0
    for k,v in features_dict.iteritems():
      if k[0]==0 and k[1]==0 and count<10: 
        print k[2:4],':',v
        assert isinstance(v[0],(int,float))
        count += 1

    feature0_from0 = [features_dict[(0,0,0,j)][0] for j in range(unit.cells)]

    print dash_line + 'feature_0 for jth cell for y,d,i = (0,0,0), order=%s, 0 at top \n'%order
    print make_grid( height, width, lst=feature0_from0, order=order)


    ## Print map from bird_locs indices to ij positions on grid
    indices = range(len(bird_locs[years[0]][days[0]]))
    im_info = make_grid(height, width, indices, order=order )
    print dash_line + '''
    Map from *bird_locs* indices (which comes from Venture
    function) to grid via function *make_grid* (order is %s,
    0 index at top) \n'''%order
    print im_info


  grids = {}
  for y,d in product(years,days):
    grids[(y,d)] = make_grid(height, width, lst=bird_locs[y][d], order=order)

  grid_to_num_birds = map(np.sum, grids.values())
  if multinomial_or_poisson(unit) == 'Multinomial':
    assert len( np.unique( grid_to_num_birds ) ) == 1
  max_num_birds =  np.max( grid_to_num_birds )


  ## FIXME what about large num days/years. need multiple plots
  nrows,ncols = len(days), len(years)
  fig,ax = plt.subplots(nrows,ncols,figsize=(4*ncols,2*nrows))

  for y,d in product(years,days):
    grid = grids[(y,d)]
    if ncols==1 and nrows==1:
      ax_dy = ax
    elif ncols==1:
      ax_dy = ax[d]
    elif nrows==1:
      ax_dy = ax[y]
    else:
      ax_dy = ax[d][y]

    my_imshow = ax_dy.imshow(grid,cmap='copper',
                             interpolation='none',
                             vmin=0, vmax=max_num_birds,
                             extent=[0,width,height,0])
    ax_dy.set_title('Bird counts: %s- y:%i d:%i'%(title,y,d))
    ax_dy.set_xticks(range(width+1))
    ax_dy.set_yticks(range(height+1))

  fig.tight_layout()  
  fig.subplots_adjust(right=0.67)
  cbar_ax = fig.add_axes([0.75, 0.7, 0.05, 0.2])
  fig.colorbar(my_imshow, cax=cbar_ax)

  
  if directory_filename is None:
    directory = 'bird_moves/'
    filename = directory + title + '.png'
  else:
    directory, filename = directory_filename
  
  ensure(directory)
  assert filename.startswith( directory )
  
  if save:    ## FIXME make images look better!
    fig.savefig( filename )
    print '\n Saved bird location images in %s \n'%filename

    for y,d in product(years,days):
      grid = grids[(y,d)]
      big_im = misc.imresize(grid,(200,200))
      misc.imsave(directory+'imsave_%02d.png'%d, big_im)
      print '\n Saved bird location images in %s \n'%directory

  return (fig,ax) if plot else None


def multinomial_or_poisson(unit):
  return unit.get_model_name()
  


def dataset_to_params(dataset):
  params = {'dataset': dataset}

  if dataset == 1:
    params['width'] = params['height'] = 4
    params['num_birds'] = 1    
    params['years'] = range(30)
    params['days'] = range(20)
  else:
    params['width'] = params['height'] = 10
    params['num_birds'] = 1000 if dataset == 2 else 1000000

    params['years'] = range(3)
    params['days'] = range(20)
    

  return params



def python_dict_to_venture_exp( d ):
  assert isinstance(d, dict)
  
  def seq_to_array_exp(seq):
    return ['array'] + list(seq)
  
  keys = ['array',]
  values = ['array',]
  for k,v in d.iteritems():
    keys.append( seq_to_array_exp( k ) )
    values.append( seq_to_array_exp( v ) )

  return ['dict', keys, values]
  

def exp_venture_string( exp ):
  if callable(exp) or isinstance(exp,dict):
    assert False, 'exp is callable or dict and not supported'
  if not isinstance(exp,list):
    return str(exp)
  else:
    return '('+ ' '.join( map(exp_venture_string,exp) ) + ')'


def python_features_to_venture_exp( features_dict ):
  exp = python_dict_to_venture_exp( features_dict )
  return exp_venture_string( exp )

def venturedict_to_pythondict(venturedict):
  remove_type_venturedict = dict(venturedict['value'].iteritems())
  return strip_types_from_dict_values(remove_type_venturedict)


def python_features_to_string_or_dict( python_features_dict, dict_string='string'):
  if dict_string=='string':
      return python_features_to_venture_exp(python_features_dict)
  else:
      return toVenture( python_features_dict )


def toVenture(thing):
  if isinstance(thing, dict):
    return venturedicts.val("dict", {k:toVenture(v) for k, v in thing.iteritems()})
  if isinstance(thing, (list, tuple)):
    return venturedicts.val("array", [toVenture(v) for v in thing])
  if isinstance(thing, (int, float)):
    return venturedicts.number(thing)
  if isinstance(thing, str):
    return venturedicts.symbol(thing)

# handles numbers, lists, tuples, and dicts
def toExpr(thing):
  if isinstance(thing, dict):
    return dictToExpr(thing)
  if isinstance(thing, (list, tuple)):
    return listToExpr(thing)
  return str(thing)  

def expr(*things):
  return "(" + " ".join(map(toExpr, things)) + ")"

def dictToExpr(dict):
  return expr("dict", dict.keys(), dict.values())

def listToExpr(list):
  return expr("array", *list)

def fold(op, exp_ind, ind, length):
  '''Outputs string of form (op  exp_0  exp_1 ... exp_(length-1) )
     where *op* is a string for a Venture SP and *exp_ind* is a
     string containing the string *ind*.'''
  return '(' + op + " " + " ".join([exp_ind.replace(ind, str(i)) for i in range(length)]) + ')'


def tree(op, exp, counter, lower, upper):
  average = (lower + upper) / 2
  if average == lower:
    return exp.replace(counter, str(lower))
  else:
    return '(' + op + " " + tree(op, exp, counter, lower, average) + ' ' + tree(op, exp, counter, average, upper) + ')'

 


##------------------------------------
## VIDEO UTILS
##------------------------------------   
from subprocess import call

def renderDot(dot,dirpath,i,fmt,colorIgnored):
  name = "dot%d" % i
  mkdir_cmd = "mkdir -p " + dirpath
  print mkdir_cmd
  call(mkdir_cmd,shell=True)
  dname = dirpath + "/" + name + ".dot"
  oname = dirpath + "/" + name + "." + fmt
  f = open(dname,"w")
  f.write(dot)
  f.close()
  cmd = ["dot", "-T" + fmt, dname, "-o", oname]
  print cmd
  call(cmd)

def renderRIPL(dirpath="graphs/onebird",fmt="svg",colorIgnored = False):
  dots = ripl.sivm.core_sivm.engine.getDistinguishedTrace().dot_trace(colorIgnored)
  i = 0
  for dot in dots:
    print "---dot---"
    renderDot(dot,dirpath,i,fmt,colorIgnored)
    i += 1

def avgFinalValue(history, name):
  series = history.nameToSeries[name]
  values = [el.values[-1] for el in series]
  return np.average(values)

def normalize(l):
  s = sum(l)
  return [e/s for e in l]

