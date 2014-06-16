from synthetic import *

def recon(steps,iterations):
    return test_recon((steps,iterations), test_hypers=True,plot=False,use_mh_filter=True)
    
recon(0,0)
inf_hypers = get_hypers(uni[2].ripl,2)
print 'inf_hypers',inf_hypers
print 'passed test'
uni,params,locs,figs,mses = recon(10,5)
inf_hypers = get_hypers(uni[2].ripl,2)
print 'inf_hypers',inf_hypers
