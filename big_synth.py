from synthetic import *

def recon(steps,iterations):
    out = test_recon((steps,iterations), test_hypers=True,plot=False,use_mh_filter=True)
    uni,params,locs,figs,mses = out
    inf_hypers = get_hypers(uni[2].ripl,2)
    print 'inf_hypers',inf_hypers
    return out
    
#test    
recon(0,0)
print 'passed test'

uni,params,locs,figs,mses = recon(10,5)

