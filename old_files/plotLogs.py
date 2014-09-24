import scipy.stats
import numpy as np
import matplotlib.pylab as plt
import os

names_block2='''getMoves_2303/ getMoves_2370/ getMoves_2653/ getMoves_4246/ getMoves_8749/ getMoves_8903/'''
# taken from folder 'block_new_cycle'


# ds3 last lines from folder /ds3_block_new_cycle
ds3_last = [(2, 49, 100, -199581.35188110976, 43794968103.0, (4.467383879543357, 2.911203977189646, 6.447483058452342, 5.8628303302623), 839.8548119068146),

(2, 45, 100, -73475.48626436318, 8437297289.0, (4.351776107302161, 10.373338278652932, 12.498465342639948, 11.466293523639093), 707.5084710121155),

(2, 45, 100, -112224.50737596091, 32343145702.0, (6.141136228091977, 13.541381458316987, 9.599260458211086, 12.050758722674344), 828.0552699565887),

(2, 48, 100, -125757.68958976108, 37016373001.0, (4.746110174294733, 6.039233185699792, 10.22170450911841, 8.479255655363414), 752.313246011734),

(2, 50, 100, -35782.6689953414, 7075125594.0, (3.6515471789366796, 9.52462050253949, 9.228424856494241, 9.098497958345822), 813.7161929607391),

(2, 40, 100, -57564.35072664273, 12210610389.0, (5.017946039975852, 9.474982546933486, 12.245076231178396, 10.596232860758263), 790.5305559635162),

(2, 50, 100, -47857.11439539446, 7213603130.0, (5.908934859455975, 14.18405017953044, 13.946689014838762, 13.739446990723874), 816.9821081161499), ]

ds3_params = [line[-2] for line in ds3_last]
for line in ds3_params:
    print np.round(line,2)
print 'DS3'
print 'mean of final samples:',np.round(np.mean(ds3_params,axis=0),2)
print 'std of final samples:',np.round(np.std(ds3_params,axis=0),2)
print 'stderr of final samples:',np.round(scipy.stats.sem(ds3_params,axis=0),2)
print 'DS3---------\n'

best_day_logscore = [-14.91, -98.79, -239.21]


prior_val = 'block2'#'ds2_long'#,'ds3'#, 'g05_005/'#, 'g1_05/'    prior_name = prior_val


if prior_val in 'ds3':
    prior_name = 'g05_005/'
    path = '/home/owainevans/birds/ds3_long_gamma_prior/'
    names = names_ds3_05_005
    print 'Prior: ',prior_name+' ', prior_val
elif prior_val in 'ds2_long':
    prior_name = 'g05_005/'
    path = '/home/owainevans/birds/long_gamma_prior/'
    names = names_ds2_05_005
    print 'Prior: ',prior_name+' ', prior_val
elif prior_val in 'block2':
    prior_name = 'g1_01/'
    path = '/home/owainevans/birds/block_new_cycle/'
    names = names_block2
    print 'Prior: ',prior_name+' ', prior_val

elif prior_val in 'block3':
    prior_name = 'g1_01/'
    path = '/home/owainevans/birds/ds3_block_new_cycle/'
    names = names_block3
    print 'Prior: ',prior_name+' ', prior_val
    

burn_in = 100; cut_off = 102 #min(40,allParams.shape[0])
print 'burn_cut',burn_in, cut_off,'\n'

names= names.split()
dump_names = []
for name in names:
    filename = path + prior_name + name + 'posteriorRunsDump.py'
    if os.path.isfile(filename):
        dump_names.append(filename)

no_runs = len(dump_names)

run_allParams = []

run_logscore_day_k = []; k=2
means = []



for name in dump_names:
    with open(name,'r') as f: 
        dump = f.read()
    logs = eval( dump[ dump.rfind('=')+1: ] )
    logs = logs[0]

    allParams = []
    allLogscores = []
    for line in logs:
        allParams.append( line[5] )

        day = line[0]
        if day==k:
            allLogscores.append( line[3] )
    allParams = np.array(allParams)
    run_allParams.append(  allParams ) 

    run_logscore_day_k.append( allLogscores )

    means.append( np.mean(allParams,axis=0) )

    cut_allParams = allParams[burn_in:cut_off,:]
    #print 'mean:',
    print np.round(np.mean(cut_allParams,axis=0),2)
    print 'std:',np.round( np.std(cut_allParams,axis=0), 2)
    print 'no_samples:',cut_allParams.shape[0]

    print name[-30:-20]
    print 'log,L2: ', np.round(logs[-1][3:5])
    print '--------\n'


run_length = len( run_allParams[0] )
run_allParams = np.array( [run[ burn_in:cut_off ] for run in run_allParams] )
#run_logscore_day_k = [run[ burn_in: ] for run in run_logscore_day_k]

flat_run_allParams = np.array( [line for run in run_allParams for line in run] )

final_ar = np.array([run[-1] for run in run_allParams])
print 'final samples: \n'
for el in final_ar:
    print np.round(el,2)

print 'mean of final samples:',np.round(np.mean(final_ar,axis=0),2)
print 'std of final samples:',np.round(np.std(final_ar,axis=0),2)
print 'stderr of final samples:',np.round(scipy.stats.sem(final_ar,axis=0),2)

# fig,ax = plt.subplots(4, 2, figsize=(16,12))
# for i in range(4):

#     for count,run in enumerate(run_allParams[:2]):
#         ax[i,0].hist(run[:,i],bins=20,alpha=.6, label='Run %i (N=%i)'%(count,len(run[:,i])))

#         ax[i,0].set_title('Param %i'%i)
#         ax[i,0].legend()
#         #ax[i,0].set_xticklabels(visible=True)
        
#     assert len(flat_run_allParams[:,i]) == no_runs * (cut_off - burn_in)
#     ax[i,1].hist(flat_run_allParams[:,i], alpha=.6, bins=30)
#     ax[i,1].set_title('Param %i, all runs'%i)
#     #ax[i,1].set_xticklabels(visible=True)

# for j in range(2):
#     ax[0,j].set_title('Prior: Gamma(0.5,.05), Runs: %i, Samples per run: %i.'%(no_runs,cut_off))
# fig.tight_layout()

# print '\n Mean collapsing samples from all runs:',np.round( np.mean( flat_run_allParams, axis=0), 2)
# print '\n std of collapsed samples',np.round( np.std( flat_run_allParams, axis=0), 2)
# print '\n Total samples',flat_run_allParams.shape[0]

# print 'mean logscores (day %i): '%k, map(np.mean, run_logscore_day_k)
# print '\n best logscores: ', best_day_logscore



# plt.show()




