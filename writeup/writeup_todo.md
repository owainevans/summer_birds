
1. References for filtering / batch.
2. Some results
3. make proper pdf from images




## Possible Optimizations
possible optimizations:
subsumes performance curves. phi procedure. fixme for performance curves section. 


### Performance Curves
No performance curves yet. We could vary how much MH/Slice we do, we could add SMC, we could vary number of parallel runs. Time is currently measured by run_params.py. This will not be especially accurate but is probably fine for comparing MH/SLICE, say. Previous results showed lots of uncertainty. Given sparsity of the data, this may be the true Bayesian outcome given non-trivial prior uncertainty. We could try to get a better sense of this by doing a couple of longer runs and seeing how much they converge. (Note: uncertainty should be lessened by doing inference on all the days -- not leaving any out. also doing joint inference on all the years, rather than doing them in parallel). 


3. Performance: parameter estimation is close to true values. Doing well here depends on inferring the latent states, because those are not given as part of the data. If a relatively small amount of MH can infer the latent states well, we expect MH to do well on the reconstruction and prediction tasks, where we don't need to do inference on the Betas. 

