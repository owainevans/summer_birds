# CP2: Birds in Venture

*Owain Evans*, *Vlad Firoiu*, and *Venture Team* (09.2014)


-------------

## Contents

####Section 0
1. Glossary
2. Introduction

####Section I: Models for Birds
3. Multinomial Model for Onebird
4. Optimization for Onebird by Pre-processing
5. Multinomial Model for Manybird
6. From Multinomial Model to Poisson Model
7. Poisson Approximation - Description
8. Poisson Approximation - Implementation

####Section 2: Implementation and Inference
9. Onebird: Implementation and inference
10. Manybird: Implementation and inference
11. Testing
12. Possible Optimizations





## Glossary
**Betas** - the four Beta Parameters that we infer in the parameter estimation task. Ground-truth values for this round are [2,1,0,1].

**Source and Destination Cells** - On any night, a bird moves from the *source* to *destination* cell, which will sometimes be the same cell (i.e. the cell with the same index).

**Onebird/Manybird and Multinomial/Poisson** - We refer to the first part of the challenge problem (associated with dataset 1) as the *Onebird* problem. We refer to the second part of the problem (associated with dataset 2) as the *Manybird* problem. Note that these terms just refer to the modeling problem and not to the Venture programs that we used to solve them. For our solutions we used a *Multinomial* model (as in the problem description) for Onebird and a *Poisson* approximation to the Multinomial for Manybird. 

**MH** - The Metropolis Hastings MCMC inference method.

**SMC** - Sequential Monte Carlo Bayesian inference. 



## Introduction
For Dataset 1, we used an exact Venture implementation of the generative model as stated in the problem. We loaded the data in batch and then did Metropolis Hastings inference to learn the parameters.

For Datasets 2-3, we implemented a different generative model (the 'Poisson' model -- see below) that approximates the true model and is more tractable. For this model, data was loaded sequentially and we did inference after each set of data was loaded. We used MH on the reconstruction and predictions problems, and MH and SMC on the parameter inference problem. 

------

## SECTION I: Models for Birds in Venture


### Overview
We first present our model and Venture program for the Onebird and explain its key components. We then generalize this model to the case of many birds. We discuss the limitations of this generalized model and motivate an optimization of it that uses a Poisson approximation to the Multinomial. 


### The Multinomial model for Onebird
Here we exhibit the Venture program for Onebird. We used Python code to pre-process data and loop over observations to condition our model on them. We also used Python to generate some parts of our Venture code that are parameterized. For expository purposes, we will discuss the Venture code first and leave the details of our use of Python till a later section. 


We first define the Beta parameters (called `hypers` in our code) and place a Gamma(1,0.1) prior on their values. We use the `scope_include` annotation to place the Betas into a particular inference scope. This allows us to specify inference instructions that will perform inference only on the Betas.

```scheme
[assume hypers_0
    (scope_include hypers  0 
     (gamma 1 0.1))]

[assume hypers_1
    (scope_include hypers  1 
     (gamma 1 0.1))]

; etc. up to hypers_3

```

The features are loaded stored as a Venture dictionary. 

```scheme
[assume features (dict
                  (array (array 0 0 0 0) (array 0 0 0 1) .... )
                  (array (array 0 0 0 2) (array .97 .70 0 0) .... ))]
```

We define the un-normalized probability of a bird moving
from cell `i` to cell `j` on day `d`.

```scheme
[assume phi
  (mem (lambda (y d i j)
    (let ((fs (lookup features (array y d i j))))
      (exp (dot_product fs hypers)))))]      
```


We define the distribution on destination cells for the bird at cell `i`. We normalize the `phi` values for the given day. (Note: the normalizing constant `sum_phi` is actually the sum of the `phi` values for each of the 16 days, and similarly for the argument to simplex.)

```scheme
[assume get_bird_move_dist
  (mem (lambda (y d i)
    (let ( (sum_phi (+ (phi y d i 0) ... (phi y d i 15))) )
      (simplex (/ (phi y d i 0) sum_phi) ... (/ (phi y d i 15) sum_phi)))))]
```

The distribution on destination cells (above) is deterministic. Apart from the Betas, the other stochastic part of the model is where the bird moves on a given day from cell `i`. We include this in a scope named `moves`, enabling us to distinguish the un-observed bird-move variables from the Betas. The variable `cell_array` is just an array of all cell indices. 

```scheme
[assume cell_array (array 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)]

[assume move
  (lambda (y d i)
    (scope_include moves  d
      (categorical (get_bird_move_dist y d i) cell_array)))]
```

In this model (in contrast to the models for datasets 2 and 3) we can usually condition directly on the move the bird makes (rather than a noisy count of the birds at a particular location). We ignore days on which the exact location is unknown. Our observations have the following form. We use Python to loop over the loaded observations, substituting in integers for the year, day and source and destination cells (having pre-processed the data to deduce the facts about movement). Here we provide a concrete example of a single `observe`:

```scheme
[observe (move 0 1 2) 1 ]
```

We perform inference by MH with 'slice sampling' proposals. We do inference only on the Betas (using the `hypers` scope). (Arguments to 'slice': 'one' refers to picking one variable from the scope at random for an MH proposal, 0.5 and 100 are slice parameters?)
```scheme
[infer (slice hypers one 0.5 100 10)]
```


### Optimization For Onebird by Pre-processing

The Onebird problem is distinct from Manybird in that we can extract (via pre-processing outside Venture) the latent movements of the bird on most of the days in the dataset. (If the bird count is zero everywhere on a given day, we cannot pin down with certainty where the bird is. Otherwise, we can). This pre-processing is what allows us to employ the simple model above, where observations specify moves exactly (i.e. without noise). It is straightforward in Venture to condition only on a (large) subset of days (rather than every day). It is also simple (via scope annotations) to ignore the latent states that remain unknown and to do inference only on the Betas.  


### Multinomial Model for Manybird Problem

For the Manybird problem, we need to instantiate the bird dynamics for multiple birds (who may start at the same source cell on a given day but move to different destination cells). We also need to be able to condition on noisy observations of the total bird count at a given cell. That is, we don't observe the movements of birds, but instead a sum of the locations of the birds at a given time-step.

We first summarize how we generalized the model above to the Manybird case. We then exhibit code for this generalized model. 

**Summary of generalized Multinomial Model**:
- We switch from a single bird to multiple birds distinguished by an integer, the `bird_id`.

- The function `move` now samples the movement of a given bird at a particular time-step. The state evolves by calling the `move` function for each bird.

- An observation of a cell at a timestep needs to count the birds at the cell (see `count_birds` below) and then add Poisson noise to them (see `observe_birds` below). Note that these new functions simply build on those present in the Onebird model. 

- Note that we do not explicitly invoke a multinomial random draw to get the number of birds moving to each possible destination cell from a source cell. However, the call to the procedure `categorical` for each bird_id computes a single-trial multinomial for each bird, which is mathematically equivalent. 


**An Instance of the Manybird Multinomial Model**
We use a Python program to generate the Venture Multinomial model for specific parameters. The key features of this code generation are as follows:

- The Python program takes as arguments the grid-size, the bird-count, the number of features and the days and years and generates appropriate Venture code.

- The instance below is a complete Venture program for Manybird with 4 birds, two features and a 2x2 grid.

- We elide the value of the `features` variable (which is a large dictionary). Note that functions like `sum_phi` and `get_bird_move_dist` have the form of unrolled loops.

- Inference programs for this program take identical form to those above because the random variables are still the Betas and the bird moves (in scopes `hypers` and `moves` respectively).



**Venture code:**

```scheme

; The Venture dictionary constructor has two arguments:
; an array of keys and an array of values. In this case, each
; key has the form (array <year> <day> <cell1> <cell2> ) and each
; value has form (array value_feature0 value_feature1).
[assume features (dict (array
                         (array 0 0 0 0)
                         (array 0 0 0 1)
                         ....
                                        )
                        (array
                          (array .5 1.2)
                          (array .6 1.2)
                          ...
                                         ) ) ]

[assume num_birds 4]

[assume hypers0 (scope_include (quote hypers) 0 (gamma 1 .1))]

[assume hypers1 (scope_include (quote hypers) 0 (gamma 1 .1))]

[assume bird_ids (list 0 1 2 3)]

[assume phi 
      (mem (lambda (y d i j)
        (let ((fs (lookup features (array y d i j))))
          (exp
            (+ (* hypers0 (lookup fs 0))
               (* hypers1 (lookup fs 1)))))))]

[assume sum_phi (mem (lambda (y d i)
                  (+ (phi y d i 0)
                     (phi y d i 1)
                     (phi y d i 2)
                     (phi y d i 3))))]

[assume get_bird_move_dist
  (mem
    (lambda (y d i)
      (simplex
        (/ (phi y d i 0) (sum_phi y d i))
        (/ (phi y d i 1) (sum_phi y d i))
        (/ (phi y d i 2) (sum_phi y d i))
        (/ (phi y d i 3) (sum_phi y d i)))))]

[assume cell_array (array 0 1 2 3)]

[assume move 
      (mem (lambda (bird_id y d i)
        (let ((dist (get_bird_move_dist y d i)))
          (scope_include (quote d) bird_id
            (categorical dist cell_array)))))]

[assume get_bird_pos 
      (mem (lambda (bird_id y d)
        (if (= d 0) 0
          (move bird_id y (- d 1) (get_bird_pos bird_id y (- d 1))))))]

[assume all_bird_pos 
       (mem (lambda (y d) 
         (map (lambda (bird_id) (get_bird_pos bird_id y d)) bird_ids)))]

[assume count_birds 
      (mem (lambda (y d i)
        (size (filter
                (lambda (x) (= x i)) (all_bird_pos y d)))))]

; This is the only new function that is stochastic. Note that we add .00001 to
; the bird_count to avoid ever giving an type-incorrect zero input to the Poisson
; sampler. (When the true count at a cell is zero, it will still be very likely
; the observation is zero). 
[assume observe_birds (mem (lambda (y d i) (poisson (+ (count_birds y d i) 0.00001))))]


[observe (observe_birds 0 0 0) 5.]
[observe (observe_birds 0 0 1) 1.]
....
; (etc. for additional observations)


```

### From the Multinomial Model to the Poisson Model
Existing work on this problem (as cited in the CP2 Documentation) makes it clear that models like the Multinomial model are ineffecient when the number of birds gets large. For the Manybird problem, we made a couple of key changes to the program above to make inference in Venture more tractable [FIXME VKM AND AXCH]. We will first describe these changes and exhibit the Venture code that implements them. After that, we discuss in detail the Poisson approximation to the Multinomial that was crucial to our inference performance. 

#### Representing the Grid
The first change from the Venture program above is to explicitly model the geometry of the grid. The dynamics in the challenge problem stipulate that birds can never move more than a fixed maximum distance. By checking the distance between cells before computing the probability of movement between them, we can avoid creating extra latent states for movements that can never happen. Venture code to implement this is as follows:

```scheme
[assume width 10]
[assume height 10]
[assume max_dist2 18]

; convert cell index to x,y value
[assume cell2X (lambda (cell) (int_div cell height))]
[assume cell2Y (lambda (cell) (int_mod cell height))]

; squared-distance
[assume dist2 
      (lambda (x1 y1 x2 y2)
        (+ (square (- x1 x2)) (square (- y1 y2))))]
    
[assume cell_dist2 
      (lambda (i j)
        (dist2
          (cell2X i) (cell2Y i)
          (cell2X j) (cell2Y j)))]

; note that we now include a distance cut-off in phi
[assume phi 
      (mem (lambda (y d i j)
        (if (> (cell_dist2 i j) max_dist2) 0
          (let ((fs (lookup features (array y d i j))))
            (exp (dot_product fs hypers) ) )))) ]
```

### The Poisson Approximation - Description
The second change is a more substantive conceptual change: we use a Poisson approximation of the Multinomial model. The Multinomial model we described above is analogous to the Multinomial model in the problem description. We used single-trial multinomial draws for each bird (representing each bird individuall). This leads to a very large number of latent variables. Hence the alternative in the problem description, which combines these single trial multinomials. We focus on this Multinomial model (i.e. the one from the problem description) in what follows.

In the Multinomial model, the probability of a single bird moving from $i$ to $j$ is a single-trial multinomial over cells $j$ reachable from $i$. Since individual bird moves are assumed to be i.i.d., the total number of birds moving from $i$ to reachable $j$ is a multi-trial multinomial: 

\begin{equation}
n_{t,t+1}(i,j) \sim Multinomial(n_{t}(i), \theta) 
\end{equation}

Here $n_{t}(i)$ is the bird-count at $i$, and $\theta$ is a probability vector that results from normalizing $\phi_{t,t+1}(i,j)$ for each reachable $j$. 

We approximate this multinomial with a set of independent Poisson distributions, one for each pair of cells $(i,j)$. The Poisson intensity parameter for each pair is given by its expectation under the corresponding multinomial. For each $(i,j)$, we have:

\begin{equation}
n_{t,t+1}(i,j) \sim Poisson( n_{t}(i) * \theta_{j} )
\end{equation}

\noindent Here $\theta_{j}$ is the $j$-th component of $\theta$. On the Poisson model, $n_{t,t+1}(i,j)$ and $n_{t,t+1}(i,k)$ are independent given $n_{t}(i)$ and the features at time $t$ for any $j \neq k$. So for any cell $i$, the sum over $n_{t,t+1}(i,j)$ for all $j$ (including the birds staying at $i$) may differ from $n_{t}(i)$. That is, the number of birds is not conserved over time. This cannot happen in the multinomial model. Lack of conservation is not a problem for the Birds task because bird-counts are observed at every time-step. Thus sequences where the total bird-count varies substantially over time are ruled out by our model as inconsistent with daily observed bird-counts. (A further issue with the Poisson model is that it will be comparatively more noisy and less accurate when the bird counts are very low, e.g. in the Onebird case). 


### The Poisson Approximation - Implementation
In the Multinomial model, we compute the probability of a bird moving from source `i` to destination `j`, and then pick `j` via the `categorical` procedure. The function below instead counts up the birds at `i`, multiplies this by the probability of moving to `j` (for each `j`) and then samples (via the `poisson` procedure) a number of birds moving from `i` to `j` for each `j`. [COULD ADD MORE DETAIL, EG SCOPE ANNOTATION]

```scheme
[assume bird_movements_loc
    (mem (lambda (y d i)
      (if (= (count_birds y d i) 0)  ;no movement from i if zero birds there  
        (lambda (j) 0)                     
          (mem (lambda (j)
            (if (= (phi y d i j) 0) 0    ;no movement from i to j if zero phi score
              (let ((n (* (count_birds y d i) (get_bird_move_prob y d i j))))
                (scope_include d (array y d i j)
                  (poisson n)))))))))]
```


-------------


## SECTION II: Implementation and Inference


## Onebird: Implementation and Inference

`model.py` (Venture program)
The Venture model is specified by the Python class `OneBird`. This class has a set of attributes that are used to generate Venture code. The model is loaded onto a Venture RIPL instance given as an argument to the class constructor. Given an instance of this Python class, it is easy to generate synthetic data from the model, or to add observations. Observes are loaded in bulk onto this ripl via a method `loadObserves`. The inference program (slice sampling) is defined by the method `inferHypers`.

`utils.py` (pre-processing)
Apart from the class `OneBird`, most of our code for this problem is either for pre-processing the input data or for running and aggregating inference. We pre-process the observations in `4x4x1-train-observations.csv` into known bird movements (which are what we condition on). 

`onebird.py` (inference)
For our inference, we do multiple independent MH runs in parallel. We take the last sample of the Betas from each run and average them. (A more principled Bayesian approach would be to compute either a serial or parallel estimate of the posterior and give the estimated MAP as the solution. Also, with a multi-modal posterior on the Betas, our averaging strategy might give bad results. However our averaging strategy does get close to the ground-truth Betas. In `onebird.py`, the function `sweep` specifies our inference program, and `onebird.runFromConditional` applies this program for a given number of iterations. We use Python's `multiprocess` module for parallelism.


----------------

## Manybird: Implementation and Inference

### Overview
We used one inference strategy for Reconstruction and Prediction and a different one for Parameter Inference. In terms of Venture inference, the important distinction between these problems is whether we have to do inference on the Betas. Since all latents probabilistically depend on the Betas, computing likelihoods for the Betas requires computing a large constant for the first day which then grows linearly in the number of days. In Reconstruction and Prediction, we avoid any inference on the hypers and just assume the ground-truth Betas as part of the model (TODO: questionable).

### Reconstruction
For the reconstruction task we implement a filtering inference program on the latents (i.e. count of birds moving from `i` to `j` on a given day). On day `d`, we observe each count for that day via the function `observe_birds` above. We then run MH only on the latents for `d` (holding fixed the values of latents for all previous days). We implement filtering by annotating the Venture program with `scope_include`. The variable for latent which corresponds to the expression `(bird_movements_loc y t i j)` in the code above, is included in a scope named `d` for the number of the day.

This inference is found in `train_birds.py`. Python code is used to loop over the data and then make calls to Venture to add observations and perform MH inference. 

```python
for y in years:
  for d in days:
   for i,n in zip(cells,observations):
     model.ripl.observe('(observe_birds %i %i %i)'%(y,d,i), '%i'%n)
     model.ripl.infer('(mh %i one %i)'%(d,number_transitions))
```
   
On each day, we condition the model on all counts for that day via the `observe` directive. The arguments to the `infer` directive specify MH proposals targeted to latents for the given day `d`. The argument `one` specifies that we uniformly pick a single latent and propose to it (rather than making a 'blocked' proposal). Finally `number_transitions` is a parameter we control for the number of MH transitions.

### Prediction
Our inference strategy for prediction is almost identical to that for reconstruction and we retain the Poisson approximation model. Full details of our implementation are in `predict_birds.py`. One difference is that after predicting the latents on a given day, we use the Venture directive `forget` to remove this sampled prediction from our model. (This prevents Venture from making spurious future inferences based on past sampled predictions.)

###Parameter Inference
It is possible to use MH to do joint inference on the Betas and latent states. On the first round of the challenge problems, we used sequential MH (as in Reconstruction) and got results close to the ground-truth Betas (when averaged over less than 10 runs). However, using MH becomes very memory- and time-intensive after only two days of observations.

For this round, we combine SMC (particle-filtering with resampling) and sequential MH. That is, we generate more than 100 particles, which we re-sample after each day of observations. We also run a comparately smaller number of MH transitions on the latent states of each particle. This provides comparable results to the pure MH solution, but with much less compute time. (Again we do inference only for the first few days). The Python code for inference is in `param_inf.py` and `train_birds_param.py`. 

### TODO: more detail on inference programs

----
## Testing
No baselines were provided to which we could compare our solutions. For example, for all problems, we do not what the scores would be for exact Bayesian inference using the same priors as us. We assume, however, that correct values for the Betas would be recovered with high confidence by Bayesian inference. If this is so, then our performance on the parameter inference tasks can form part of a test of the soundness of our approach and the integrity of our code.

In both Onebird and Manybird, we got values for the parameters close to the ground-truth. In both cases we started in prior values with reasonable variance, and then improved our estimates as we sequentially added observations and inference. (In both cases, there is variance in our estimates over repeated independent runs. It is unclear how much this is Monte Carlo noise in our inference techniques vs. variation in the Bayesian posterior).

In the Manybird case, parameter estimation requires some of amount of inference on the latents. Given that this inference was successful, we have increased confidence that the same inference techniques applied to the latents only (given the correct values for the Betas) will be reasonably successful. Again, it is not clear to us how much uncertainty there would be in the true Bayesian posterior on reconstruction and prediction. Moreover, our Poisson approximation means that we would not be able to recover the exact latents states, even if they were recovarable in principle. 

## Possible Optimizations


possible optimizations:
subsumes performance curves. phi procedure. fixme for performance curves section. 








### Performance Curves
No performance curves yet. We could vary how much MH/Slice we do, we could add SMC, we could vary number of parallel runs. Time is currently measured by run_params.py. This will not be especially accurate but is probably fine for comparing MH/SLICE, say. Previous results showed lots of uncertainty. Given sparsity of the data, this may be the true Bayesian outcome given non-trivial prior uncertainty. We could try to get a better sense of this by doing a couple of longer runs and seeing how much they converge. (Note: uncertainty should be lessened by doing inference on all the days -- not leaving any out. also doing joint inference on all the years, rather than doing them in parallel). 


3. Performance: parameter estimation is close to true values. Doing well here depends on inferring the latent states, because those are not given as part of the data. If a relatively small amount of MH can infer the latent states well, we expect MH to do well on the reconstruction and prediction tasks, where we don't need to do inference on the Betas. 

















to generate observations, we map *move* across all birds, sum
the birds at each cell (via *count_birds*) and add Poisson noise

Observations at a cell add Poisson node to the bird-count. We add 0.0001 to the bird count to avoid ever passing the Poisson zero parameter value. The function `count_birds` counts the birds at a cell (always 0 or 1 in this case).
```[assume observe\_birds (lambda (y d i) (poisson (+ (count_birds y d i) 0.0001)]




