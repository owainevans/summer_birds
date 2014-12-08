# CP4: P4 - HDP-LDA

*Owain Evans and Vlad Firoiu, Venture Team (12.14)*

## Overview
We implemented the metrics and Python baselines for this problem. We implemented the HDP-LDA model in Venture. We provide tests on small amounts of data that suggest that the model is able to learn topic structure. We do not provide any performance curves for our model. 


## Venture Model code

The Venture model code is found in `lda.py`.


```scheme

; We implement the DPmem higher-order function in
; terms of the primitive CRP constructor 'make_crp'

[assume dpmem (lambda (alpha f)
  (let ((crp (make_crp alpha))
        (g (mem (lambda (table) (f)))))
    (lambda () (g (crp)))))]

[assume pymem (lambda (alpha d f)
  (let ((crp (make_crp alpha d))
        (g (mem (lambda (table) (f)))))
    (lambda () (g (crp)))))]

[assume noisy_dpmem (lambda (alpha f)
  (let ((crp (make_crp alpha))
        (g (mem (lambda (table) (f)))))
    (lambda () (noisy_id 0.01 (g (crp))))))]

[assume noisy_pymem (lambda (alpha d f)
  (let ((crp (make_crp alpha d))
        (g (mem (lambda (table) (f)))))
    (lambda () (noisy_id 0.01 (g (crp))))))]


[assume topic_base_alpha (gamma 1 1)]
[assume topic_doc_alpha (gamma 1 1)]
[assume topic_word_alpha (gamma 1 1)]

; the vocabulary size is a fixed parameter that we vary
; depending on the number of distinct words in the corpus
[assume vocab_size {vocab}]

[assume topic_base_sampler (make_crp topic_base_alpha)]

[assume get_topic_doc_sampler
  (mem (lambda (doc)
    (dpmem topic_doc_alpha topic_base_sampler)))]

[assume topic_doc_sampler
  (lambda (doc)
    ((get_topic_doc_sampler doc)))]

[assume topic_position_sampler
  (mem (lambda (doc pos)
    (topic_doc_sampler doc)))]

[assume get_word_topic_sampler
  (mem (lambda (topic)
    (make_sym_dir_mult topic_word_alpha vocab_size)))]

[assume word_position_sampler
  (mem (lambda (doc pos)
    ((get_word_topic_sampler (topic_position_sampler doc pos)))))]

```


## Testing our Model

### Quality test
We provide a simple test for whether our implementation of HDP-LDA is able to do correct topic inference on very small amounts of data. The test is found in the `smoketest.py` script under the name `quality_test()`. In the test, we condition the model on four arguments of 16 words each. There are two disjoint topics and each document contains just one of these topics. Our model passes the test if (1) inference is able to improve the score of the model on both of the challenge problem metrics, and (2) if Hamming score of the model is small after inference. 

### Smoketest
Our smoketest tests our code on all elements of the challenge problem (while leaving out most of the training documents). The test loads a subset of the training documents and all the test documents. All training and partial test documents are loaded and a small number of MH transitions are performed. Our goal is to compute metrics for the model on the queries. As discussed below, we can answer the queries by generating marginal histograms over the test documents. In our model, we compute empirical histograms by sampling from the test document CRPs repeatedly. The histogram gives us the probability of each word (given the document) which we use to compute query 1, and the Hamming distance on the ground-truth document (by just scaling up the histogram to the document length). 



## Metrics

### Interpreting the Queries and Metrics
We had some confusion about the queries and metrics. The queries ask us to compute some function of the HDP-LDA model on this problem. Since the model and the data are fully specified, this is a well-defined problem. A natural metric would be to compare PPL estimates for the two queries against one or two benchmark implementations of HDP-LDA. Instead, the metrics do not involve LDA at all, and depend purely on the test documents. Especially since there are only five test documents, it's possible that optimizing for the metrics is mostly orthogonal to optimizing for improved approximation to the HDP-LDA model inference.

Our main confusion concerned Query 2. One interpretation of this query reduces to finding the word that has the highest probability given the document (marginalizing out other parameters). This conflicts with the metric for the problem. One would perform poorly on the Hamming distance by selecting only one word. Another intepretation is the document-completion that arg-maxes the following quantity: the marginal probability of the document completion (marginalizing over all uncertain parameters in the HDP-LDA). This quantity is more consisent with the metric. And we assume that appoximatiing this quantity is the aim of the query.

We have two approaches to computing this quantity via Venture. The first is a simple approximation. For any document, we can compute from our model an estimate of the word histogram for the document. (By *word histogram* we mean the distribution on words given the document. This is computed by  marginalizing out the topics). We can then computing the Hamming distance of this histogram compared to the ground-truth histogram.

A more sophisticated approach is also possible in Venture. TODO: add stuff about particle filtering.


### Our Implementation of the metrics
Python functions for computing the metrics are given in `metric_utils.py`. This script also contains additional utilities for computing the metrics given estimated marginal histograms on words for a test document. 


## Performance Curves
We do not provide performance curves for our solution. 





