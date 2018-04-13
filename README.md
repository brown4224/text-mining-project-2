# Classifier-based Cranfield search

### What is this?
This is a very simple ranking implementation using classifier for the cranfield dataset.


### What should be the goal for this project?

Update the 1)  `create_model` file, 2) make sure you implement `evaluate_model` and`load_models` functions as required  such that you can increase the nDCG score.



### What is the cranfield dataset?
Cranfield is a small curated dataset that is very extensively used in the information retrieval experiments.
In the dataset, there are 226 queries (search terms), 1400 documents, and 1837 (evaluations).
The dataset is supposed to be complete in the sense that the documents that should be returned for each known are known.
This makes the evaluation easier. [Click here more details](http://ir.dcs.gla.ac.uk/resources/test_collections/cran/)



### What is nDCG score?
[nDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) is a very common metric used in search evaluations. 
Higher nDCG score (close to 1.0 ) describes a search system that gives all the relevant results with most relevant ones on the top.

### What is the current nDCG score?
`Final ncdg for all queries is 0.197136293042`


### What are the current classification metrics?

       label        precision    recall  f1-score   support

          1           0.24        0.17      0.20       122
          2           0.25        0.23      0.24       120
          3           0.39        0.50      0.44       246
          4           0.17        0.13      0.15       119

    avg / total       0.29        0.31      0.30       607


### What I should not do?

Please do not modify `search.py` or `readers.py` or `eval.py` 