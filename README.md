# MultiVariate Gaussian Classifier
Implementation of Multi Variate Gaussian Classifier on Boston50, Boston75 and Digits datasets

## Objectives
Develop two parametric classifiers by modeling each class’s conditional distribution p(x|Ci) as multivariate Gaussians with 
- (a) full covariance matrix Σi and 
- (b) diagonal covariance matrix Σi

The classification will be done based on the following discriminant function:

__gi(x) = log p(Ci) + log p(x|Ci)__

where  _p(Ci)_ is maximum likelihood estimate of the class prior probabilities, \
_p(x|Ci )_ is maximum likelihood estimate of the class conditional probabilities \
based on the maximum likelihood estimates of the mean _μiˆ_ and the (full/diagonal) covariance _Σi_ for each class _Ci_. 

Compare the performance of below three models on the datasets: Boston50, Boston75, and Digits
- MultiGaussClassify with full class covariance matrices
- MultiGaussClassify with diagonal covariance matrices, and 
- LogisticRegression
