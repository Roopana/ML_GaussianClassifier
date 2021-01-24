# MultiVariate Gaussian Classifier
Implementation of Multi Variate Gaussian Classifier on Boston50, Boston75 and Digits datasets

## Objectives
Develop two parametric classifiers by modeling each class’s conditional distribution p(x|Ci) as multivariate Gaussians with 
- (a) full covariance matrix Σi and 
- (b) diagonal covariance matrix Σi

The classification will be done based on the following discriminant function:
gi(x) = log p(Ci) + log p(x|Ci) 

where  p(Ci) is maximum likelihood estimate of the class prior probabilities,  
p(x|C ) is the class conditional probabilities p(x|C ) based on the maximum likelihood estimates of the mean μˆ and the (full/diagonal) covariance Σi for each class Ci. 

Compare the performance of three models on three datasets: Boston50, Boston75, and Digits
- MultiGaussClassify with full class covariance matrices
- MultiGaussClassify with diagonal covariance matrices, and 
- LogisticRegression
