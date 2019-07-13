# Support-Vector-Machine-Implementation

### Overview
We train SVMs with polynomial and RBF kernels and experiment with hard and
soft margin SVMs. We use a simplified form of the Sequential
Minimization Optimization (SMO) algorithm (John Platt, 1998) to train SVMs in their
dual form. We perform one-vs-one and one-vs-rest classification on the MNIST dataset.
We also perform PCA on the data and compare the results of SVM after doing PCA vs
SVM on the original data.

#### To run: 

`python3 svm.py`

Optional arguments:

```
  -h, --help            show this help message and exit
  --max_passes MAX_PASSES max times to iterate over α's without changing
                        Default: 2
  --do1v1               One vs One model Default: False
  --hard                SVM hard model Default: False
  --pca                 Do PCA before SVM Default: False
  --digs1v1_first DIGS1V1_FIRST [DIGS1V1_FIRST ...]
                        first set of digits for one vs one
  --digs1v1_second DIGS1V1_SECOND [DIGS1V1_SECOND ...]
                        second set of digits for one vs one
  --onevsrest ONEVSREST
                        Digit for one vs rest classification Default: 0
  --num_train NUM_TRAIN
                        Number of training samples Default: 1000
  --num_test NUM_TEST   Number of testing samples Default: 10000
  --kernel KERNEL       Kernel Type (rbf/poly/dot) Default: rbf
  --poly_constant POLY_CONSTANT
                        constant for polynomial kernel Default: 1.0
  --poly_degree POLY_DEGREE
                        degree of polynomial kernel Default: 2
  --rbf_sigma RBF_SIGMA
                        sigma for rbf kernel Default: 5.0
  --C C                 Regularization Parameter Default: 5.0
  --tol TOL             Numerical Tolerance Default: 0.1
```

For implementation details and results, refer to `Report.pdf`.
