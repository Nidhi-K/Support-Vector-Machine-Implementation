import scipy.io
import numpy as np
import random
import argparse
from sklearn.decomposition import PCA

############################# Parsing Args ####################################
def _parse_args():
  parser = argparse.ArgumentParser(description='svm.py')

  parser.add_argument('--max_passes', type=int, default=2, \
                      help='max times to iterate over α\'s without changing \
                      Default: 2')

  parser.add_argument('--do1v1', dest='do_one_vs_one', default=False, \
                       action='store_true', help='One vs One model \
                       Default: False')

  parser.add_argument('--hard', dest='hard_margin', default=False, \
                       action='store_true', help='SVM hard model \
                       Default: False')

  parser.add_argument('--pca', dest='pca', default=False, \
                       action='store_true', help='Do PCA before SVM \
                       Default: False')

  parser.add_argument('--digs1v1_first', nargs='+', type = int, default=0, \
                      help='first set of digits for one vs one')

  parser.add_argument('--digs1v1_second', nargs='+', type = int, default=0, \
                      help='second set of digits for one vs one')

  parser.add_argument('--onevsrest', type=int, default=0, \
                      help='Digit for one vs rest classification \
                      Default: 0')

  parser.add_argument('--num_train', type=int, default=1000, \
                      help='Number of training samples \
                      Default: 1000')

  parser.add_argument('--num_test', type=int, default=10000, \
                      help='Number of testing samples \
                      Default: 10000')

  parser.add_argument('--kernel', type=str, default='rbf', \
                      help='Kernel Type (rbf/poly/dot) \
                      Default: rbf')
  
  parser.add_argument('--poly_constant', type=float, default=1.0, \
                      help='constant for polynomial kernel \
                      Default: 1.0')
  
  parser.add_argument('--poly_degree', type=int, default=2, \
                      help='degree of polynomial kernel \
                      Default: 2')

  parser.add_argument('--rbf_sigma', type=float, default=5.0, \
                      help='sigma for rbf kernel \
                      Default: 5.0')

  parser.add_argument('--C', type=float, default=5.0, \
                      help='Regularization Parameter \
                      Default: 5.0')

  parser.add_argument('--tol', type=float, default=0.1, \
                      help='Numerical Tolerance \
                      Default: 0.1')

  args = parser.parse_args()
  return args
############################# Parsing Args ####################################

############################# Hyperparameters #################################
args = _parse_args()
max_passes = args.max_passes # max times to iterate over α's without changing
num = args.onevsrest # Classify each digit as this no. or not this no. 
set_num_train = args.num_train 
set_num_test = args.num_test

set1 = args.digs1v1_first
set2 = args.digs1v1_second

kernel_type = args.kernel

C = np.inf if args.hard_margin else args.C # Regularization Parameter
tol = args.tol # Numerical tolerance 
############################# Hyperparameters #################################

############################ Kernel Functions #################################
def kernel_dot(x_i, x_j):
  return x_i.dot(x_j)

def kernel_poly(x_i, x_j, c = args.poly_constant, p = args.poly_degree):
  return (x_i.dot(x_j) + c)**p

def kernel_rbf(x_i, x_j, sigma = args.rbf_sigma):
  return np.exp((-(np.linalg.norm(x_i - x_j)**2))/(2*(sigma**2)))

def kernel (x_i, x_j):
  if kernel_type == 'dot':
    return kernel_dot(x_i, x_j)
  if kernel_type == 'poly':
    return kernel_poly(x_i, x_j)
  if kernel_type == 'rbf':
    return kernel_rbf(x_i, x_j)
################################ Functions ####################################

############################### Loading Data ##################################
mat = scipy.io.loadmat('digits.mat')
num_train = 60000; num_test = 10000

train_label = mat['trainLabels']; test_label = mat['testLabels']
train_set = mat['trainImages']; test_set = mat['testImages']
train_set = train_set.reshape(-1, num_train)
train_set = train_set.T # 60k x 784
test_set = test_set.reshape(-1, num_test) 
test_set = test_set.T # 10k x 784  
train_label = train_label.reshape(num_train) # 60k
test_label = test_label.reshape(num_test) # 10k
train_set = train_set/255.0; test_set = test_set/255.0

if args.do_one_vs_one:
  train_set = [train_set[i] for i in range(num_train) if train_label[i] in set1 or train_label[i] in set2]
  train_label = [train_label[i] for i in range(num_train) if train_label[i] in set1 or train_label[i] in set2]
  num_train = len(train_set)
  y_train = [1 if train_label[i] in set1 else -1 for i in range(num_train)]

  indices = [i for i in range(num_train)]
  random.shuffle(indices) 
  train_set = [train_set[i] for i in indices]
  y_train = [y_train[i] for i in indices]

  test_set = [test_set[i] for i in range(num_test) if test_label[i] in set1 or test_label[i] in set2]
  test_label = [test_label[i] for i in range(num_test) if test_label[i] in set1 or test_label[i] in set2]
  num_test = len(test_set)
  y_test = [1 if test_label[i] in set1 else -1 for i in range(num_test)]

  indices = [i for i in range(num_test)]
  random.shuffle(indices)
  test_set = [test_set[i] for i in indices]
  y_test = [y_test[i] for i in indices]

else:
  y_train = [1 if train_label[i] == num else -1 for i in range(num_train)]
  y_test = [1 if test_label[i] == num else -1 for i in range(num_test)]

if args.pca:
  pca = PCA(n_components=200)
  train_set = pca.fit_transform(train_set)
  test_set = pca.transform(test_set)

train_set = np.array(train_set); test_set = np.array(test_set)
y_train = np.array(y_train); y_test = np.array(y_test)

num_train = set_num_train; num_test = set_num_test
train_set = train_set[:num_train]; y_train = y_train[:num_train]
test_set = test_set[:num_test]; y_test = y_test[:num_test]

if num_train > train_set.shape[0]: num_train = train_set.shape[0]
if num_test > test_set.shape[0]: num_test = test_set.shape[0]

############################### Loading Data ##################################

############################## Training ######################################
alpha = np.zeros(num_train) # Lagrange Multipliers for solution
b = 0 # intercept b in SVM
passes = 0
tot_passes = 0
while passes < max_passes:
  num_changed_alphas = 0
  for i in range(num_train):
    x_i = train_set[i]
    y_i = y_train[i]
    fx_i = np.sum([0 if alpha[ii] == 0 else alpha[ii] * y_train[ii] * \
                   kernel(train_set[ii],x_i) \
                   for ii in range(num_train)]) + b
    E_i = fx_i - y_i
    if (y_i*E_i < -tol and alpha[i] < C) or (y_i*E_i > tol and alpha[i] > 0):
      j = random.choice(list(range(0,i)) + list(range(i+1,num_train)))
      x_j = train_set[j]
      y_j = y_train[j]
      fx_j = np.sum([0 if alpha[jj] == 0 else alpha[jj] * y_train[jj] * \
                     kernel(train_set[jj],x_j) \
                     for jj in range(num_train)]) + b
      
      E_j = fx_j - y_j
      old_alpha_i = alpha[i]; old_alpha_j = alpha[j]
 
      if y_i != y_j:
        L = max(0, alpha[j] - alpha[i])
        H = min(C, C + alpha[j] - alpha[i]) 
      elif y_i == y_j:
        L = max(0, alpha[i] + alpha[j] - C)
        H = min(C, alpha[i] + alpha[j])

      if L == H: continue
      
      eta = 2 * kernel(x_i, x_j) - kernel(x_i, x_i) - kernel(x_j, x_j)

      if eta >= 0: continue
       
      alpha[j] = alpha[j] - (y_j*(E_i - E_j))/eta
       
      if alpha[j] > H: alpha[j] = H
      if alpha[j] < L: alpha[j] = L

      if abs(alpha[j] - old_alpha_j) < 1e-5: continue
      
      alpha[i] = alpha[i] + y_i*y_j*(old_alpha_j - alpha[j])
       
      b1 = b - E_i - y_i*(alpha[i] - old_alpha_i)*kernel(x_i, x_i) \
           - y_j*(alpha[j] - old_alpha_j)*kernel(x_i, x_j)
       
      b2 = b - E_j - y_i*(alpha[i] - old_alpha_i)*kernel(x_i, x_j) \
           - y_j*(alpha[j] - old_alpha_j)*kernel(x_j, x_j)

      if 0 < alpha[i] and alpha[i] < C: 
        b = b1
      elif 0 < alpha[j] and alpha[j] < C: 
        b = b2
      else:
        b = (b1 + b2)/2.0

      num_changed_alphas = num_changed_alphas + 1
  print("a pass is done")
  tot_passes += 1
  if num_changed_alphas == 0:  
    passes = passes + 1
  else:
    passes = 0          
############################### Training ######################################

#################### Saving and Loading variables #############################
#print("Saving variables")
#np.save('alpha.npy', alpha)
#np.save('b.npy', b)

#try:
#  alpha = np.load('alpha.npy')
#except:
#  alpha = alpha = np.zeros(num_train)
#
#try:
#  b = np.load('b.npy')
#except:
#  b = 0
#

#alpha = np.load('alpha.npy')
#b = np.load('b.npy')
#################### Saving and Loading variables #############################

############################### Testing #######################################
print("Testing")
# Testing
num_correct = 0
confused_with = np.zeros(10)
num_positive = 0; num_negative = 0
num_positive_correct = 0; num_negative_correct = 0 
for i in range(num_test):
  x = test_set[i]
  f_predict = np.sum([0 if alpha[ii] == 0 else alpha[ii] * y_train[ii] * \
                      kernel(train_set[ii],x) \
                      for ii in range(num_train)]) + b
  y_predict = 1 if f_predict > 0 else -1  

  if (y_predict == y_test[i]):
    num_correct += 1
    if (y_predict == 1):
      num_positive_correct += 1
    else:
      num_negative_correct += 1
  elif y_test[i] != 1:
    confused_with[test_label[i]]+=1

  if y_test[i] == 1:
    num_positive += 1 
  else:
    num_negative += 1

acc = num_correct/num_test
############################### Testing #######################################

####################### Printing Hyperparameters ##############################
print("Hyperparameters:")
print("num_train_samples: ", num_train, sep = '')
print("C: ", C, sep='')
print("tol: ",tol, sep='')

if kernel_type == 'dot':
  print("Kernel Type: dot")
elif kernel_type == 'poly':
  print("Kernel Type: Polynomial")
  print("degree: ", args.poly_degree)
  print("constant: ", args.poly_constant)
elif kernel_type == 'rbf':
  print("Kernel Type: RBF")
  print("sigma: ", args.rbf_sigma)

if args.pca: print ("PCA")
print("Number of passes to converge:",tot_passes)
####################### Printing Hyperparameters ##############################

########################## Printing Results ###################################
print("Accuracy: ", acc*100, "%", sep ='')

if args.do_one_vs_one:
  print("One Vs One")
  print("set1:", set1)
  print("set2:", set2)
  print("True Positive %: ", \
        "{0:.2f}".format((num_positive_correct/num_positive)*100), sep = '')
else:
  print("one_vs_rest: ", num, sep = '')
  print("Percent correctly predicted as ", num,": ", \
        "{0:.2f}".format((num_positive_correct/num_positive)*100), sep = '')

  print("Predicted these numbers as ", num, sep = '')

  test_buckets = np.zeros(10)

  for i in range(set_num_test):
    test_buckets[test_label[i]]+= 1

  print("Num: percent")
  for i in range(10):
    percent = (confused_with[i]/(test_buckets[i]*1.0))*100
    print("{0:.2f}".format(percent), sep = '')
########################## Printing Results ###################################

if args.do_one_vs_one:
  train_buckets = np.zeros(2)
  test_buckets = np.zeros(2)

  for i in range(num_train):
    if y_train[i] == 1:
      train_buckets[0]+= 1
    else:
      train_buckets[1] += 1

  for i in range(num_test):
    if y_test[i] == 1:
      test_buckets[0]+= 1
    else:
      test_buckets[1] += 1

  print("Train:")
  print("set1: ", train_buckets[0])
  print("set2: ", train_buckets[1])

  print("Test:")
  print("set1: ", test_buckets[0])
  print("set2: ", test_buckets[1])

#else:
  #train_buckets = np.zeros(10)
  #test_buckets = np.zeros(10)

  #for i in range(num_train):
  #  train_buckets[train_label[i]]+= 1

  #for i in range(num_test):
  #  test_buckets[test_label[i]]+= 1

  #print("Train:")
  #for i in range(10):
  #  print(i,": ", train_buckets[i])

  #print("Test:")
  #for i in range(10):
  #  print(i,": ", test_buckets[i])
