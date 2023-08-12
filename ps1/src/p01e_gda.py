import numpy as np
import util
import sys
from scipy import stats
from numpy.linalg import inv 
from numpy.linalg import norm

def sigmoid(z):
     return 1/(1+np.exp(-z)) #sigmoid(z): vector output. Shape (m,).

def binom_normality(X,y):
    X1 = X[y==1]
    X0 = X[y==0]
    resX1 = stats.normaltest(X1)
    resX0 = stats.normaltest(X0)
    print('GDA p-value for normality test of X|Y=0 is: ', resX1.pvalue)
    print('GDA p-value for normality test of X|Y=1 is: ', resX0.pvalue)
    return

def transform_X(X_in,dim):
    X_new = np.copy(X_in)
    # remove negative examples before taking log
    # print('dim_value: ',dim,(X_new[:,dim]))
    # print(np.log(X_new[:,dim]))
    X_new[:,dim] = np.log(X_new[:,dim])
    return X_new

def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    X_train, y_train = util.load_dataset(train_path, add_intercept=False)
    binom_normality(X_train,y_train)
    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to pred_path
    
    X_eval, y_eval = util.load_dataset(eval_path,add_intercept = True)
    clf = GDA()
    
    # for problem 1h: transform training and evaluation data to normalize:
    # X_train_mod = transform_X(np.copy(X_train),1)
    # X_eval_mod = transform_X(np.copy(X_eval),2) # why 2? Remember, we add offset col after fitting theta0 with training
    # clf.fit(X_train_mod,y_train)
    # y_predict = clf.predict(X_eval_mod)

    # fit the model and predict
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_eval)

    print('GDA prediction accuracy:', sum(y_predict == y_eval)/len(y_eval))

    #plotting and saving outputs
    util.plot(X_eval, y_eval, clf.theta, pred_path)
    np.savetxt(pred_path, y_predict) 

    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, X, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Notes:
        # 1. define MLE estimators for ∑, µ_0, µ_1, phi
        # 2. define theta0 and theta1 with MLE estimators
        
        # define shape of X input
        m, n = X.shape

        # reshape y for matrix multiplication
        if self.theta == None:
            theta = np.zeros((n,1)) #None interpreted as col vector of zeros (nx1)
        elif (self.theta).shape == (n,1):
             theta = self.theta
        else:
            print('error: must specify theta as None or vector of size nx1')
            sys.exit(1)

        # define a vector for 1{y=1} and 1{y=0} that is mx1
        y1 = np.asmatrix(y).T
        y0 = np.asmatrix(np.logical_not(y).astype(int)).T
       
        # MLE estimators
        phi = sum(y)/m # scalar
        mu0 = 1/sum(y)*X.T@y0 # nx1
        mu1 = 1/sum(y)*X.T@y1 # nx1
    
        # construct sigma estimator
        sigma = np.zeros((n,n))
        for i in range(m):
            x_i = np.asmatrix(X[i,:]).T
            y_i = y[i]
            # a. take the expression for sigma for 1 example
            sig_i = (x_i-((1-y_i)*mu0)-(y_i*mu1))@(x_i-((1-y_i)*mu0)-(y_i*mu1)).T 
            # b. sum the sigma estimators over all examples           
            sigma += sig_i
        # c. divide the total by 1/m examples
        sigma = 1/m*sigma

        # define theta and theta0 based on phi, mu0, mu1, and sigma, estimated with training data
        theta = -1*inv(sigma)@((mu0-mu1)) # nx1
        theta0 = 1/2*((mu0.T@inv(sigma)@mu0)-(mu1.T@inv(sigma)@mu1))-np.log((1-phi)/phi) #scalar 1x1

        # concatenate theta0 to first index of theta, then save it to self.theta
        theta_total = np.insert(theta,0,theta0,axis=0)
        self.theta = np.resize(theta_total,((n+1),)) #np.resize(theta,(n,))
        # *** END CODE HERE ***
    
    def predict(self, X):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # 1. z is the linear combination of X (mxn) and theta (nx1) a mx1 vector
        z = X@self.theta # a vector
        
        # 2. logistic regression is defined by a probability with sigmoid function and z
        y_prob = np.resize(sigmoid(z),(len(X),))
        y_predict = (y_prob>0.5).astype(int)

        # 3. return the y prediction (m,)
        return y_predict
        # *** END CODE HERE
