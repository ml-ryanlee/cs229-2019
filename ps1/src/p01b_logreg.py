import numpy as np
import sys
from numpy.linalg import norm
from numpy.linalg import inv
import util


def sigmoid(z):
     return 1/(1+np.exp(-z)) #sigmoid(z): vector output. Shape (m,).

# (nx1) gradient definition
def gradient(X,y,theta):
    m = len(X)
    z = X@theta
    return -(1/m)*(X.T)@(y-sigmoid(X@theta)) 
        
# (nxn) hessian definition
def hessian(X,theta):
    m = len(X)
    z = X@theta
    hessian_scalar = ((1/m)*(sigmoid(z.T)))@(1-sigmoid(z))
    return np.multiply(hessian_scalar, (X.T@X))

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    X_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to pred_path

    X_eval, y_eval = util.load_dataset(eval_path, add_intercept = True)
    clf = LogisticRegression()
    print('1. shape of X_eval ',X_eval.shape)
    print('2. shape of y_eval ',y_eval.shape)
    clf.fit(X_train,y_train)
    print('3. shape of X_eval ',X_eval.shape)
    print('4. shape of y_eval ',y_eval.shape)
    y_predict = clf.predict(X_eval)
    print('5. shape of y_predict: ', y_predict.shape)
    print('size of clf.theta ', clf.theta.shape)
    util.plot(X_eval, y_eval, clf.theta, pred_path)
    np.savetxt(pred_path, y_predict) #only writing out the predictions, but not the x_evals?
    
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(X_train, y_train)
        > clf.predict(X_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            X: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
       
        # define Theta, y for fit as vectors of size (n,1) and (m,1) respectively
        n = X.shape[1]
        #y = np.asmatrix(y).T #becomes a mx1 column vector.

        if self.theta == None:
            theta = np.zeros((n,1)) #None interpreted as col vector of zeros (nx1)
        elif (self.theta).shape == (n,1):
             theta = self.theta
        else:
            print('error: must specify theta as None or vector of size Nx1')
            sys.exit(1)
        
        # set a convergence flag to determine if convergence has been reached
        convergence_flag = False

        # Newton's Method
        for iter in range(self.max_iter):
            # 1. calculate the gradient given theta
            grad = gradient(X,(np.asmatrix(y).T),theta)

            # 2. calculate the Hessian
            H = hessian(X,theta)

            # 3. update theta's with Newton's method
            theta_new = theta - inv(H)@grad
            
            # 4. if the L1 norm of difference new theta and old theta is < eps, break loop
            if norm((theta-theta_new),1) < self.eps: convergence_flag = True; break
            
            # 5. continue to update theta and loop until convergenec
            theta = theta_new

        # 6. after solving for thetas, store thetas in self.theta, if convergence has been reached.
        if convergence_flag:
             self.theta = np.resize(theta,(n,))
        else:
             print('Convergence not reached, thetas have not been updated !!! theta is ', self.theta)
        return
        # *** END CODE HERE ***

    def predict(self, X):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        ## PSEUDO CODE
        
        # 1. z is the linear combination of X (mxn) and theta (nx1) a mx1 vector
        z = X@self.theta # a vector
        
        # 2. logistic regression is defined by a probability with sigmoid function and z
        y_predict = np.resize(sigmoid(z),(len(X),))
        
        #y_predict = np.reshape(y_predict, X.shape[0])
        #print('4.5 shape of y_predict AFTER resize X, y_predict.shape', y_predict.shape)
        # 3. return the y prediction (m,)
        return y_predict
        # *** END CODE HERE ***

    