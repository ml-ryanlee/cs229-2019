import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv
import util

#where to put these functions? We might want to call them later if we have a clf object?
def sigmoid(z):
        """ return the output of the sigmoid function
        Args:
            z: vector input. Shape (m,).

        Returns: 
            sigmoid(z): vector output. Shame (m,).
        """
        return 1/(1+np.exp(-z)) #exp is a vectorized. if z is large, exp(-z) -> inf
    
def gradient(X,y,theta):
        """ return the gradient of the loss of logistic regression, given an X,y,and theta
        Args:
            X: matrix input. Shape (m,n).
            y: vector input. Shape (m,)
            theta: vector input. Shape (m,)

        Returns: 
            gradient: gradient given X,y,theta. Shape (n,)
        """
        m = len(X)
        #vectorized gradient (nx1 size)
        return -(1/m)*(X.T)@(y-sigmoid(X@theta)) #@ is the matrix multiplication operator

def hessian(X,theta):
        m = len(X)
        z = X@theta
        #vectorized hessian (nxn size)
        return (1/m)*(sigmoid(z.T)@(1-sigmoid(z)))*X.T@X

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
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_eval) 
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
        theta = self.theta
        convergence_flag = False

        #PSEUDOCODE: LEARN THE WEIGHTS or THETAS BASED ON X, Y. 
        for iter in range(self.max_iter):        # until maximum iteration step, solve for thetas
            # 1. calculate the gradient given theta
            gradient = gradient(X,y,theta)

            # 2. calculate the Hessian
            hessian = hessian(X,theta)

            # 3. update theta's with Newton's method theta -   
            theta_new = theta - inv(hessian)@gradient
            
            # 4. if the L1 norm of difference new theta and old theta is < eps, break loop
            if norm((theta-theta_new),1) < self.eps: convergence_flag = True; break
            
            # 5. continue to update theta and loop until convergenec
            theta = theta_new

        # 6. after solving for thetas, store thetas in self.theta, if convergence has been reached.
        if convergence_flag:
             self.theta = theta
             print('convergence reached !!! theta is: ',self.theta)
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
        y_predict = sigmoid(z)

        # 3. return the y prediction (mx1)
        return y_predict
        # *** END CODE HERE ***

    