import numpy as np
import util

from linear_model import LinearModel
from numpy.linalg import inv
from numpy.linalg import norm


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path,add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path,add_intercept=True)
    
    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train,y_train)
    clf.predict(x_eval)
    util.plot(x_eval,y_eval,clf.theta,pred_path)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # 0. initialize theta to (n,) 1-D zero array if 'None'
        if self.theta == None: self.theta = np.zeros(x.shape[1])
    
        # iterate until convergence
        for iter in range(self.max_iter):
            m,n = x.shape
            z = x@self.theta
            h_x = 1/(1+np.exp(-z))

            # 1. define gradient
            grad = -1/m*(x.T@(y-h_x))
            assert(grad.shape == (n,))

            # 2. define hessian (use of broadcasting/elementwise operations)
            hess = (x.T * (h_x*(1-h_x)))@x/m
            assert(hess.shape == (n,n))

            # 3. define Newton's method update
            theta_prev = np.copy(self.theta)
            self.theta -= inv(hess)@(grad)

            # 4. define convergence criteria based on 1-norm
            if norm((theta_prev-self.theta),ord=1)<self.eps:
                break

    # *** END CODE HERE ***
            
    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        z = x@self.theta
        return 1/(1+np.exp(-z))
        # *** END CODE HERE ***
