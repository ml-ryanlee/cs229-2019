import numpy as np
import matplotlib.pyplot as plt
import util

from linear_model import LinearModel
from numpy.linalg import norm

def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    
    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    clf = PoissonRegression(step_size=lr, max_iter=50000) #150
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_eval)
    print('predictions: ', y_pred)

    plt.figure()
    plt.plot(y_eval, y_pred,'o')
    plt.xlabel('true counts')
    plt.ylabel('predict counts')
    plt.savefig('output/p03.png')
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        batch_size = 32

        # initialize theta
        if self.theta == None: self.theta = np.zeros(n)
        
        # batch gradient descent for GLM (converges right away)
        # for iter in range(self.max_iter):
        #     theta_prev = np.copy(self.theta)
        #     self.theta += self.step_size*x.T@(y-np.exp(x@self.theta))/m
            
        #     if norm((self.theta-theta_prev),ord=1)<self.eps: 
        #         print('convergence loop number: ', iter)
        #         print('batch descent delta: ', self.step_size*x.T@(y-np.exp(x@self.theta))/m)
        #         break

        # modified stochastic GD (force stochastic gd through entire dataset, checks between runs through entire set)
        for iter in range(self.max_iter):
            theta_prev = np.copy(self.theta)
            for i in range(len(x)):
                self.theta += self.step_size*(y[i]-np.exp(self.theta@x[i,:]))*x[i,:]
                assert((y[i]-np.exp(self.theta@x[i,:])).shape == ())
                assert(x[i,:].shape == (n,))
            if norm((self.theta-theta_prev),ord=1)<self.eps:
                print('convergence iteration: ',iter)
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x@self.theta)
        # *** END CODE HERE ***