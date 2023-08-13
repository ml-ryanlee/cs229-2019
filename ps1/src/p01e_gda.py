import numpy as np
import util

from linear_model import LinearModel
from scipy import stats
from numpy.linalg import inv 
from numpy.linalg import norm

def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval,y_eval = util.load_dataset(eval_path,add_intercept = True)
   
    # *** START CODE HERE ***
    clf = GDA()
    clf.fit(x_train,y_train)
    predict = clf.predict(x_eval)
    binom_normality(x_eval,y_eval)
    print('validation GDA accuracy: ', accuracy(predict,y_eval))
    util.plot(x_eval, y_eval, clf.theta, pred_path)

    # for part g) transform the second feature of x by taking log(x2)
    x_train_tf = np.copy(x_train)
    x_eval_tf = np.copy(x_eval)

    x_train_tf[:,1] = np.log(x_train_tf[:,1])
    x_eval_tf[:,2] = np.log(x_eval_tf[:,2])

    clf_tf = GDA()
    clf_tf.fit(x_train_tf,y_train)
    predict_tf = clf_tf.predict(x_eval_tf)
    binom_normality(x_eval_tf,y_eval)
    print('validation GDA accuracy, transformed DS: ', accuracy(predict_tf,y_eval))
    util.plot(x_eval_tf, y_eval, clf_tf.theta, pred_path)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # 0. initialize theta to ((n+1),) 1-D zero array if 'None'
        if self.theta == None: self.theta = np.zeros(x.shape[1]+1)

        # Gaussian MLE, derived from gradient of log-likelihood(X|Y=1)
        # log(product(x's*p(x)'s) where x is normal pdf, p(x) = (1-phi)^y*phi^y
        m,n = x.shape
        y0  = 1-y 
        assert(y0.shape == (m,))
        
        phi = sum(y)/m
        assert(phi.shape == ())

        mu0 = sum(x[y==0],0)/sum(y0)
        assert(mu0.shape == (n,))
        
        mu1 = sum(x[y==1],0)/sum(y)
        assert(mu1.shape == (n,))
        
        y0mu0 = y0.reshape(m,1) * mu0.reshape(1,n)
        y1mu1 = y.reshape(m,1) * mu1.reshape(1,n)
        sigma = (x-y0mu0-y1mu1).T@(x-y0mu0-y1mu1)/m
        assert(y0mu0.shape == (m,n))
        assert(y1mu1.shape == (m,n))
        assert(sigma.shape == (n,n))

        # defining sigmoid theta in terms of MLE (derived with Bayes Theorem)
        theta0 = 1/2*((mu0.reshape(1,n) @ inv(sigma) @ mu0.reshape(n,1)) \
                - (mu1.reshape(1,n) @ inv(sigma) @ mu1.reshape(n,1))) \
                - np.log((1-phi)/phi)
        theta = -inv(sigma)@(mu0-mu1)
        assert(theta0.shape == (1,1))
        assert(theta.shape == (n,))

        self.theta = np.insert(theta,0,np.ndarray.item(theta0),axis=0)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        p_y = 1/(1+np.exp(-x@self.theta))
        h_x = (p_y>0.5).astype(int)
        return h_x
        # *** END CODE HERE

def binom_normality(x,y):
    m,n = x.shape
    x_new = np.copy(x[:,1:])
    assert(x_new.shape == (m,(n-1)))

    res_x0 = stats.normaltest(x_new[y==0])
    res_x1 = stats.normaltest(x_new[y==1])
    
    print('normality pvalues for X|Y=0: ', res_x0.pvalue)
    print('normality pvalues for X|Y=1: ', res_x1.pvalue)
    return 

def accuracy(predict, y):
    return sum(predict==y)/len(y)
