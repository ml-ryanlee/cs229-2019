import util
import sys
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import inv

np.seterr(all='raise')
factor = 2.0

def main(train_path, eval_path):
    '''
    Run all experiments
    '''
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    
    # *** START CODE HERE ***
    # part b/c) plot polynomial linear regression specified by ks onto training
    x_max = max(x_train[:,1])
    x_min = min(x_train[:,1])
    x_span = np.linspace(x_min,x_max,1000).reshape((1000,1))
    x_span = util.add_intercept(x_span)

    ks=[3,5,10,20]
    k_label = ['train data','k=3','k=5','k=10','k=20']
    ks_line = ['b-','g-','c-','m-']
    clf = LinearModel()
    plt.figure()
    plt.plot(x_train[:,1:],y_train,'o')
    for i in range(len(ks)):
        x_train_map = create_poly(ks[i],x_train)
        clf.fit(x_train_map,y_train)
        
        phi_x_span = create_poly(ks[i],x_span)
        y_span = clf.predict(phi_x_span)
        
        plt.plot(x_span[:,1:],y_span, ks_line[i])
    plt.legend(k_label)
    plt.xlabel('x attribute')
    plt.ylabel('y response')
    plt.title('Training Dataset vs. Predicted Response, polynomial features')
    plt.show()

    # part d)
    plt.figure()
    plt.plot(x_train[:,1:],y_train,'o')
    
    # test
    x_train_map = create_poly(3,x_train)
    print('x_train after polyfit: ',x_train_map.shape)
    print('x_train: ','\n',x_train_map)
    x_train_map = sinetransform(x_train_map) 
    print('x_train after sine transform: ',x_train_map.shape)
    print('x_train: ','\n',x_train_map)
    
    for i in range(len(ks)):
        x_train_map = create_poly(ks[i],x_train)
        x_train_map = sinetransform(x_train_map)
        clf.fit(x_train_map,y_train)
        
        phi_x_span = create_poly(ks[i],x_span)
        phi_x_span = sinetransform(phi_x_span)
        y_span = clf.predict(phi_x_span)
        
        plt.plot(x_span[:,1:],y_span, ks_line[i])
    plt.legend(k_label)
    plt.xlabel('x attribute')
    plt.ylabel('y response')
    plt.title('Training Dataset vs. Predicted Response, polynomial features, sin(x)')
    plt.show()
    # *** END CODE HERE ***

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, x, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # 0. initialize theta
        n = x.shape[1]
        self.theta = np.zeros(n)
        
        # 1. update to theta with normal equations
        self.theta = inv(x.T@x)@(x.T@y)
        assert(self.theta.shape == (n,))
        # *** END CODE HERE ***

    def predict(self, x):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return x@self.theta
        # *** END CODE HERE ***

def create_poly(k, x):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            x: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        m = len(x)
        min_k = 2
        if (k<min_k): return np.copy(x)
        phi_x = np.copy(x)
        
        for i in range(k-1):
            attr = x[:,-1].reshape((m,1))    
            phi_x = np.append(phi_x,attr**(i+min_k),axis=1)
        return phi_x
        # *** END CODE HERE ***

def sinetransform(x):
    #assumes offset column added in x
    m = len(x)
    attr = x[:,1].reshape((m,1))
    attr_sin = np.sin(attr).reshape((m,1))
    assert(attr_sin.shape == (m,1)) 
    return np.append(x,attr_sin,axis=1)

if __name__ == '__main__':
    main(train_path='train.csv',
        eval_path='test.csv')