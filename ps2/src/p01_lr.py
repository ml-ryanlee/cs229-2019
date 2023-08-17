# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10
    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        #learning_rate = 1/(i*i)
        if i % 10000 == 0:
            print('theta: ', theta)
            #print('gradient values: ',grad)
            print('Finished {} iterations, update scale: {}'.format(i, np.linalg.norm(prev_theta - theta)))
        if np.linalg.norm(prev_theta - theta) < 1e-10:
            print('Converged in %d iterations' % i)
            break
    return theta

def main():
    
    # Want to view data
    save_path = '../output/p1_decision_boundary.txt'


    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    #theta_a = logistic_regression(Xa, Ya)
   
    print('\n==== Training model on data set B ====')
    
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    #theta_b = logistic_regression(Xb, Yb)
    
    # theta_nolr_a = np.zeros(Xa.shape[1])
    # theta_nolr_b = np.zeros(Xb.shape[1])
    # util.plot(Xa,Ya,theta_a,save_path)
    # util.plot(Xb,Yb,theta_nolr_b,save_path)

    # model sigmoid fit for increasingly separated data for X|Y=0 and X|Y=1
    plt.figure()
    dist = np.array([0.01,0.04,0.08,0.1])
    sigma = 0.1
    examples = 1000
    n_plot = 1000
    n_each = int(examples/2)
    for d in dist:
        #model gaussian distributed feature x|y=0 & x|y=1, with increasing mean distance
        theta = np.zeros(n_each)
        mu = d
        x_plus = np.random.normal(mu,sigma,n_each).reshape(n_each,1)
        x_minus = np.random.normal(-mu,sigma,n_each).reshape(n_each,1)
        x = np.append(x_plus,x_minus,axis=0)
        y_plus = np.ones(n_each).reshape(n_each,1)
        y_minus = -np.ones(n_each).reshape(n_each,1)
        y = np.append(y_plus,y_minus,axis=0).reshape((examples,))
        theta = logistic_regression(x,y)
        print('theta with',d,'distance','is',theta)
        x_plot = np.linspace(-5,5,n_plot).reshape(n_plot,1)
        plt.plot(x_plot,g(x_plot,theta))
    plt.legend((dist*2.0))
    plt.title('sigmoid fitted vs. mean distance between labeled examples')
    plt.xlabel('x')
    plt.ylabel('sigmoid(x) fitted')
    plt.show()

    # plt.figure()
    # plt.plot(Xa[0],Xa[1],'o')
    # plt.title('data set A')
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.plot(Xb,Yb,'o')
    # plt.title('data set A')
    # plt.legend()
    # plt.show()

def g(x,theta):
    return 1/(1+np.exp(-x@theta))


if __name__ == '__main__':
    main()
    # test()
