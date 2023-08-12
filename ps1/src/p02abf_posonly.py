import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    # Part (b): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
   
    #part (a) train logistic regression on true labels t
    X_train, t_train = util.load_dataset(train_path,label_col='t', add_intercept=True)
    X_test, y_test = util.load_dataset(test_path,label_col='t', add_intercept=True)
    clf = LogisticRegression()
    clf.fit(X_train,t_train)
    y_predict = clf.predict(X_test)
    util.plot(X_test, y_test, clf.theta, pred_path,correction=0.0)

    #part (b) train logreg on y labels
    X_train, y_train = util.load_dataset(train_path,label_col='y', add_intercept=True)
    X_test, y_test = util.load_dataset(test_path,label_col='t', add_intercept=True)
    clf_y = LogisticRegression()
    clf_y.fit(X_train,y_train)
    y_predict_on_y = clf_y.predict(X_test)
    np.savetxt(pred_path, y_predict_on_y) 
    util.plot(X_test,y_test,clf_y.theta,pred_path,correction=0.0)

    #part (f) estimate 
    X_train, y_train = util.load_dataset(train_path,label_col='y', add_intercept=True)
    X_valid, y_valid = util.load_dataset(valid_path,label_col='y', add_intercept=True)
    X_test, y_test = util.load_dataset(test_path,label_col='t', add_intercept=True)
    
    # fit the model using only the labeled training data
    clf_a = LogisticRegression()
    clf_a.fit(X_train, y_train)
    
    # estimate alpha using the validation set
    p_y = clf_a.predict(X_valid)
    print('p_y:', p_y)
    h_x = (p_y>0.5)
    print('sum of h(x) in validation set is',np.sum(h_x))

    # Current issue: h_x does not predict any positive labels!

    # Vplus = np.sum(y_valid)
    # alpha = 1/Vplus*np.sum(h_x)
    # print(alpha)

    # use the alpha parameter to adjust the predictions for the test set
    # p_y = clf_a.predict(X_test)
    # p_t = 1/alpha*p_y
    # y_predict = (p_t>0.5).astype(int)
    #np.savetxt(pred_path, y_predict) 

    #How can I adjust the theta's of clf_a for the plot?
    #c_factor = np.log((2-alpha)/alpha)
    #util.plot(X_test,y_test,clf_a.theta,pred_path,correction = c_factor)

    # *** END CODE HERE
