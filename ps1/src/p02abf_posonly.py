import numpy as np
import util

from p01b_logreg import LogisticRegression
from p01b_logreg import accuracy

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
    pred_path_a = pred_path.replace(WILDCARD, 'c')
    pred_path_b = pred_path.replace(WILDCARD, 'd')
    pred_path_f = pred_path.replace(WILDCARD, 'e')

    # Load datasets
    x_train, t_train = util.load_dataset(train_path,label_col='t', add_intercept=True)
    x_train, y_train = util.load_dataset(train_path,label_col='y', add_intercept=True)
    x_val, y_val = util.load_dataset(valid_path, label_col= 'y', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path,label_col='t', add_intercept=True)

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels, save outputs to pred_path_a
    clf_t = LogisticRegression()
    clf_t.fit(x_train,t_train)
    t_predict = clf_t.predict(x_test)
    np.savetxt(pred_path_a, t_predict) 
    util.plot(x_test, t_test, clf_t.theta, pred_path,correction=1.0)
    print('LR accuracy on test DS, trained on true labels: ', accuracy(t_predict,t_test))

    # Part (b): Train on y-labels and test on true labels, save outputs to pred_path_b
    clf_y = LogisticRegression()
    clf_y.fit(x_train,y_train)
    y_predict = clf_y.predict(x_test)
    np.savetxt(pred_path_b, y_predict) 
    util.plot(x_test,t_test,clf_y.theta,pred_path,correction=1.0)
    print('LR accuracy on test DS, trained on y labels: ', accuracy(y_predict,t_test))
    
    # Part (e): Apply correction factor using validation set, test on true labels. Save outputs to pred_path_f
    
    # calculate alpha based on subset of positive y examples in validation set
    x_val_vplus = x_val[y_val==1]
    h_x_vplus = clf_y.predict_prob(x_val_vplus)
    vplus_len = len(h_x_vplus)
    alpha = sum(h_x_vplus)/vplus_len

    # adjustment to logreg prediction for labeled predictions
    alpha_prob = (1/alpha) * clf_y.predict_prob(x_test)
    alpha_predict = (alpha_prob>0.5).astype(int)
    
    # plotting correction factor
    theta0 = clf_y.theta[0]
    alpha_correction = (1/theta0)*np.log((2-alpha)/alpha)+1

    util.plot(x_test,t_test,clf_y.theta,pred_path,correction=alpha_correction)
    print('LR accuracy on test DS, with alpha factor: ', accuracy(alpha_predict,t_test))
    # print('number of positive predictions: ',sum(h_x_alpha))



    # *** END CODE HERE
