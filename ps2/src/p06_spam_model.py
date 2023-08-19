import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    return message.lower().split()
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message. 

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    dict_msgcount = {}
    dict_common = {}
    msg_threshold = 5

    #track number of different messages the word appeared in with dict_msgcount 
    for line in messages:
        # get a list of (unique) words from message line
        line_list = np.unique(get_words(line))
        
        # for every unique word in line list
        for word in line_list:
            if word in dict_msgcount:
                dict_msgcount[word] += 1
            else:
                dict_msgcount[word] = 1
    
    #create a dict that contains words that have appeared in at least 5 messages
    word_idx = 0
    for key in sorted(dict_msgcount.keys()): #make this alphabetical, because that's the convention
        if (dict_msgcount[key] >= msg_threshold):
            dict_common[key] = word_idx
            word_idx+=1    
    
    return dict_common
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each 
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that 
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    # *** START CODE HERE ***
    text_array = np.zeros((len(messages),len(word_dictionary)))

    row_idx = 0
    for line in messages:
        line_list = get_words(line)
        for word in line_list:
            if word in word_dictionary:
                col_idx = word_dictionary[word]
                text_array[row_idx,col_idx]+=1
        row_idx += 1
    return text_array
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider
    
    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    # *** END CODE HERE ***


def main():
    # Load datasets
    train_messages, train_labels = util.load_spam_dataset('../data/ds6_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('../data/ds6_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('../data/ds6_test.tsv')
    
    # create dictionary
    dictionary = create_dictionary(train_messages)
    util.write_json('./output/p06_dictionary', dictionary)

    # create NB multinomial event model feature matrices
    train_matrix = transform_text(train_messages, dictionary)
    np.savetxt('./output/p06_sample_train_matrix', train_matrix[:100,:])
    val_matrix = transform_text(val_messages,dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    # create a naive bayes classifier, fit it with training matrix and labels
    clf = naive_bayes()
    clf.fit(train_matrix,train_labels)

    # return probabilities* of val_matrix
    probs0, probs1 = clf.predict(val_matrix)
    print('probs0:', probs0,'\nprobs1:',probs1)

    # print('clf priors:',clf.py0,clf.py1,'\nclf probs:', clf.phi0,clf.phi1)
    # print('clf probs shape x|y=1:', clf.phi1.shape, 'clf probs x|y=0:',clf.phi0.shape)
    #naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)
    
    #print(naive_bayes_model)
    # naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    # np.savetxt('./output/p06_naive_bayes_predictions', naive_bayes_predictions)

    # naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    # print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    # top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    # print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    # util.write_json('./output/p06_top_indicative_words', top_5_words)

    # optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    # util.write_json('./output/p06_optimal_radius', optimal_radius)

    # print('The optimal SVM radius was {}'.format(optimal_radius))

    # svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    # svm_accuracy = np.mean(svm_predictions == test_labels)

    # print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))

class naive_bayes(object):
    def __init__(self, py0=None,py1=None,phi0=0,phi1=0):
        self.py0 = py0
        self.py1 = py1
        self.phi0 = phi0
        self.phi1 = phi1
    
    def fit(self, matrix,labels):
        
        # Priors: P(Y=1),P(Y=0)
        self.py1 = sum(labels)/len(labels)
        self.py0 = 1-self.py1
    
        # MLE estimators, P(xj=k|Y=1) with Laplace Smoothing
        n = matrix.shape[1] #number of indexed words in dict & feature matrix
        x_y1 = matrix[labels==1]
        x_y0 = matrix[labels==0]

        # xj|Y prob = (# of xj words in Y=1) / (# of words in Y=1)
        # Laplace smoothing: [(# of xj words in Y=1)+1] / [(# of words in Y=1)+n]
        self.phi1 = (sum(x_y1,0)+1)/(sum(x_y1.flatten())+n)
        self.phi0 =(sum(x_y0,0)+1)/(sum(x_y0.flatten())+n)
        assert(self.phi1.shape == (n,))
        assert(self.phi0.shape == (n,))
    
    def predict(self, matrix):
        print('in predict method')
        m = len(matrix)
        
        # calculate log(p(x|y=k)),l_pxyk, to prevent underflow and for efficiency
        l_pxy0 = matrix@np.log(self.phi0)
        l_pxy1 = matrix@np.log(self.phi1)

        # extract p(x|y=1) by taking exp of log
        pxy0 = np.exp(l_pxy0)
        pxy1 = np.exp(l_pxy1)
        assert (pxy0.shape == (m,))
        assert(pxy1.shape == (m,))

        # calculate naive bayes classifier probabilities, with Bayes Rule
        probs_y0 = (pxy0*self.py0)/(pxy0*self.py0+pxy1*self.py1)
        probs_y1 = (pxy1*self.py1)/(pxy0*self.py0+pxy1*self.py1)
        return probs_y0,probs_y1



if __name__ == "__main__":
    main()