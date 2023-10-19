import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import openpyxl
from sklearn import model_selection, metrics, neighbors
from sklearn.model_selection import StratifiedKFold


def task1():
    df = pd.read_excel('datasets/movie_reviews.xlsx')
    # Review / Sentiment(negative/positive) / Split(train/test)
    df['Sentiment'] = df['Sentiment'].str.replace('positive', '1')
    df['Sentiment'] = df['Sentiment'].str.replace('negative', '0')
    df['Sentiment'] = df['Sentiment'].apply(pd.to_numeric, errors='coerce')

    # reviews
    training_data_reviews = df.loc[df['Split'] == 'train', 'Review'].tolist()
    test_data_reviews = df.loc[df['Split'] == 'test', 'Review'].tolist()
    # sentiment labels
    training_data_sentiment = df.loc[df['Split'] == 'train', 'Sentiment'].tolist()
    test_data_sentiment = df.loc[df['Split'] == 'test', 'Sentiment'].tolist()

    print("Training data: Positive reviews\n{}".format(training_data_sentiment.count(1)))
    print("Training data: Negative reviews\n{}".format(training_data_sentiment.count(0)))
    print("Test data: Positive reviews\n{}".format(test_data_sentiment.count(1)))
    print("Test data: Negative reviews\n{}".format(test_data_sentiment.count(0)))

    return (training_data_reviews, test_data_reviews, training_data_sentiment, test_data_sentiment,
            training_data_sentiment.count(1), training_data_sentiment.count(0))


def clean(review):
    review = (re.sub('[^a-zA-Z0-9\s]', '', review)).lower()
    return review


def task2(review, min_word_length, min_word_count):
    review = review.split()

    # populating dictionary with words above minimum length
    word_dictionary = {}
    for word in review:
        if word not in word_dictionary:
            if len(word) > min_word_length:
                word_dictionary[word] = 1
        else:
            word_dictionary[word] += 1

    # extracting all words above minimum count to a list
    valid_words = []
    for word in word_dictionary:
        if word_dictionary[word] > min_word_count:
            valid_words.append(word)

    # returning our valid word list
    return valid_words


def task3(review, count, word):
    if word in review:
        return count + 1
    else:
        return count


def task4(len_words, review_dictionary):
    # laplace alpha = 1
    # feature can be positive or negative, so 2
    # word occurrence + alpha
    # total positives + (alpha * features)
    alpha = 1
    features = 2
    x = (len_words + (alpha * features))
    return_dictionary = {}
    for rev in review_dictionary:
        y = (int(review_dictionary[rev]) + alpha)
        z = y/x
        return_dictionary.update({rev: z})
    return return_dictionary


def task5(review, positive_likelihood, negative_likelihood, priors):
    review_list = review.split()
    positive_count = 0
    negative_count = 0
    for word in review_list:
        if word in positive_likelihood:
            positive_count += math.log(positive_likelihood[word])
        if word in negative_likelihood:
            negative_count += math.log(negative_likelihood[word])

    p = positive_count - negative_count
    if p >= priors:
        return 1
    else:
        return 0


def main():
    # task 1
    (training_data_reviews, test_data_reviews, training_data_sentiment,
     test_data_sentiment, positive_count, negative_count) = task1()

    # task 2
    clean_training_data_reviews = []
    for review in training_data_reviews:
        clean_training_data_reviews.append(clean(review))

    clean_test_data_reviews = []
    for review in test_data_reviews:
        clean_test_data_reviews.append(clean(review))

    word_list = []
    for review in clean_training_data_reviews:
        # tested on word count of 4
        # change word length, 1 - 10 to test accuracy
        # 10 = 53%
        # 9 = 57%
        # 8 = 61%
        # 7 = 68%
        # 6 = 72%
        # 5 = 74%
        # 4 = 78.1%
        # 3 = 78.9%
        # 2 = 78.3%
        # 1 = 78.2%
        word_list.append(task2(review, 3, 4))

    # task 3
    word_list_numpy = np.concatenate([np.array(w) for w in word_list])
    word_list = np.unique(word_list_numpy).tolist()

    positive_dictionary = {}
    negative_dictionary = {}
    for w in word_list:
        positive_dictionary.update({w: 0})
        negative_dictionary.update({w: 0})
        for x in range(0, len(clean_training_data_reviews)):
            if training_data_sentiment[x] == 1:
                # positive
                new_val = task3(clean_training_data_reviews[x], positive_dictionary[w], w)
                positive_dictionary.update({w: new_val})
            else:
                # negative
                new_val = task3(clean_training_data_reviews[x], negative_dictionary[w], w)
                negative_dictionary.update({w: new_val})

    # task 4
    # stored in log form
    positive_likelihood = task4(positive_count, positive_dictionary)
    negative_likelihood = task4(negative_count, negative_dictionary)

    prior_positive = positive_count / len(clean_training_data_reviews)
    prior_negative = negative_count / len(clean_training_data_reviews)
    priors = prior_positive - prior_negative

    # task 5
    new_review = clean_test_data_reviews[201]
    new_sentiment = test_data_sentiment[201]

    likelihood = task5(new_review, positive_likelihood, negative_likelihood, priors)
    if likelihood == new_sentiment:
        print("Predicted test review correctly")
    else:
        print("Predicted test review incorrectly")
    print()

    # task 6
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    my_predictions = []
    i = 0
    for review in clean_test_data_reviews:
        likelihood_loop = task5(review, positive_likelihood, negative_likelihood, priors)
        my_predictions.append(likelihood_loop)
        # true
        if likelihood_loop == test_data_sentiment[i]:
            # my_predictions.append(1)
            if test_data_sentiment[i] == 1:
                true_positive += 1
            else:
                true_negative += 1
        # false
        else:
            # my_predictions.append(0)
            if test_data_sentiment[i] == 1:
                false_positive += 1
            else:
                false_negative += 1
        i += 1

    print("True positive: {}".format(true_positive/len(test_data_sentiment)))
    print("True negative: {}".format(true_negative/len(test_data_sentiment)))
    print("False positive: {}".format(false_positive/len(test_data_sentiment)))
    print("False negative: {}".format(false_negative/len(test_data_sentiment)))
    print()
    print("Correct predictions: {}".format((true_positive+true_negative)/len(test_data_sentiment)))
    print()
    confusion = metrics.confusion_matrix(test_data_sentiment, my_predictions)
    print("Confusion:\n {}".format(confusion))

    accuracy = metrics.accuracy_score(my_predictions, test_data_sentiment)
    print("Accuracy: {}".format(accuracy))


main()
