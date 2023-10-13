import pandas as pd
import openpyxl


def task1():
    df = pd.read_excel('datasets/movie_reviews.xlsx')
    # Review / Sentiment(negative/positive) / Split(train/test)
    df['Sentiment'] = df['Sentiment'].str.replace('positive', '1')
    df['Sentiment'] = df['Sentiment'].str.replace('negative', '0')

    # reviews
    training_data_reviews = df.loc[df['Split'] == 'train', 'Review'].tolist()
    test_data_reviews = df.loc[df['Split'] == 'test', 'Review'].tolist()
    # sentiment labels
    training_data_sentiment = df.loc[df['Split'] == 'train', 'Sentiment'].tolist()
    test_data_sentiment = df.loc[df['Split'] == 'test', 'Sentiment'].tolist()

    print("Training data: Positive reviews\n{}".format(training_data_sentiment.count('1')))
    print("Training data: Negative reviews\n{}".format(training_data_sentiment.count('0')))
    print("Test data: Positive reviews\n{}".format(test_data_sentiment.count('1')))
    print("Test data: Negative reviews\n{}".format(test_data_sentiment.count('0')))

    return training_data_reviews, test_data_reviews, training_data_sentiment, test_data_sentiment


def task2(training_data_reviews):
    print(training_data_reviews)


def main():
    training_data_reviews, test_data_reviews, training_data_sentiment, test_data_sentiment = task1()
    task2(training_data_reviews)

main()