import pandas as pd
import argparse
import numpy as np
from naive_bayes_clf import NaiveBayesClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_csv", help="Enter the dataset's CSV filename", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_csv)
    X = np.array(df.iloc[:, 1:-1]) # I scored higher on 1:-2
    y = np.array(df.iloc[:, -1])

    # Encode the data from str to number so it works with Sklearn:
    oe = OrdinalEncoder()
    X = oe.fit_transform(X)

    le = LabelEncoder()
    y = le.fit_transform(y)

    my_accuracy=0
    skl_accuracy=0
    
    for i in range(len(X)): # Leave one out cross validation
        X_train = np.delete(X, i, 0)
        y_train = np.delete(y, i, 0)

        X_test = np.array([X[i]])
        y_test = np.array([y[i]])

        my_bayes = NaiveBayesClassifier()
        my_bayes.fit(X_train, y_train)

        skl_bayes = MultinomialNB()
        skl_bayes.fit(X_train, y_train)

        my_accuracy += accuracy_score(y_test, my_bayes.predict(X_test))
        skl_accuracy += accuracy_score(y_test, skl_bayes.predict(X_test))

        print(my_bayes.predict(X_test), skl_bayes.predict(X_test), y_test)

    print("My mean accuracy: ", my_accuracy/len(X))
    print("Skl mean accuracy: ", skl_accuracy/len(X))


if __name__ == '__main__':
    test()