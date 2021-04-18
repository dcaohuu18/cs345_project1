# Dong Cao
# test_naive_bayes.py


import pandas as pd
import argparse
import numpy as np
from naive_bayes_clf import NaiveBayesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from helper_functions import leave_one_out_cv


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

    my_cv_score = leave_one_out_cv(NaiveBayesClassifier, X, y)
    print("My cv score: ", my_cv_score)
    
    skl_cv_score = leave_one_out_cv(MultinomialNB, X, y)
    print("Skl cv score: ", skl_cv_score)


if __name__ == '__main__':
    test()