from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import pandas as pd
import argparse
import numpy as np
from ada_boost import DecisionStump, AdaBoost
from helper_functions import leave_one_out_cv


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_csv", help="Enter the dataset's CSV filename", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_csv)
    X = np.array(df.iloc[:, 1:-1])
    y = np.array(df.iloc[:, -1])

    oe = OrdinalEncoder()
    X = oe.fit_transform(X)

    le = LabelEncoder()
    y = le.fit_transform(y)

    stump_cv_score = leave_one_out_cv(DecisionStump, X, y)
    print("Stump cv score: ", stump_cv_score)

    max_boost_cv_score = 0

    for _ in range(100):
        cv_score = leave_one_out_cv(AdaBoost, X, y, learner_num=10)
        
        if cv_score > max_boost_cv_score:
            max_boost_cv_score = cv_score

    print("Max AdaBoost cv score: ", max_boost_cv_score)


if __name__ == '__main__':
    test()