# Dong Cao
# naive_bayes_clf.py


import numpy as np
from helper_functions import check_X, check_X_y


class NaiveBayesClassifier():
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.classes_dict = {}
        self.fit_called = False

    def fit(self, X, y):
        self.fit_called = True

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        for class_ in np.unique(y):
            class_rows = X[y==class_] # get the array of feature rows that belong to each class
            self.classes_dict[class_] = {
                'feature_rows': class_rows,
                'prob': len(class_rows)/len(y)
            }

        # Return the classifier
        return self

    def predict_single(self, x_row):
        max_prob_class = None
        max_prob = float('-inf')
        for class_, class_info in self.classes_dict.items():
            class_rows = class_info['feature_rows']
            log_cond_probs = 0

            for column_idx, column_val in enumerate(x_row):
                # rows of this class that have matching value with this column of x_row:
                match_rows = class_rows[class_rows[:, column_idx] == column_val]

                # Laplace smoothing to avoid zero probability:
                # it works by adding a pseudo count (alpha) to each distinct value of the feature column
                pseudo_total = len(class_rows) + self.alpha*len(np.unique(class_rows[:, column_idx])) 

                if len(match_rows)==0: # column_val is not present in the column
                    pseudo_total = pseudo_total + self.alpha 
                    # + self.alpha for the absent column_val, which is considered another distinct value

                # take the sum of log(P(column|class)):
                # use log instead of multiplying the probabilities to avoid the final prob being too close to 0
                # which may cause an underflow
                cond_prob = (len(match_rows)+self.alpha)/pseudo_total
                log_cond_probs = log_cond_probs + np.log(cond_prob)

            total_prob = np.log(class_info['prob']) + log_cond_probs
            if total_prob > max_prob:
                max_prob = total_prob
                max_prob_class = class_

        return max_prob_class

    def predict(self, X):
        # Check if fit() had been called
        assert self.fit_called==True, "fit() has not been called"

        # Input validation
        X = check_X(X)

        predict_array = []

        for x_row in X:
            prediction = self.predict_single(x_row)
            predict_array.append(prediction)

        return np.array(predict_array)

        