# Dong Cao
# ada_boost.py


import numpy as np
import copy
import random
import bisect
from collections import defaultdict
import operator
from helper_functions import check_X, check_X_y


class DecisionStump:
    def __init__(self):
        self.fit_called = False
        pass

    def fit(self, X, y):
        self.fit_called = True

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.classes_rows = {}
        self.classes_ = np.unique(y)
        
        for class_ in np.unique(y):
            class_rows = X[y==class_] # get the array of feature rows that belong to each class
            self.classes_rows[class_] = class_rows

        self.chosen_col_idx = 0
        self.decision_dict = {}
        min_gini = float('inf') # we pick the column with the minimum Gini impurity

        for column_idx in range(X.shape[1]):
            column_gini = 0
            col_decision_dict = {} # the decision_dict of this column, it may not be chosen in the end

            for uni_val in np.unique(X[:, column_idx]): # this only works with categorical data
                uni_val_gini = 1
                # select from the entire training set (X) the rows whose value at column_idx = uni_val:
                uni_val_rows = X[X[:, column_idx]==uni_val] 

                max_votes = 0
                voted_class = None 
                # the class that is voted for this uni_val of this column, which has the highest votes: max_votes

                for class_ in self.classes_:
                    class_rows = self.classes_rows[class_]
                    # select from the the rows of this class the rows whose value at column_idx = uni_val:
                    uni_val_class_rows = class_rows[class_rows[:, column_idx]==uni_val]

                    if len(uni_val_class_rows)>max_votes:
                        max_votes = len(uni_val_class_rows)
                        voted_class = class_

                    uni_val_gini = uni_val_gini - ( len(uni_val_class_rows)/len(uni_val_rows) )**2
                
                col_decision_dict[uni_val] = voted_class

                # the Gini impurity of the column = the WEIGHTED sum of the column's unique values' individual gini
                column_gini += len(uni_val_rows)*uni_val_gini 

            if column_gini < min_gini:
                min_gini = column_gini
                self.chosen_col_idx = column_idx
                self.decision_dict = col_decision_dict
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check if fit() had been called
        assert self.fit_called==True, "fit() has not been called"

        # Input validation
        X = check_X(X)

        predict_array = []

        for x_row in X:
            x_row_chosen_col_val = x_row[self.chosen_col_idx]
            try:
                prediction = self.decision_dict[x_row_chosen_col_val]
            except KeyError: # x_row_chosen_col_val is not present in the training set
                prediction = None
            predict_array.append(prediction)

        return np.array(predict_array)


class AdaBoost:
    def __init__(self, learner_num, learner_type=DecisionStump, **learner_kwargs):
        self.learner_num = learner_num # total number of small, weak learners used
        self.learner_obj = learner_type(**learner_kwargs)
        self.learner_say_dict = {} # key is learner, value is its amount of say in the final classification
        self.fit_called = False


    def fit(self, X, y):
        self.fit_called = True

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # assign weight to each sample/row in the original dataset to specify its "importance"
        # at first, they're all equally important
        self.sample_weights = np.array([1/len(X) for r in range(len(X))])
        # we will ocassionally resample our data
        # note that each row/sample in the original dataset
        # can be selected and appear more than once in the resampled version
        # the higher the weight of a sample/row in the original dataset (recorded in self.sample_weights) is
        # the more frequently it will appear in the resampled version
        # however, the weights of all the rows in the resampled version, once selected, are (always) equal
        resampled_weights = np.array([1/len(X) for r in range(len(X))])

        for i in range(self.learner_num):
            if i==0:
                # no resampling in the 1st round:                
                resampled_idx, resampled_X, resampled_y = np.array(range(len(X))), X, y
                # resampled_idx contains the indexes in the original dataset 
                # of the rows in the resampled version
            else:
                resampled_idx, resampled_X, resampled_y = self.resample(X, y)
            
            learner_i = copy.deepcopy(self.learner_obj)
            learner_i.fit(resampled_X, resampled_y)

            learner_i_preds = learner_i.predict(resampled_X)
            # the total error is the sum of the weights associated with the incorrectly classified rows
            # we add a small error term to make sure it's never equal to 0 or 1, 
            # otherwise, it will mess up the amount of say formula
            tot_error = abs( resampled_weights[learner_i_preds!=resampled_y].sum() - (1/(100*len(resampled_X))) )
            # apply the formula to get this learner's amount of say:
            learner_i_say_amt = 0.5*np.log( (1-tot_error)/tot_error )
            
            self.learner_say_dict[learner_i] = learner_i_say_amt

            self.update_sample_weights(resampled_idx, learner_i_preds, resampled_y, learner_i_say_amt)

        # Return the classifier
        return self

    def update_sample_weights(self, resampled_idx, learner_i_preds, resampled_y, learner_i_say_amt):
        # Note: we only update the weight of each row in the original dataset ONCE
        # even though there may be multiple copies of it in the resampled version

        # increase the weight of the incorrect guesses:
        for wrong_wei_idx in np.unique(resampled_idx[learner_i_preds!=resampled_y]):
            self.sample_weights[wrong_wei_idx] *= np.exp(learner_i_say_amt)
        # decrease the weight of the correct guesses:
        for right_wei_idx in np.unique(resampled_idx[learner_i_preds==resampled_y]):
            self.sample_weights[right_wei_idx] *= np.exp(-learner_i_say_amt) # raised to a negative power
        # normalize the weights so they add up to 1:
        self.sample_weights /= self.sample_weights.sum()

    def resample(self, X, y):
        cumulative_weights = np.cumsum(self.sample_weights)
        resampled_idx = []
        for i in range(len(X)):
            # select a random number between 0 and 1
            rand_num = random.uniform(0,1)
            # find which "range" this random number belongs to
            # then pick the corresponding row index
            # Note: the higher the weight of a row, the larger its range
            chosen_row_idx = bisect.bisect(cumulative_weights, rand_num)  
            resampled_idx.append(chosen_row_idx)

        resampled_idx = np.array(resampled_idx)

        return resampled_idx, X[resampled_idx], y[resampled_idx]

    def predict_single(self, x_row):
        votes_dict = defaultdict(int) 
        # the key is the classification/prediction, 
        # the value is the total amount of say it gets
        for learner, learner_say_amt in self.learner_say_dict.items():
            learner_pred = learner.predict([x_row])[0]
            if learner_pred is None:
                continue 
                # don't consider this learner's prediction 
                # because it fails to classify x_row
            votes_dict[learner_pred] += learner_say_amt

        # return the prediction with the highest amt of say:
        return max(votes_dict.items(), key=operator.itemgetter(1))[0]   

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

