import numpy as np


class NaiveBayesClassifier():
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.classes_dict = {}

    def fit(self, X, y):
        # Check that X and y have correct shape
        
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
        # Check is fit() had been called

        # Input validation

        predict_array = []

        for x_row in X:
            prediction = self.predict_single(x_row)
            predict_array.append(prediction)

        return np.array(predict_array)


if __name__ == '__main__':
    from sklearn.metrics import accuracy_score
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
    import pandas as pd

    df = pd.read_csv('play_tennis.csv')
    X = np.array(df.iloc[:, 1:-1]) # I scored higher on 1:-2
    y = np.array(df.iloc[:, -1])

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

    # the two cases I get wrong:
    # Overcast,Mild,High,Strong,Yes
    # Rain,Mild,High,Weak,Yes

    print("My mean accuracy: ", my_accuracy/len(X))
    print("Skl mean accuracy: ", skl_accuracy/len(X))