from sklearn.metrics import accuracy_score
import numpy as np


def check_X(X):
    X = np.array(X)
    if len(X.shape)!=2:
        raise ValueError("X must be a two-dimensional array of the shape (m,n)")
    else:
        return X


def check_y(y):
    y = np.array(y)
    if len(y.shape)!=1:
        raise ValueError("y must be a one-dimensional array of the shape (m,1)")
    else:
        return y


def check_X_y(X, y):
    if len(X) != len(y):
        raise ValueError("X and y must be of the same size")
    return check_X(X), check_y(y)


def leave_one_out_cv(clf_object, X, y, **clf_kwarg):
    accuracy_sum=0

    for i in range(len(X)):
        X_train = np.delete(X, i, 0)
        y_train = np.delete(y, i, 0)

        X_test = np.array([X[i]])
        y_test = np.array([y[i]])

        clf = clf_object(**clf_kwarg)
        clf.fit(X_train, y_train)

        accuracy_sum += accuracy_score(y_test, clf.predict(X_test))

    return accuracy_sum/len(X)