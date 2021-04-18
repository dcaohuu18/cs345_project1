# cs345_project1
In this project for CS345 Software Enginnering, I implement two Machine Learning classification algorithms from scratch: Multinomial Naive Bayes and AdaBoost.

## API
* My API is inspired by the API of ``scikit-learn`` with a ``Classifier()`` object and two methods ``fit()`` and ``predict()``. An example call would be:
```python
clf = Classifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```
* All the ``X`` arrays should be two-dimensional and is of the form ``m x n`` where ``m`` is the number of rows/samples and ``n`` is the number of columns/features. The ``y`` array, i.e. the target, should be a one-dimensional array of the form ``m x 1``.
* Arguments of the class ``NaiveBayesClassifier``:
  * ``alpha``: the pseudo amount used in [Laplace smoothing](https://en.wikipedia.org/wiki/Additive_smoothing). The default value is 1.
* Arguments of the class ``AdaBoost``:
  * ``learner_num``: the number of individual (weak) learners used
  * ``learner_type``: the type of learner used. By default, ``DecisionStump`` is used. However, any learner object with ``fit()`` and ``predict()`` methods can be used. For example, you can use ``NaiveBayesClassifier()``.
  * ``**learner_kwargs``: the keyword arguments for the learner object.
  
 ## Testing
  * To quickly test the algorithms, from the command line, run: ``$ python3 test_naive_bayes.py <dataset.csv>``. You can follow the same syntax for ``test_ada_boost.py``. 
  * The two available datasets in the repository are ``play_tennis.csv`` and ``will_wait.csv``. **Note:** the two algorithms only work with categorical features for now.
  * When you run ``test_naive_bayes.py``, it will print out the leave-one-out cross validation scores of my implementation and the implementation of ``scikit-learn`` for comparision.
  * When you run ``test_ada_boost.py``, it will print out the leave-one-out cross validation score of one ``DecisionStump`` and the max score of ``AdaBoost`` after 100 trials. This is because my AdaBoost makes use of (random) resampling instead of reweighing, so there is some randomness involved.
  
  ## Dependencies:
  * For building the algorithms:
    * ``numpy``
  * For testing the algorithms:
    * ``pandas``
    * ``scikit-learn``
