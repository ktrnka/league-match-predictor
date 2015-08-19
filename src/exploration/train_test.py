from __future__ import unicode_literals
import sys
import argparse
import pandas
import numpy
import sklearn.ensemble
import sklearn.cross_validation
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.svm
import sklearn.grid_search
import sklearn.tree
import sklearn.learning_curve
import matplotlib.pyplot as plt
from operator import itemgetter


def learning_curve(training_x, training_y, filename, classifier):
    """Make a learning graph and save it"""
    split_iterator = sklearn.cross_validation.StratifiedShuffleSplit(training_y, n_iter=10, random_state=4)
    train_sizes, train_scores, test_scores = sklearn.learning_curve.learning_curve(classifier, training_x,
                                                                                   training_y, cv=split_iterator,
                                                                                   train_sizes=numpy.linspace(.1, 1.,
                                                                                                              10),
                                                                                   verbose=0)

    training_means = train_scores.mean(axis=1)
    training_std = train_scores.std(axis=1)

    test_means = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    print "{:.2f}% accuracy on test +/- {:.2f}".format(100. * test_means[-1], 100. * test_std[-1])

    plt.figure()
    plt.title("Random Forest Classifier")
    plt.xlabel("Training size")
    plt.ylabel("Accuracy")
    plt.ylim((0, 1.01))
    plt.grid()

    plt.plot(train_sizes, training_means, "o-", color="b", label="Training")
    plt.plot(train_sizes, test_means, "o-", color="r", label="Testing")
    plt.legend(loc="best")

    plt.fill_between(train_sizes, training_means - training_std, training_means + training_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, test_means - test_std, test_means + test_std, alpha=0.1, color="r")

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def print_tuning_scores(tuned_estimator, reverse=True):
    """Show the cross-validation scores and hyperparamters from a grid or random search"""
    for test in sorted(tuned_estimator.grid_scores_, key=itemgetter(1), reverse=reverse):
        print "Validation score {:.2f} +/- {:.2f}, Hyperparams {}".format(100. * test.mean_validation_score,
                                                                          100. * test.cv_validation_scores.std(),
                                                                          test.parameters)


def print_feature_importances(columns, tuned_classifier):
    paired_features = zip(columns, tuned_classifier.best_estimator_.feature_importances_)
    print "Feature importances"
    for feature_name, importance in sorted(paired_features, key=itemgetter(1), reverse=True):
        print "\t{:20s}: {}".format(feature_name, importance)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="CSV of possible features")
    parser.add_argument("output", help="Image file of learning curve")
    return parser.parse_args()


def main():
    args = parse_args()

    data = pandas.read_csv(args.input, header=0)

    X = data.values[:, :-1]
    y = data.values[:, -1]

    split_iterator = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=10, random_state=4)

    forest = sklearn.ensemble.RandomForestClassifier(10)

    forest_hyperparameters = {
        "n_estimators": [10, 100],
        "min_samples_split": [100, 200],
        "min_samples_leaf": [10, 20, 30]
    }

    grid_search = sklearn.grid_search.GridSearchCV(forest, forest_hyperparameters, n_jobs=-1, cv=split_iterator)
    grid_search.fit(X, y)

    print "Random forest"
    print_tuning_scores(grid_search)
    print_feature_importances(data.drop("IsBlueWinner", axis=1).columns, grid_search)

    logistic = sklearn.linear_model.LogisticRegression()

    logistic_hyperparameters = {
        "C": [0.01, 0.1, 1., 10.],
        "penalty": ["l2"]
    }
    grid_search = sklearn.grid_search.GridSearchCV(logistic, logistic_hyperparameters, n_jobs=-1, cv=split_iterator)
    grid_search.fit(X, y)

    print "Logistic regression"
    print_tuning_scores(grid_search)

    gradient_boosting = sklearn.ensemble.GradientBoostingClassifier()

    gbc_hyperparameters = {
        "learning_rate": [0.5, 1.],
        "max_depth": [3, 5, 10],
        # "min_samples_leaf": [1, 10, 20],
        # "subsample": [0.8, 0.9, 1.]
    }

    grid_search = sklearn.grid_search.GridSearchCV(gradient_boosting, gbc_hyperparameters, n_jobs=-1, cv=split_iterator)
    grid_search.fit(X, y)

    print "Gradient boosting classifier"
    print_tuning_scores(grid_search)


if __name__ == "__main__":
    sys.exit(main())