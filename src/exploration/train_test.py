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
import re

def learning_curve(training_x, training_y, filename, classifier):
    """Make a learning graph and save it"""
    split_iterator = sklearn.cross_validation.StratifiedShuffleSplit(training_y, n_iter=10, random_state=4)
    train_sizes, train_scores, test_scores = sklearn.learning_curve.learning_curve(classifier, training_x,
                                                                                   training_y, cv=split_iterator,
                                                                                   train_sizes=numpy.linspace(.1, 1., 10),
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="CSV of possible features")
    parser.add_argument("output", help="Image file of learning curve")
    return parser.parse_args()


def main():
    args = parse_args()

    data = pandas.read_csv(args.input, header=0)

    X = data.values[:,:-1]
    y = data.values[:,-1]

    forest = sklearn.ensemble.RandomForestClassifier(10, min_samples_split=50)
    learning_curve(X, y, args.output, forest)



if __name__ == "__main__":
    sys.exit(main())