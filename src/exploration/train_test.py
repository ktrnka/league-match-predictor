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
import sklearn.decomposition
import sklearn.preprocessing
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

def convert_to_indicators(data, column, drop=True, min_values=1):
    """Create indicator features for a column, add them to the data, and remove the original field"""
    dummies = pandas.get_dummies(data[column], column, dummy_na=True)
    for col in dummies.columns:
        if dummies[col].sum() < min_values:
            print "Column {} has only {} values, dropping".format(col, int(dummies[col].sum()))
        else:
            data[col] = dummies[col]

    if drop:
        data.drop(column, axis=1, inplace=True)


def print_feature_importances(columns, tuned_classifier):
    paired_features = zip(columns, tuned_classifier.best_estimator_.feature_importances_)
    print "Feature importances"
    for feature_name, importance in sorted(paired_features, key=itemgetter(1), reverse=True):
        print "\t{:20s}: {}".format(feature_name, importance)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forest", default=False, action="store_true", help="Experiments with random forests")
    parser.add_argument("--logistic", default=False, action="store_true", help="Experiments with logistic regression")
    parser.add_argument("--xg", default=False, action="store_true", help="Experiments with gradient boosting trees")
    parser.add_argument("--elastic", default=False, action="store_true", help="Experiments with elastic nets")
    parser.add_argument("input", help="CSV of possible features")
    return parser.parse_args()


def random_forest(X, y, data, split_iterator):
    forest = sklearn.ensemble.RandomForestClassifier(10)
    hyperparameter_space = {
        "n_estimators": [100],
        "min_samples_split": [100],
        "min_samples_leaf": [15]
    }

    grid_search = sklearn.grid_search.GridSearchCV(forest, hyperparameter_space, n_jobs=3, cv=split_iterator)
    grid_search.fit(X, y)

    print "Random forest"
    print_tuning_scores(grid_search)
    print_feature_importances(data.drop("IsBlueWinner", axis=1).columns, grid_search)


def print_logistic_regression_feature_importances(column_names, classifier):
    parameters = classifier.coef_[0,:]
    paired = zip(column_names, parameters)

    print "Feature weights in logistic regression"
    for name, weight in sorted(paired, key=itemgetter(1), reverse=True):
        print "\t{:20s}: {}".format(name, weight)


def logistic_regression(X, y, data, split_iterator, pca=False):
    logistic = sklearn.linear_model.LogisticRegression()
    hyperparameter_space = {
        "C": [0.1, 0.5, 1., 2, 4],
        "penalty": ["l2"]
    }

    grid_search = sklearn.grid_search.GridSearchCV(logistic, hyperparameter_space, n_jobs=3, cv=split_iterator)
    grid_search.fit(X, y)

    print "Logistic regression"
    print_tuning_scores(grid_search)

    # The feature weights really aren't interpretable without also having the scale of the feature
    # print_logistic_regression_feature_importances(data.drop("IsBlueWinner", axis=1).columns, grid_search.best_estimator_)

    # with feature scaling
    X_scaled = sklearn.preprocessing.scale(X)
    grid_search = sklearn.grid_search.GridSearchCV(logistic, hyperparameter_space, n_jobs=3, cv=split_iterator)
    grid_search.fit(X_scaled, y)

    print "Logistic regression with feature scaling"
    print_tuning_scores(grid_search)
    print_logistic_regression_feature_importances(data.drop("IsBlueWinner", axis=1).columns, grid_search.best_estimator_)

    # with PCA (slow)
    if pca:
        num_components = 0.99
        pca = sklearn.decomposition.PCA(n_components=num_components, copy=True, whiten=False)
        X_pca = pca.fit_transform(X)

        grid_search = sklearn.grid_search.GridSearchCV(logistic, hyperparameter_space, n_jobs=3, cv=split_iterator)
        grid_search.fit(X_pca, y)

        print "Logistic regression with PCA at {} components".format(num_components)
        print_tuning_scores(grid_search)

def elastic_net(X, y):
    # note that GridSearchCV doesn't work with ElasticNet; need to use ElasticNetCV to select alpha and such
    train_X, test_X, train_y, test_y = sklearn.cross_validation.train_test_split(X, y, test_size=0.1, random_state=4)
    splits = sklearn.cross_validation.StratifiedShuffleSplit(train_y, 10)

    grid_search = sklearn.linear_model.ElasticNetCV(n_jobs=3, cv=splits, n_alphas=100)
    grid_search.fit(train_X, train_y)

    print "Elastic net on testing data", sklearn.metrics.accuracy_score(test_y.astype(int), grid_search.predict(test_X).astype(int))
    print "Elastic net on training data", sklearn.metrics.accuracy_score(train_y.astype(int), grid_search.predict(train_X).astype(int))



def gradient_boosting_exp(X, y, split_iterator):
    gradient_boosting = sklearn.ensemble.GradientBoostingClassifier()
    hyperparameter_space = {
        "learning_rate": [0.2],
        "max_depth": [3],
        "min_samples_leaf": [1, 10, 20],
        # "subsample": [0.8, 0.9, 1.]
    }

    grid_search = sklearn.grid_search.GridSearchCV(gradient_boosting, hyperparameter_space, n_jobs=3, cv=split_iterator,
                                                   verbose=1)
    grid_search.fit(X, y)

    print "Gradient boosting classifier"
    print_tuning_scores(grid_search)


def dataframe_to_ndarrays(data):
    """Convert the Pandas DataFrame into X and y numpy.ndarray for sklearn"""
    assert isinstance(data, pandas.DataFrame)
    X = data.drop("IsBlueWinner", axis=1).values
    y = data.IsBlueWinner.values
    return X, y


def merge_roles(data, team, suffix, include_log_sum=False):
    cols = [col for col in data.columns if team in col and col.endswith(suffix)]
    assert len(cols) == 5

    data[team + suffix + "_Sum"] = data[cols].sum(axis=1)
    if include_log_sum:
        data[team + suffix + "_LogSum"] = numpy.log(data[team + suffix + "_Sum"] + 1)
    data[team + suffix + "_Min"] = data[cols].min(axis=1)
    data[team + suffix + "_Max"] = data[cols].max(axis=1)
    return data.drop(cols, axis=1)


def preprocess_features(data):
    print "Before preprocessing"
    data.info()
    print "Columns: " + ", ".join(sorted(data.columns))

    data = data.drop(["GameVersion"], axis=1)

    for col in [c for c in data.columns if "Damage" in c]:
        #data[col] = pandas.qcut(data[col], 5)
        data[col + "_qcut5"] = pandas.qcut(data[col], 5)

    data = data.drop([c for c in data.columns if "Champ" in c], axis=1)
    # # convert the per-role champions into boolean indicators per side
    # # e.g., Blue_1_Ahri, Blue_2_Ahri get merged into Blue_Ahri
    # for team in ["Blue", "Red"]:
    #     cols = [col for col in data.columns if team in col and "Champ" in col]
    #     indicator_dfs = [pandas.get_dummies(data[col], prefix=team) for col in cols]
    #     merged = reduce(lambda a, b: a.combineAdd(b), indicator_dfs[1:], indicator_dfs[0])
    #     data = pandas.concat((data.drop(cols, axis=1), merged), axis=1)

    # convert the per-role summoner spells into sums per side
    # e.g., Blue_1_Spell_1, (0-1) Blue_2_Spell_2 (0-1), ... get merged into Blue_Flash (0-5)
    for team in ["Blue", "Red"]:
        cols = [col for col in data.columns if team in col and "Spell" in col]
        indicator_dfs = [pandas.get_dummies(data[col], prefix="{}_Summoners".format(team)) for col in cols]
        merged = reduce(lambda a, b: a.combineAdd(b), indicator_dfs[1:], indicator_dfs[0])
        data = pandas.concat((data.drop(cols, axis=1), merged), axis=1)

    # merge win rates and such across the team
    for team in ["Blue", "Red"]:
        # are they good at a weird champion?
        # for i in range(1, 6):
        #     data["{}_{}_DarkHorse".format(team, i)] = data["{}_{}_WinRate".format(team, i)] / data["{}_{}_GeneralPlayRate".format(team, i)]
        # data = merge_roles(data, team, "_DarkHorse")

        data = merge_roles(data, team, "_Played", include_log_sum=True)
        data = merge_roles(data, team, "_TotalPlayed", include_log_sum=True)

        data = merge_roles(data, team, "_GeneralPlayRate")
        data = merge_roles(data, team, "_WinRate")
        data = merge_roles(data, team, "_TotalWinRate")
        data = merge_roles(data, team, "_GeneralWinRate")

        data[team + "_Combined_WR_LP"] = data[team + "_WinRate_Sum"] * data[team + "_Played_LogSum"]

    data = pandas.get_dummies(data)

    # speeds up learning a little
    data = data.drop("Blue_Summoners_Unknown Red_Summoners_Unknown".split(), axis=1)

    print "After preprocessing"
    data.info()
    print data.describe()
    print "Columns: " + ", ".join(sorted(data.columns))

    return data


def check_data(X, y):
    """Simple checks to prevent a long runtime that's operating on bad data"""
    assert abs(y.mean() - 0.5) < 0.02
    assert X.shape[0] > 10000
    assert y.shape[0] > 10000
    assert X.shape[0] == y.shape[0]


def main():
    args = parse_args()

    data = pandas.read_csv(args.input, header=0)
    data = preprocess_features(data)

    X, y = dataframe_to_ndarrays(data)
    check_data(X, y)

    cross_val_splits = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=10, random_state=4)

    if args.forest:
        random_forest(X, y, data, cross_val_splits)

    if args.logistic:
        logistic_regression(X, y, data, cross_val_splits)

    if args.elastic:
        elastic_net(X, y)

    if args.xg:
        gradient_boosting_exp(X, y, cross_val_splits)


if __name__ == "__main__":
    sys.exit(main())