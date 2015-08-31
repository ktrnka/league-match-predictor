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

N_JOBS = 3

def learning_curve(training_x, training_y, filename, classifier, classifier_name):
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
    plt.title(classifier_name)
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


def print_feature_importances(columns, classifier):
    """Show feature importances for a classifier that supports them like random forest or gradient boosting"""
    paired_features = zip(columns, classifier.feature_importances_)
    field_width = unicode(max(len(c) for c in columns))
    format_string = "\t{:" + field_width + "s}: {}"
    print "Feature importances"
    for feature_name, importance in sorted(paired_features, key=itemgetter(1), reverse=True):
        print format_string.format(feature_name, importance)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forest", default=False, action="store_true", help="Experiments with random forests")
    parser.add_argument("--logistic", default=False, action="store_true", help="Experiments with logistic regression")
    parser.add_argument("--xg", default=False, action="store_true", help="Experiments with gradient boosting trees")
    parser.add_argument("--elastic", default=False, action="store_true", help="Experiments with elastic nets")
    parser.add_argument("--learning-curve", default=None, help="Generate a learning curve and save to file")
    parser.add_argument("--decision-tree", default=False, action="store_true", help="Experiments with decision trees")
    parser.add_argument("input", help="CSV of possible features")
    return parser.parse_args()


def random_forest(X, y, data, split_iterator):
    forest = sklearn.ensemble.RandomForestClassifier(10)
    hyperparameter_space = {
        "n_estimators": [100],
        "min_samples_split": [50],
        "min_samples_leaf": [5]
    }

    grid_search = sklearn.grid_search.GridSearchCV(forest, hyperparameter_space, n_jobs=N_JOBS, cv=split_iterator)
    grid_search.fit(X, y)

    print "Random forest"
    print_tuning_scores(grid_search)
    print_feature_importances(data.drop("IsBlueWinner", axis=1).columns, grid_search.best_estimator_)


def decision_tree(X, y, data, split_iterator):
    tree = sklearn.tree.DecisionTreeClassifier()

    scores = sklearn.cross_validation.cross_val_score(tree, X, y, cv=split_iterator)

    print "Decision tree: {:.2f}% +/- {:.2f}%".format(100. * scores.mean(), 100. * scores.std())

    tree.fit(X, y)
    sklearn.tree.export_graphviz(tree, "decision_tree.dot", feature_names=data.drop("IsBlueWinner", axis=1).columns)


def print_logistic_regression_feature_importances(column_names, classifier):
    parameters = classifier.coef_[0,:]
    paired = zip(column_names, parameters)
    field_width = unicode(max(len(c) for c in column_names))
    format_string = "\t{:" + field_width + "s}: {}"
    print "Feature weights in logistic regression"
    for name, weight in sorted(paired, key=itemgetter(1), reverse=True):
        print format_string.format(name, weight)


def logistic_regression(X, y, data, split_iterator, pca=False):
    logistic = sklearn.linear_model.LogisticRegression()

    # fast grid search
    # logistic_search = sklearn.linear_model.LogisticRegressionCV()

    # old grid search
    hyperparameter_space = {
        "C": [0.1, 0.5, 1., 2, 4],
        "penalty": ["l2"]
    }

    grid_search = sklearn.grid_search.GridSearchCV(logistic, hyperparameter_space, n_jobs=N_JOBS, cv=split_iterator)
    grid_search.fit(X, y)

    print "Logistic regression"
    print_tuning_scores(grid_search)

    # with feature scaling, mostly useful to get reasonable feature importances
    X_scaled = sklearn.preprocessing.scale(X)
    grid_search = sklearn.grid_search.GridSearchCV(logistic, hyperparameter_space, n_jobs=N_JOBS, cv=split_iterator)
    grid_search.fit(X_scaled, y)

    print "Logistic regression with feature scaling"
    print_tuning_scores(grid_search)
    print_logistic_regression_feature_importances(data.drop("IsBlueWinner", axis=1).columns, grid_search.best_estimator_)

    # with PCA (slow)
    if pca:
        num_components = 0.99
        pca = sklearn.decomposition.PCA(n_components=num_components, copy=True, whiten=False)
        X_pca = pca.fit_transform(X)

        grid_search = sklearn.grid_search.GridSearchCV(logistic, hyperparameter_space, n_jobs=N_JOBS, cv=split_iterator)
        grid_search.fit(X_pca, y)

        print "Logistic regression with PCA at {} components".format(num_components)
        print_tuning_scores(grid_search)


def elastic_net(X, y, split_iterator):
    # note that GridSearchCV doesn't work with ElasticNet; need to use ElasticNetCV to select alpha and such
    grid_search = sklearn.linear_model.ElasticNetCV(n_jobs=N_JOBS, cv=split_iterator, n_alphas=100)
    grid_search.fit(X, y)

    print "Elastic net scores"

    for i, alpha in enumerate(grid_search.alphas_):
        mse = grid_search.mse_path_[i, :]

        print "\tAlpha {:.3f} = {:.2f}% +/- {:.2f}%".format(alpha, 100 * (1 - mse.mean()), 100 * mse.std())


def gradient_boosting_exp(X, y, data, split_iterator):
    gradient_boosting = sklearn.ensemble.GradientBoostingClassifier()
    hyperparameter_space = {
        "learning_rate": [0.75, 0.9, 1.],
        "min_samples_leaf": [20]
    }

    grid_search = sklearn.grid_search.GridSearchCV(gradient_boosting, hyperparameter_space, n_jobs=N_JOBS, cv=split_iterator,
                                                   verbose=1)
    grid_search.fit(X, y)

    print "Gradient boosting classifier"
    print_tuning_scores(grid_search)
    print_feature_importances(data.drop("IsBlueWinner", axis=1).columns, grid_search.best_estimator_)


def dataframe_to_ndarrays(data):
    """Convert the Pandas DataFrame into X and y numpy.ndarray for sklearn"""
    assert isinstance(data, pandas.DataFrame)
    X = data.drop("IsBlueWinner", axis=1).values
    y = data.IsBlueWinner.values
    return X, y


def merge_roles(data, team, suffix, include_log_sum=False, include_sum=True, include_min=True, include_max=True):
    """Merge the 5 positions for the specified team and feature suffix"""
    cols = [col for col in data.columns if team in col and col.endswith(suffix)]
    assert len(cols) == 5

    if include_sum:
        data[team + suffix + "_Sum"] = data[cols].sum(axis=1)

    if include_log_sum:
        data[team + suffix + "_LogSum"] = numpy.log(data[cols].sum(axis=1) + 1)

    if include_min:
        data[team + suffix + "_Min"] = data[cols].min(axis=1)

    if include_max:
        data[team + suffix + "_Max"] = data[cols].max(axis=1)

    return data.drop(cols, axis=1)


def make_diff_feature(data, feature_suffix, remove_original=True):
    """Make a Red_X - Blue_X feature"""
    data["Delta" + feature_suffix] = data["Blue" + feature_suffix] - data["Red" + feature_suffix]

    if remove_original:
        return data.drop(["Blue" + feature_suffix, "Red" + feature_suffix], axis=1)
    else:
        return data


def preprocess_champions(data, include_features):
    """Convert Blue_1_Ahri, Blue_2_Ahri, etc to Blue_Ahri"""
    if include_features:
        for team in ["Blue", "Red"]:
            cols = [col for col in data.columns if team in col and "Champ" in col]
            indicator_dfs = [pandas.get_dummies(data[col], prefix=team) for col in cols]
            merged = reduce(lambda a, b: a.combineAdd(b), indicator_dfs[1:], indicator_dfs[0])
            data = pandas.concat((data.drop(cols, axis=1), merged), axis=1)
    else:
        data = data.drop([c for c in data.columns if "Champ" in c], axis=1)

    return data


def preprocess_summoner_spells(data, include_features):
    """Convert Blue_1_Flash (0-1), Blue_2_Flash (0-1), etc to Blue_Flash (0-5)"""
    if include_features:
        for team in ["Blue", "Red"]:
            cols = [col for col in data.columns if team in col and "Spell" in col]
            indicator_dfs = [pandas.get_dummies(data[col], prefix="{}_Summoners".format(team)) for col in cols]
            merged = reduce(lambda a, b: a.combineAdd(b), indicator_dfs[1:], indicator_dfs[0])
            data = pandas.concat((data.drop(cols, axis=1), merged), axis=1)
    else:
        data = data.drop([c for c in data.columns if "Spell" in c], axis=1)

    return data

def preprocess_damage_types(data, include_qcut_features):
    """Add damage type features for each quartile. Useful for logistic regression."""
    if include_qcut_features:
        for col in [c for c in data.columns if "Damage" in c]:
            data[col + "_qcut5"] = pandas.qcut(data[col], 5)

    return data


def drop_bad_features(data):
    """Drop features that lead to overfitting and such"""
    # MatchId would allow overfitting for no benefit
    # GameVersion overfit
    bad_cols = ["MatchId", "GameVersion"]

    # streak features lead to slight overfitting
    bad_cols.extend([c for c in data.columns if "Streak" in c])

    # drop the per champ*summoners win rates, overfit a little bit
    bad_cols.extend([c for c in data.columns if "champion summoners recent" in c])

    return data.drop(bad_cols, axis=1)


def preprocess_features(data, show_example=False):
    print "Before preprocessing"
    data.info()
    old_columns = sorted(data.columns)
    print "Columns: " + ", ".join(old_columns)

    data = drop_bad_features(data)

    data = preprocess_damage_types(data, False)

    data = preprocess_champions(data, False)

    data = preprocess_summoner_spells(data, False)

    # merge win rates and such across the team
    for team in ["Blue", "Red"]:
        # player-specific games played
        data = merge_roles(data, team, "_NumGames(player champion season)", include_log_sum=True, include_sum=False, include_min=False)
        data = merge_roles(data, team, "_NumGames(player season)", include_log_sum=True, include_sum=False, include_min=False)

        # player-specific win rates
        data = merge_roles(data, team, "_WinRate(player champion season)")
        data = merge_roles(data, team, "_WinRate(player season)")

        # win rates and play rates from the general population
        data = merge_roles(data, team, "_WinRate(champion season)", include_max=False, include_min=False)
        data = merge_roles(data, team, "_PlayRate(champion season)", include_max=False, include_min=False)
        data = merge_roles(data, team, "_WinRate(champion recent)", include_min=False)
        data = merge_roles(data, team, "_WinRate(champion version recent)", include_min=False)

        data[team + "_Combined_WinRateSum_PlayedLogSum(player champion season)"] = data[team + "_WinRate(player champion season)_Sum"] * data[team + "_NumGames(player champion season)_LogSum"]
        data[team + "_Combined_WinRateSum_PlayedLogSum(player season)"] = data[team + "_WinRate(player season)_Sum"] * data[team + "_NumGames(player season)_LogSum"]
        data[team + "_Combined_WinRateSum_PlayedLogSum(cvr ps)"] = data[team + "_WinRate(champion version recent)_Sum"] * data[team + "_NumGames(player season)_LogSum"]

    # a few diff features
    for feature_suffix in ["_WinRate(champion version recent)", "_WinRate(champion recent)", "_WinRate(champion season)", "_WinRate(player season)", "_PlayRate(champion season)"]:
        data["Delta" + feature_suffix + "_Sum"] = data["Blue" + feature_suffix + "_Sum"] - data["Red" + feature_suffix + "_Sum"]
        data = data.drop(["Blue" + feature_suffix + "_Sum", "Red" + feature_suffix + "_Sum"], axis=1)


    data = make_diff_feature(data, "_Combined_WinRateSum_PlayedLogSum(player champion season)")
    data = make_diff_feature(data, "_Combined_WinRateSum_PlayedLogSum(player season)")
    data = make_diff_feature(data, "_Combined_WinRateSum_PlayedLogSum(cvr ps)")

    data = pandas.get_dummies(data)

    print "After preprocessing"
    data.info()
    print data.describe()
    print "Columns: " + ", ".join(sorted(data.columns))
    print "Number of columns changed from {} -> {}".format(len(old_columns), len(data.columns))

    if show_example:
        print "Example data"
        for name, value in zip(data.columns, data.values[0, :]):
            print "{}: {}".format(name, value)

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

    if args.decision_tree:
        decision_tree(X, y, data, cross_val_splits)

    if args.learning_curve:
        learning_curve(X, y, args.learning_curve, sklearn.ensemble.RandomForestClassifier(100), "Random Forest Classifier")

    if args.forest:
        random_forest(X, y, data, cross_val_splits)

    if args.logistic:
        logistic_regression(X, y, data, cross_val_splits)

    if args.elastic:
        elastic_net(X, y, cross_val_splits)

    if args.xg:
        gradient_boosting_exp(X, y, data, cross_val_splits)


if __name__ == "__main__":
    sys.exit(main())