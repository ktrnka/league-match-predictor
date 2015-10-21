from __future__ import unicode_literals
import sys
import argparse
import keras.models
import keras.layers.core
import keras.regularizers
import sklearn.metrics
import sklearn.base

"""
Wrappers around the scikit-learn classifiers
"""


class NnWrapper(sklearn.base.BaseEstimator):
    """Wrapper for Keras feed-forward neural network to enable things like grid search"""
    def __init__(self, hidden_layer_sizes=[100], dropout=0.5, show_accuracy=True):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.show_accuracy = show_accuracy

        self.model_ = None

    def fit(self, X, y, **kwargs):
        model = keras.models.Sequential()

        # hidden layers
        first = True
        for layer_size in kwargs.get("hidden_layer_sizes", self.hidden_layer_sizes):
            if first:
                model.add(keras.layers.core.Dense(output_dim=layer_size, input_dim=X.shape[1], init="glorot_uniform"))
                first = False
            else:
                model.add(keras.layers.core.Dense(output_dim=layer_size, init="glorot_uniform"))
            model.add(keras.layers.core.Activation("relu"))
            model.add(keras.layers.core.Dropout(kwargs.get("dropout", self.dropout)))

        # output layer
        model.add(keras.layers.core.Dense(output_dim=1, init="glorot_uniform"))
        model.add(keras.layers.core.Activation("relu"))

        model.compile(loss="mse", optimizer="adam", class_mode="binary")

        # minibatch followed by some full batch
        model.fit(X, y, nb_epoch=400, batch_size=1024, show_accuracy=self.show_accuracy)
        model.fit(X, y, nb_epoch=100, batch_size=X.shape[0], show_accuracy=self.show_accuracy)

        self.model_ = model

    def predict(self, X):
        return self.model_.predict_classes(X)

    def predict_proba(self, X):
        return self.model_.predict(X)

    def score(self, X, y):
        return sklearn.metrics.accuracy_score(y, self.predict(X))

def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    sys.exit(main())