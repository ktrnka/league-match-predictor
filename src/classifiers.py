from __future__ import unicode_literals
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
    def __init__(self, hidden_layer_sizes=(100,), dropout=0.5, show_accuracy=True, batch_spec=((400, 1024), (100, -1)), activation="relu"):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.show_accuracy = show_accuracy
        self.batch_spec = batch_spec
        self.activation = activation

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
            model.add(keras.layers.core.Activation(kwargs.get("activation", self.activation)))
            model.add(keras.layers.core.Dropout(kwargs.get("dropout", self.dropout)))

        # output layer
        model.add(keras.layers.core.Dense(output_dim=1, init="glorot_uniform"))
        model.add(keras.layers.core.Activation(kwargs.get("activation", self.activation)))

        model.compile(loss="mse", optimizer="adam", class_mode="binary")

        # batches as per configuration
        for num_iterations, batch_size in kwargs.get("batch_spec", self.batch_spec):
            if batch_size < 0:
                batch_size = X.shape[0]
            if num_iterations > 0:
                model.fit(X, y, nb_epoch=num_iterations, batch_size=batch_size, show_accuracy=self.show_accuracy)

        self.model_ = model

    def predict(self, X):
        return self.model_.predict_classes(X)

    def predict_proba(self, X):
        return self.model_.predict(X)

    def score(self, X, y):
        return sklearn.metrics.accuracy_score(y, self.predict(X))

    @staticmethod
    def generate_batch_params(mini_batch_iter, total_epochs=200, mini_batch_size=1024):
        for mini_batch_epochs in mini_batch_iter:
            assert mini_batch_epochs <= total_epochs
            yield ((mini_batch_epochs, mini_batch_size), (total_epochs - mini_batch_epochs, -1))
