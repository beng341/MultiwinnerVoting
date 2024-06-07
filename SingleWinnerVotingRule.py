import numpy as np
import tensorflow as tf
from utils import ml_utils
import os


class SingleWinnerVotingRule:

    def __init__(self, num_candidates, config, **kwargs):
        """
        In future, args should be able to contain data specifying a network structure.
        :param num_candidates:
        :param args:
        """
        self.num_candidates = num_candidates
        self.experiment = kwargs["experiment"]
        self.feature_column = config["feature_column"]
        self.target_column = config["target_column"]
        self.tied_target_column = config["tied_target_column"]

        self.num_hidden_layers = config["hidden_layers"]
        self.nodes_per_layer = config["hidden_nodes"]

        self.config = config
        self.train_df = None
        self.test_df = None

        self.num_inputs = kwargs["num_features"]

        self.model = None
        self.reset()

    def reset(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(self.num_inputs,)))
        for _ in range(self.num_hidden_layers):
            # self.model.add(tf.keras.layers.Dropout(0.2))
            self.model.add(tf.keras.layers.Dense(self.nodes_per_layer, activation="relu"))
            # self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(self.num_candidates))

        self.model.compile(optimizer='adam',
                           # loss='mean_absolute_error',
                           loss="mean_squared_error",
                           # loss="mean_squared_logarithmic_error",
                           # loss="poisson",
                           # loss="hinge",
                           # loss="binary_crossentropy",
                           # loss="categorical_crossentropy",
                           # loss="cosine_similarity",
                           # loss=tf.keras.losses.CategoricalCrossentropy(),
                           # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy']
                           )

    def rule_name(self):
        target = self.target_column.replace("-single_winner", "")
        return f"NeuralNetRule-{target}"

    def __str__(self):
        return f"NeuralNetRule(m={self.num_candidates})"

    def train(self):
        """
        Transform the given raw_profiles and normalized rank/score lists into correct format then train network on them.
        :param x: Input to the current voting rule
        :param y: Corresponding correct output to learn
        :return:
        """

        # stop training after improvement stops. Do this to allow training for many epochs when needed
        callback = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=self.config["min_delta_loss"], patience=10)

        # features = [eval(elem) for elem in self.train_df[self.feature_column].tolist()]
        features = ml_utils.features_from_column_names(self.train_df, self.feature_column)
        targets = self.train_df[self.target_column].tolist()

        x = np.asarray(features)
        # y = np.asarray(targets)

        # On a few rules, the mean over 30 trials is about 0.02 lower with this normalization on than off
        # (for Mallows preferences at least)
        # layer = tf.keras.layers.Normalization(axis=None)
        # layer.adapt(x)

        x_train = tf.convert_to_tensor(x, dtype=tf.float32)
        # y_train = tf.convert_to_tensor(y, dtype=tf.float32)
        y_train = tf.one_hot(targets, depth=self.num_candidates)

        # Fit data to model
        history = self.model.fit(x_train, y_train, epochs=self.config["epochs"], verbose=False, callbacks=[callback])
        return history

    def save_model(self, suffix="", base_path=None, verbose=False):
        out_folder = self.config["output_folder"]
        if not base_path:
            base_path = os.getcwd()
        path = os.path.join(base_path, f"{out_folder}/trained_networks")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"Save location: {path}/NN-{self.config['experiment_name']}-{suffix}.keras")
        self.model.save(f"{path}/NN-{self.config['experiment_name']}-{suffix}.keras")

    def has_scores(self):
        return False

    def has_ranks(self):
        return False

    def has_single_winner(self):
        return True

    def predict_winner(self, train=True):
        """
        Predict and return winners. Rule already knows where to find train/test data.
        :param train: List of full ordered ballots for each voter participating in the election
        :return:
        """
        if train:
            # features = [eval(elem) for elem in self.train_df[self.feature_column].tolist()]
            features = ml_utils.features_from_column_names(self.train_df, self.feature_column)
        else:
            # features = [eval(elem) for elem in self.test_df[self.feature_column].tolist()]
            features = ml_utils.features_from_column_names(self.test_df, self.feature_column)

        x = tf.convert_to_tensor(np.asarray(features), dtype=tf.float32)
        y = self.model.predict(x)
        y_pred = [np.argmax(y_i) for y_i in y]

        return y_pred
