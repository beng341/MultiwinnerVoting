import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ignite.handlers import EarlyStopping
from utils import ml_utils
import os

class MultiWinnerVotingRule(nn.Module):

    def __init__(self, num_candidates, config, **kwargs):
        """
        In future, args should be able to contain data specifying a network structure.
        :param num_candidates:
        :param args:
        """
        super(MultiWinnerVotingRule, self).__init__()

        self.num_candidates = num_candidates
        self.experiment = kwargs["experiment"]
        self.feature_column = config["feature_column"]
        self.target_column = config["target_column"]
        #self.tied_target_column = config["tied_target_column"]

        self.num_hidden_layers = config["hidden_layers"]
        self.nodes_per_layer = config["hidden_nodes"]
        self.loss = config["loss"]

        self.config = config
        self.train_df = None
        self.test_df = None

        self.num_inputs = kwargs["num_features"]

        self.model = None
        self.reset()
    
    def forward(self, x):
        return self.model(x)

    def reset(self):
        layers = []

        layers.append(nn.Linear(self.num_inputs, self.nodes_per_layer))
        layers.append(nn.ReLU())

        for i in range(self.num_hidden_layers):
            layers.append(nn.Linear(self.nodes_per_layer, self.nodes_per_layer))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(self.nodes_per_layer, self.num_candidates))

        self.model = nn.Sequential(*layers)

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def rule_name(self):
        target = self.target_column.replace("-single_winner", "")
        return f"NeuralNetRule-{target}"

    def __str__(self):
        return f"NeuralNetRule(m={self.num_candidates})"

    def trainmodel(self):
        """
        Transform the given raw_profiles and normalized rank/score lists into correct format then train network on them.
        :param x: Input to the current voting rule
        :param y: Corresponding correct output to learn
        :return:
        """
        
        self.model.train()

        # features = [eval(elem) for elem in self.train_df[self.feature_column].tolist()]
        features = ml_utils.features_from_column_names(self.train_df, self.feature_column)
        targets = self.train_df[self.target_column].apply(eval).tolist()

        # x = np.asarray(features)
        # y = np.asarray(targets)

        # On a few rules, the mean over 30 trials is about 0.02 lower with this normalization on than off
        # (for Mallows preferences at least)
        # layer = tf.keras.layers.Normalization(axis=None)
        # layer.adapt(x)

        #x_train = tf.convert_to_tensor(x, dtype=tf.float32)
        x_train = torch.tensor(features, dtype=torch.float32)
        # y_train = tf.convert_to_tensor(y, dtype=tf.float32)
        y_train = torch.tensor(targets, dtype=torch.float32)

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Fit data to model
        train_losses = []
        avg_train_losses = []
        patience = 10
        num_epochs = self.config["epochs"]

        patience_counter = 0
        best_loss = float('inf')

        for epoch in range(num_epochs):
            epoch_loss = 0
            for i, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_train_losses.append(avg_epoch_loss)

            print(f'Epoch {epoch + 1}, Training loss: {avg_epoch_loss:.4f}')

            if avg_epoch_loss < best_loss - self.config["min_delta_loss"]:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Stopping early")
                    break
        return avg_train_losses

        

    def save_model(self, suffix="", base_path=None, verbose=False):
        out_folder = self.config["output_folder"]
        if not base_path:
            base_path = os.getcwd()
        path = os.path.join(base_path, f"{out_folder}/trained_networks")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"Save location: {path}/NN-{self.config['experiment_name']}-{suffix}.pt")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_candidates': self.num_candidates,
            'config': self.config,
            'kwargs': {
                'experiment': self.experiment,
                'num_features': self.num_inputs
            }
        }

        torch.save(checkpoint, f"{path}/NN-{self.config['experiment_name']}-{suffix}.pt")

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
        self.model.eval()

        if train:
            # features = [eval(elem) for elem in self.train_df[self.feature_column].tolist()]
            features = ml_utils.features_from_column_names(self.train_df, self.feature_column)
        else:
            # features = [eval(elem) for elem in self.test_df[self.feature_column].tolist()]
            features = ml_utils.features_from_column_names(self.test_df, self.feature_column)

        x = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            y = self.model(x)
        
        y_pred = [torch.argmax(y_i).item() for y_i in y]

        return y_pred
