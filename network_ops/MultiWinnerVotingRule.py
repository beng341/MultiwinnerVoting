import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ignite.handlers import EarlyStopping
from utils import ml_utils
import os
from utils import data_utils as du
from utils.losses import condorcet_winner_loss as cwl
from utils.losses import majority_loss as ml
from utils.losses import TopK_Differentiable as tk
from torch.autograd import grad
import torch.nn.functional as F

 
def softmax_w(x, w, t=0.0001):
    logw = np.log(w + 1e-12)  # use 1E-12 to prevent numeric problem
    x = (x + logw) / t
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
 
def top_k(x, k):
    y = np.zeros(len(x))
    for i in range(k):
        x1 = softmax_w(x, w=(1 - y))
        y = y + x1
    return y
 


class MultiWinnerVotingRule(nn.Module):

    def __init__(self, num_candidates, num_voters, num_winners, config, **kwargs):
        """
        In future, args should be able to contain data specifying a network structure.
        :param num_candidates:
        :param args:
        """
        super(MultiWinnerVotingRule, self).__init__()

        self.num_candidates = num_candidates
        self.num_voters = num_voters
        self.num_winners = num_winners
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
        #self.criterion = self.loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

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

        features = ml_utils.features_from_column_names(self.train_df, self.feature_column)
        targets = self.train_df[self.target_column].apply(eval).tolist()
        rank_matrix = self.train_df["rank_matrix"].apply(eval).tolist()
        cand_pairs = self.train_df["candidate_pairs"].apply(eval).tolist()


        x_train = torch.tensor(features, dtype=torch.float32, requires_grad=True)
        y_train = torch.tensor(targets, dtype=torch.float32, requires_grad=True)
        rank_matrix = torch.tensor(rank_matrix, dtype=torch.float32, requires_grad=True)
        cand_pairs = torch.tensor(cand_pairs, dtype=torch.float32, requires_grad=True)

        train_dataset = TensorDataset(x_train, cand_pairs, rank_matrix, y_train)#, rank_matrix, cand_pairs)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=0)


        #criterion = self.loss()

        # Fit data to model
        avg_train_losses = []
        patience = 20
        num_epochs = self.config["epochs"]

        patience_counter = 0
        best_loss = float('inf')

        for epoch in range(num_epochs):
            epoch_loss = 0
            maj_winner_loss = 0
            maj_loser_loss = 0
            cond_win_loss = 0

            for i, (data, cp, rm, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                data = data.detach().requires_grad_(True)
                cp = cp.detach().requires_grad_(True)
                
                output = self.forward(data)
                target = target.float()

                #for name, param in self.named_parameters():
                #    print(name, param.requires_grad)

                #loss_calculate(output, cp, rc, n_winners, num_candidates, num_voters)
                #loss = cwl.loss_calculate(output, cp, rm, self.n_winners[0], self.num_candidates, self.num_voters)#cwl.condorcet_loss(output, cp, self.n_winners[0], self.num_candidates)
                #loss = cwl.loss_calculate(output, cp, rm, self.n_winners[0], self.num_candidates, self.num_voters)
                #loss = nn.CrossEntropyLoss()(output, target)

                loss = self.loss(output, target)

                loss.backward()

                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                #for name, param in self.named_parameters():
                #    if param.grad is not None:
                #        print(f"{name} gradient: {param.grad.norm().item()}")
                #    else:
                #        print(f"{name} has no gradient")
                
                self.optimizer.step()

                #for name, param in self.model.named_parameters():
                #    if param.grad is not None:
                #        print(f"{name} gradient norm: {param.grad.norm()}")
                #    else:
                #        print(f"{name} has no gradient")
                #print('--------')
                
                #print(f"Output grad: {output.grad}")
                #print(f"CP grad: {cp.grad}")
                #print(f"Loss grad: {loss.grad}")
                #exit(1)


                # loss = main_loss + maj_win + maj_loser + cond_win

                #print(output.grad_fn)
                #print(cp.grad_fn)
                
                #print(cp.grad)
                #loss.backward()

                #print(c_indices.grad)
                #print(d_indices.grad)
                #print(data.grad)
                #print('')

                #self.optimizer.step()


                epoch_loss += loss.item()
                #maj_winner_loss += maj_win.item()
                # maj_loser_loss += maj_loser.item()
                # cond_win_loss += cond_win.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            #avg_maj_winner_epoch_loss = maj_winner_loss / len(train_loader)
            #avg_maj_loser_epoch_loss = maj_loser_loss / len(train_loader)
            #avg_cond_win_epoch_loss = cond_win_loss / len(train_loader)
            avg_train_losses.append(avg_epoch_loss)

            print(f'Epoch {epoch + 1}, Training loss: {avg_epoch_loss:.4f}, ')
                #f'Majority Winner Loss: {avg_maj_winner_epoch_loss:.4f}, '
                #f'Majority Loser Loss: {avg_maj_loser_epoch_loss:.4f}, '
                #f'Condorcet Winner Loss: {avg_cond_win_epoch_loss:.4f}')

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
            'n_winners': self.num_winners,
            'num_voters': self.num_voters,
            'config': self.config,
            'kwargs': {
                'experiment': self.experiment,
                'num_features': self.num_inputs
            }
        }

        torch.save(checkpoint, f"{path}/NN-{self.config['experiment_name']}-{suffix}.pt")
        # torch.save(self.model.state_dict(), f"{path}/NN-{self.config['experiment_name']}-{suffix}-STATEDICTONLYREMOVETHISSUFFIX.pt")


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
