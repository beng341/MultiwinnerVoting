import torch
import torch.nn as nn
import math
import itertools

class MajorityWinnerLoss(nn.Module):
    def __init__(self):
        super(MajorityWinnerLoss, self).__init__()

    def all_majority_committees(self, rank_counts, k, batch_size=64):

        m = int(math.sqrt(len(rank_counts)))
        # rank_counts = torch.detach().numpy().array(rank_counts)

        rank_counts = rank_counts.view(m, m)
        half_col_sums = rank_counts.sum(axis=0) / 2

        majority_winner = [i for i in range(m) if rank_counts[i][0] > half_col_sums[i]]

        if len(majority_winner) > 1:
            ValueError("Found more than 1 majority winner")
        if len(majority_winner) == 0:
            return None

        # majority_winner = majority_winner[0]

        possible_other_committee_members = list(set(range(m)) - set(majority_winner))
        num_required = k-1
        majority_winner = majority_winner[0]

        all_valid_committees = []
        for combo in itertools.combinations(possible_other_committee_members, num_required):
            committee = [0] * m
            committee[majority_winner] = 1
            for idx in combo:
                committee[idx] = 1
            all_valid_committees.append(committee)

        return torch.tensor(all_valid_committees, dtype=torch.float32)


    def forward(self, outputs, rank_counts, k=2):
        """
        Good function names are hard :/ Largely made by ChatGPT.
        Find the highest k indices in the outputs and the targets.
        Count how many people ranked that candidate first for both the output and the target.
        Return absolute difference of the two scores.
        Should be minimized when the candidates with most first place rankings are elected.

        Just something that gives a non-binary value that might actually teach a network.
        :param outputs:
        :param rank_counts:
        :return:
        """

        distances_across_batch = []
        for idx in range(len(outputs)):
            # generate list of all possible valid (majority-satisfying) committees for current index value of batch
            rc = rank_counts[idx]
            valid_committees = self.all_majority_committees(rc, k=k)
            if valid_committees is None:
                all_distances = torch.abs(outputs[idx] - outputs[idx])
                min_distance = torch.min(all_distances)
            else:
                all_distances = torch.abs(outputs[idx] - valid_committees)
                all_distances = torch.sum(all_distances, dim=1)
                min_distance = torch.min(all_distances)

            distances_across_batch.append(min_distance)

        distances_across_batch = torch.stack(distances_across_batch)
        loss = torch.mean(distances_across_batch)
        return loss