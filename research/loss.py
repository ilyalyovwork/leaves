import torch
from torch import nn
import numpy as np


class LossBinary:
    """
    Loss defined as (1 - \alpha) BCE - \alpha SoftJaccard
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def soft_jaccard(self, ground_truth, p_outputs):
        eps = 1e-15
        intersection = (ground_truth * p_outputs).sum()
        union = ground_truth.sum() + p_outputs.sum()

        return (intersection + eps) / (union - intersection + eps)

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            jaccard_target = (targets == 1).float()
            jaccard_output = torch.sigmoid(outputs)

            loss -= self.jaccard_weight * torch.log(self.soft_jaccard(jaccard_target, jaccard_output))

        return loss