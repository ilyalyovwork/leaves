import numpy as np
from torch import nn
import torch
from metrics import batch_jaccard

def validation(model, criterion, valid_loader, device):
    with torch.no_grad():
        model.eval()
        losses = []

        jaccard = []

        for inputs, targets in valid_loader:
            inputs = inputs.view(inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3],
                                 inputs.shape[4])
            inputs = inputs.to(device)
            targets = targets.view(targets.shape[0] * targets.shape[1], targets.shape[2], targets.shape[3],
                                       targets.shape[4])
            #targets = targets.to(device)
            outputs = model(inputs).cpu()
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            jaccard += batch_jaccard(targets, (outputs > 0).float())

        valid_loss = np.mean(losses)  # type: float

        valid_jaccard = np.mean(jaccard).astype(np.float64)

        print('Valid loss: {:.5f}, jaccard: {:.5f}'.format(valid_loss, valid_jaccard))
        metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard}
        return metrics