import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class MixUp:
    '''
    Including data augmentation and the calculation of mixup loss

    Args:
    sampling_method: 1 => beta  2 => uniformly sampled
    device: current device (cuda or cpu)
    '''
    def __init__(self, sampling_method, device):
        assert sampling_method in [1, 2], 'Sampling method can only be 1 or 2'
        self.sampling_method = sampling_method
        self.device = device

    def mixup_data(self, x, y, alpha=1.0, lambda_range=(0.1, 0.9)):
        '''
        Implement data augmentation for a batch of dataset using two sampling methods
        
        Args:
        x: input data that need to be mixuped
        y: labels
        alpha: the shape parameters of beta distribution
        lambda_range: tuple variable, the range of shape parameters of uniform distribution

        Returns:
        mixed_x: the results of mixup data
        y_a: ground truth labels
        y_b: labels which are out-of-order
        lam: the ratio of data and out-of-ordered data
        '''
        if self.sampling_method == 1:
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1
        elif self.sampling_method == 2:
            lam = np.random.uniform(lambda_range[0], lambda_range[1])

        batch_size = x.size()[0]
        indices = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[indices]
        y_a, y_b = y, y[indices]

        return mixed_x, y_a, y_b, lam


    def mixup_loss(self, pred, y_a, y_b, lam):
        '''
        Calculate loss for mixup method

        Args:
        pred: predicted value from model
        y_a, y_b, lam: same as the output of 'mixup_data' function

        Returns:
        loss: loss of training process using data augmentation
        '''
        criterion = nn.CrossEntropyLoss()
        loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
        return loss


  
