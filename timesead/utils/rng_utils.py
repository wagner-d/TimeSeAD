import random
import numpy as np
import torch


def set_seed(seed : int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class RandomBatchSampler(object):

    def __init__(self, dataset : torch.utils.data.Dataset, batch_size : int = 1):

        super(RandomBatchSampler, self).__init__()

        self.dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def sample(self):

        return next(iter(self.dataset))
