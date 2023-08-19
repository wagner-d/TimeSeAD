from typing import Tuple, Iterator

import torch.nn
from torch.nn import Parameter


class BaseModel(torch.nn.Module):
    def grouped_parameters(self) -> Tuple[Iterator[Parameter], ...]:
        return self.parameters(),

from . import baselines
from . import common
from . import generative
from . import layers
from . import other
from . import prediction
from . import reconstruction
