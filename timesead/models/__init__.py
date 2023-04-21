from typing import Tuple, Iterator

import torch.nn
from torch.nn import Parameter


class BaseModel(torch.nn.Module):
    def grouped_parameters(self) -> Tuple[Iterator[Parameter], ...]:
        return self.parameters(),
