from itertools import chain
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import AnomalyDetector, RNN
from ...models import BaseModel
from ...optim.loss import Loss
from ...optim.trainer import Trainer
from ...utils.torch_utils import tensor2scalar


# Define the THOC Model:
class THOC(BaseModel):
    def __init__(
        self,
        input_size,
        hidden_sizes: Union[Sequence[int], int] = 128,
        n_hidden_layers: Optional[int] = 3,
        dilations: Union[Sequence[int], int] = [1, 2, 4],
        clusters_dims: Union[Sequence[int], int] = 6,
        tau: float = 100.0,
    ) -> None:
        super(THOC, self).__init__()

        if isinstance(hidden_sizes, int):
            if n_hidden_layers is None:
                hidden_sizes = [hidden_sizes]
            else:
                hidden_sizes = [hidden_sizes for _ in range(n_hidden_layers)]

        if isinstance(dilations, int):
            dilations = [dilations for _ in range(len(hidden_sizes))]
        elif len(dilations) != len(hidden_sizes):
            raise ValueError(
                "dilations must be int or list with length equals the number of hidden layers."
            )

        if isinstance(clusters_dims, int):
            clusters_dims = [clusters_dims for _ in range(len(hidden_sizes))]
        elif len(clusters_dims) != len(hidden_sizes):
            raise ValueError(
                "clusters_dims must be int or list with length equals the number of hidden layers."
            )

        self.dilations = dilations
        self.clusters_dims = clusters_dims
        self.tau = tau

        self.drnn = RNN("lstm", "s2as", input_size, hidden_sizes, dilation=dilations)

        # flag to see if centers are not yet initialized
        self._c_not_init = True

        self.centers = nn.ParameterList()
        self.transforms = nn.ModuleList()
        self.out_project = nn.ModuleList()
        self.join_f = nn.ModuleList()

        for l in range(len(hidden_sizes)):
            # init centers with Uniform[-1,1]
            self.centers.append(
                nn.Parameter(
                    (torch.rand(self.clusters_dims[l], hidden_sizes[l]) - 0.5) * 2
                )
            )

            # define fuse layers (equation 7)
            self.transforms.append(nn.Linear(hidden_sizes[l], hidden_sizes[l]))

            # define prediction layer (equation 12)
            self.out_project.append(nn.Linear(hidden_sizes[l], input_size, bias=False))

            # define join layer (equation 8)
            if l >= 1:
                self.join_f.append(
                    nn.Linear(hidden_sizes[l - 1] + hidden_sizes[l], hidden_sizes[l])
                )

    def grouped_parameters(self) -> Tuple[Iterator[nn.Parameter], ...]:
        # All parameters EXCEPT join_f are trained with same Optimizier
        params1 = chain(
            self.drnn.parameters(),
            self.centers.parameters(),
            self.transforms.parameters(),
            self.out_project.parameters(),
        )
        return (params1, self.join_f.parameters())

    def knn_init_centers(self, dl: torch.utils.data.DataLoader, num_batches: int):
        """
        Function that initiates the centers by Kmeans on the training set. Since taking the whole training data is to large, we only
                consider the first (shuffled) ``num_batches`` batches.

        dl (torch.utils.data.DataLoader): Dataloader from which we take the first ``num_batches`` batches.
            Ideally the dataloader should shuffle the batches.
        num_batches (int, optional): Number of batches to use for inititalization. If ``num_batches = 0`` the
            centers will not be initialized by Kmeans at all. Default is 20.
        """
        # Get current device of the model
        device = self.centers[0].device

        # Extract first 'num_batches' batches (dl should be shuffled)
        x_set = ()
        for i, ((x,), *_) in enumerate(dl):
            x = x.to(device)
            x_set += (x,)
            if i + 1 == num_batches:
                break
        x_set = torch.cat(x_set)

        # get hidden features
        with torch.no_grad():
            *_, hidden_f = self.forward(x_set)

        # Shape [bz*num_batches * seq_len * cluster_previous, hidden_size]
        hidden_f = [hid.reshape(-1, hid.shape[-1]).cpu().numpy() for hid in hidden_f]

        for i, hidden in enumerate(hidden_f):
            kmeans = KMeans(self.clusters_dims[i]).fit(hidden)
            self.centers[i].data = torch.tensor(kmeans.cluster_centers_, device=device)

        # Turn flag to True
        self._c_not_init = False

    @staticmethod
    def _get_similarity(f: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Implemenation of the similarity score function (Equation 6)

        inputs:
            f (torch.Tensor) of shape [bz, seq_len, previous_num_cluster, features]
            c (torch.Tensor) of shape [num_clusters, features]
        output:
            score (torch.Tensor) of shape [bz, seq_len, num_clusters, previous_num_cluster]
        """
        f_norm = F.normalize(f, dim=-1)
        c_norm = F.normalize(c, dim=-1)
        # Authors implementation:
        return 0.5 * (torch.einsum("bspf,cf -> bscp", f_norm, c_norm) - 1)
        # better in my opinion but just different scaling:
        # return torch.einsum('bspf,cf -> bscp', f_norm, c_norm)

    def _do_assignment(
        self, f: torch.Tensor, c: torch.Tensor, R_last: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implemenation of the assignment step (Equation 5 and 6)

        inputs:
            f (torch.Tensor) of shape [bz, seq_len, previous_num_cluster, features]
            c (torch.Tensor) of shape [num_clusters, features]
            R_last (torch.Tensor, optional) of shape [bz, seq_len, previous_num_cluster]
        output:
            P (torch.Tensor) of shape [bz, seq_len, num_clusters, previous_num_cluster]
            R (torch.Tensor) of shape [bz, seq_len, num_clusters]
        """
        alpha = 1 / self.tau

        # Legend: b = batch; s = seq; c = num_cluster;
        #         p = previous_num_cluster; f = features

        # Out shape: [b, s, c, p]
        sim_score = self._get_similarity(f, c)
        P = F.softmax(alpha * sim_score, dim=2)

        # Out shape: [b, s, c]
        if R_last is not None:
            R = torch.einsum("bscp, bsp -> bsc", P, R_last)
            R = F.softmax(R, dim=-1)
        else:
            R = F.softmax(P.squeeze(-1), dim=-1)

        return P, R

    @staticmethod
    def _update_f(
        f: torch.Tensor, P: torch.Tensor, transform: torch.nn.Linear
    ) -> torch.Tensor:
        """
        Implemenation of the update step (Equation 7)

        inputs:
            f (torch.Tensor) of shape [bz, seq_len, previous_num_cluster, features]
            P (torch.Tensor) of shape [bz, seq_len, num_cluster, previous_num_cluster]
            transform (torch.nn.Linear) a linear torch.nn.Module
        output:
            f_bar (torch.Tensor) of shape [bz, seq_len, num_cluster, features]
        """
        # Authors implementation (wrong in my opinion):
        # return torch.einsum('bscp, bspf -> bscf', F.softmax(P, dim=-1), torch.relu(transform(f)))
        # Correct following the paper is P without softmax: Out shape: [b, s, c, f]
        return torch.einsum("bscp, bspf -> bscf", P, torch.relu(transform(f)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x must be in shape: [batch_size, seq_len, num_features]
        # Run dilated RNN (Equation 4)
        drnn_outs = self.drnn(x)

        # Inits
        R_last = None
        f_bar = None
        hidden_f = []
        # Init number of previous cluster
        num_prev_clusters = 1

        for l, drnn_out in enumerate(drnn_outs):
            # Stretch drnn output for number of previous clusters
            f = drnn_out.unsqueeze(2).repeat(1, 1, num_prev_clusters, 1)
            # Join f from previous and current layer (Equ. 8)
            if f_bar is not None:
                f = self.join_f[l - 1](torch.cat([f_bar, f], dim=-1))
            # Run assignment (Equation 5 and 6)
            P, R_last = self._do_assignment(f, self.centers[l], R_last)
            # Update f (Equation 7)
            f_bar = self._update_f(f, P, self.transforms[l])
            # Update number of previous clusters
            num_prev_clusters = self.clusters_dims[l]
            # Save the hidden_features
            hidden_f.append(f)

        return f_bar, R_last, drnn_outs, hidden_f


class THOCLoss(Loss):
    def __init__(self, model: THOC, lambda_orth: float = 1.0, lambda_tss: float = 10.0):
        super(THOCLoss, self).__init__()

        self.model = model
        self.lambda_orth = lambda_orth
        self.lambda_tss = lambda_tss

    def thoc_loss(self, f_final: torch.Tensor, R_last: torch.Tensor) -> torch.Tensor:

        f_norm = F.normalize(f_final, dim=-1)
        c_norm = F.normalize(self.model.centers[-1], dim=-1)

        # Authors implementation:
        score = 0.5 * (1 - torch.einsum("bscf,cf -> bsc", f_norm, c_norm))
        # Correct following the paper in my opinion is, but this is just different scaling:
        # score = - torch.einsum('bscf,cf -> bsc', f_norm, c_norm)

        return (R_last * score).mean()

    def orth_loss(self):
        loss_orth = 0
        for c in self.model.centers:
            num_clust = c.shape[0]
            if num_clust > 1:
                # In my Opinion: We should normalize, NOT in THOC authors code
                c_norm = F.normalize(c, dim=-1)
                CCT = c_norm @ c_norm.T
                loss_orth += torch.mean(
                    (CCT - torch.eye(num_clust, device=c.device)) ** 2
                )

        return loss_orth / len(self.model.centers)

    def tss_loss(
        self,
        drnn_outs: List[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calclulates the time-series prediction error. (Equation 12)

        inputs:
            drnn_outs (torch.Tensor)
            x (torch.Tensor)
        """
        pred_loss = 0
        for i, drnn_out in enumerate(drnn_outs):
            prediction_i = self.model.out_project[i](drnn_out)
            pred_loss += F.mse_loss(
                prediction_i[:, : -self.model.dilations[i], :],
                target[:, self.model.dilations[i] :, :],
            )

        return pred_loss / len(drnn_outs)

    def forward(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: Tuple[torch.Tensor, ...] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        f, R, drnn_outs = predictions
        (x,) = targets

        # Calc losses
        thoc_loss = self.thoc_loss(f, R)
        orth_loss = self.lambda_orth * self.orth_loss()
        tss_loss = self.lambda_tss * self.tss_loss(drnn_outs, x)

        return thoc_loss, orth_loss, tss_loss


class THOCTrainer(Trainer):
    def __init__(
        self,
        *args,
        tau_decrease_steps: int = 5,
        tau_decrease_gamma: float = 2.0 / 3.0,
        init_centers_batches: int = 20,
        **kwargs,
    ):
        super(THOCTrainer, self).__init__(*args, **kwargs)

        self.tau_decrease_steps = tau_decrease_steps
        self.tau_decrease_gamma = tau_decrease_gamma
        self.init_centers_batches = init_centers_batches
    
    def validate_batch(self, network: torch.nn.Module, val_metrics: Dict[str, Callable],
                       b_inputs: Tuple[torch.Tensor, ...], b_targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> Dict[str, float]:
        (x,) = b_inputs
        f, R, drnn_outs, _ = network(x)
        res = (f, R, drnn_outs)

        batch_metrics = {}
        for m_name, m in val_metrics.items():
            if m_name == 'loss_0':
                thoc_l, orth_l, tss_l = m(res, b_targets, b_inputs, *args, **kwargs)
                batch_metrics['loss_0'] = tensor2scalar(thoc_l.detach().cpu()) * b_inputs[0].shape[self.batch_dimension]
                batch_metrics['loss_1'] = tensor2scalar((orth_l+tss_l).detach().cpu()) * b_inputs[0].shape[self.batch_dimension]

        return batch_metrics

    def train_batch(
        self,
        network: THOC,
        losses: List[Loss],
        optimizers: List[torch.optim.Optimizer],
        epoch: int,
        num_epochs: int,
        b_inputs: Tuple[torch.Tensor, ...],
        b_targets: Tuple[torch.Tensor, ...],
    ) -> List[float]:
        (x,) = b_inputs

        opt_all, opt_join = optimizers
        loss_all, *_ = losses

        # Forward pass
        f, R, drnn_outs, _ = network(x)

        thoc_l, orth_l, tss_l = loss_all((f, R, drnn_outs), b_targets)
        l = thoc_l + orth_l + tss_l

        # Backward pass
        opt_all.zero_grad(True)
        opt_join.zero_grad(True)
        l.backward()
        opt_all.step()
        opt_join.step()

        if epoch % self.tau_decrease_steps == 0 and epoch > 0:
            # Make assignment slowly converge to a hard one
            if network.tau > 1.0 / 300.0:
                network.tau *= self.tau_decrease_gamma

        return [thoc_l.item(), orth_l.item(), tss_l.item()]

    def train(self, network: THOC, *args, **kwargs):
        # init centers at beginning
        if network._c_not_init:
            network.knn_init_centers(self.train_iter, self.init_centers_batches)

        super(THOCTrainer, self).train(network, *args, **kwargs)

class THOCAnomalyDetector(AnomalyDetector):
    def __init__(self, model: THOC) -> None:
        super(THOCAnomalyDetector, self).__init__()

        self.model = model

    def compute_online_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        # Input of shape [B, T, F] output of shape (B,)
        (x,) = inputs

        with torch.no_grad():
            f_final, R_last, *_ = self.model(x)
            f_norm = F.normalize(f_final, dim=-1)
            c_norm = F.normalize(self.model.centers[-1], dim=-1)

            # Authors implementation:
            score = 0.5 * (1 - torch.einsum("bscf,cf -> bsc", f_norm, c_norm))
            # Correct following the paper in my opinion is, but this is just different scaling:
            # score = - torch.einsum('bscf,cf -> bsc', f_norm, c_norm)

            # Mean over clusters in shape [B, T, C] -> out [B, T] 
            anomaly_scores = (R_last * score).mean(dim=-1)

        # Return last observations
        return anomaly_scores[:,-1]

    def compute_offline_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        pass

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Input of shape (B, T) output of shape (B)
        (target,) = targets
        # Just return the last label of the window
        return target[:, -1]
