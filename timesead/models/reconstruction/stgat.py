"""
Code taken from https://github.com/zhanjun717/STGAT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

from ...models import BaseModel


class InputLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """
    def __init__(self, n_features, kernel_size=7):
        super(InputLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back


class StgatBlock(nn.Module):
    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None):
        super(StgatBlock, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.alpha = alpha
        self.embed_dim = embed_dim if embed_dim is not None else n_features

        self.embed_dim *= 2

        self.feature_gat_layers = GATConv(window_size, window_size)
        self.temporal_gat_layers = GATConv(n_features, n_features)

        self.temporal_gcn_layers = GCNConv(n_features, n_features)

    def forward(self, data, fc_edge_index, tc_edge_index):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps
        # ft = data.clone().detach()
        # tp = data.clone().detach()

        # ft = ft.permute(0, 2, 1)
        # batch_num, node_num, all_feature = ft.shape
        # ft = ft.reshape(-1, all_feature).contiguous()
        # f_out = self.feature_gat_layers(ft, fc_edge_index)
        # f_out = F.relu(f_out)
        # f_out = f_out.view(batch_num, node_num, -1)
        # f_out = f_out.permute(0, 2, 1)
        #
        # batch_num, node_num, all_feature = tp.shape
        # tp = tp.reshape(-1, all_feature).contiguous()
        # t_out = self.temporal_gat_layers(tp, tc_edge_index)
        # t_out = F.relu(t_out)
        # t_out = t_out.view(batch_num, node_num, -1)
        #
        # return f_out + t_out   #self.relu(res + h + z)

        x = data.clone().detach()
        x = x.permute(0, 2, 1)
        batch_num, node_num, all_feature = x.shape

        x = x.reshape(-1, all_feature).contiguous()
        # f_out = self.feature_gat_layers(x, fc_edge_index, return_attention_weights = True)
        f_out = self.feature_gat_layers(x, fc_edge_index)
        f_out = F.relu(f_out)
        f_out = f_out.view(batch_num, node_num, -1)
        f_out = f_out.permute(0, 2, 1)
        z = f_out.reshape(-1, node_num).contiguous()

        t_out = self.temporal_gcn_layers(z, tc_edge_index)
        t_out = F.relu(t_out)
        t_out = t_out.view(batch_num, node_num, -1)

        return t_out.permute(0, 2, 1)   #self.relu(res + h + z)


class BiLSTMLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(BiLSTMLayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.bilstm = nn.LSTM(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout, bidirectional=True)

    def forward(self, x):
        out, h = self.bilstm(x)
        out = out.permute(1,0,2)[-1, :, :] #, h[-1, :, :]  # Extracting from last layer
        return out


class BiLSTMDecoder(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(BiLSTMDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.bilstm = nn.LSTM(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout, bidirectional=True)

    def forward(self, x):
        decoder_out, _ = self.bilstm(x)
        return decoder_out


class ReconstructionModel(nn.Module):
    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = BiLSTMDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(2 * hid_dim, out_dim)

    def forward(self, x):
        # x will be last hidden state of the GRU layer
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)
        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out


class Forecasting_Model(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()


# graph is 'fully-connect'
def get_fc_graph_struc(n_features):
    edge_indices = torch.tensor([[i, j] for j in range(n_features) for i in range(n_features) if i != j])
    return edge_indices.T.contiguous()


def get_tc_graph_struc(temporal_len):
    edge_indices = torch.tensor([[i, j] for j in range(temporal_len) for i in range(j)])
    return edge_indices.T.contiguous()


class STGAT(BaseModel):
    def __init__(self, input_dim: int, window_size: int, embed_dim: int = None, layer_numb: int = 2,
                 lstm_n_layers: int = 1, lstm_hid_dim: int = 150, recon_n_layers: int = 1, recon_hid_dim: int = 150,
                 dropout: float = 0.2, alpha: float = 0.2):
        super(STGAT, self).__init__()

        layers1 = []
        layers2 = []
        layers3 = []

        self.layer_numb = layer_numb
        self.h_temp = []

        self.input_1 = InputLayer(input_dim, 1)
        self.input_2 = InputLayer(input_dim, 5)
        self.input_3 = InputLayer(input_dim, 7)

        for i in range(layer_numb):
            layers1 += [StgatBlock(input_dim, window_size, dropout, alpha, embed_dim)]
        for i in range(layer_numb):
            layers2 += [StgatBlock(input_dim, window_size, dropout, alpha, embed_dim)]
        for i in range(layer_numb):
            layers3 += [StgatBlock(input_dim, window_size, dropout, alpha, embed_dim)]

        self.stgat_1 = nn.Sequential(*layers1)
        self.stgat_2 = nn.Sequential(*layers2)
        self.stgat_3 = nn.Sequential(*layers3)

        self.bilstm = BiLSTMLayer(input_dim * 3, lstm_hid_dim, lstm_n_layers, dropout)
        self.recon_model = ReconstructionModel(window_size, 2 * lstm_hid_dim, recon_hid_dim, input_dim, recon_n_layers,
                                               dropout)

        # Register as buffers so that tensors are moved to the correct device along with the rest of the model
        self.register_buffer('fc_edge_index', get_fc_graph_struc(input_dim), persistent=False)
        self.register_buffer('tc_edge_index', get_tc_graph_struc(window_size), persistent=False)

    def forward(self, inputs):
        # x (B, T, D)
        x, = inputs

        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        fc_edge_index_sets = get_batch_edge_index(self.fc_edge_index, x.shape[0], x.shape[2])
        tc_edge_index_sets = get_batch_edge_index(self.tc_edge_index, x.shape[0], x.shape[1])

        h_cat_1 = x
        h_cat_2 = self.input_2(x)
        h_cat_3 = self.input_3(x)

        for layer in range(self.layer_numb):
            h_cat_1 = h_cat_1 + self.stgat_1[layer](h_cat_1, fc_edge_index_sets, tc_edge_index_sets)
            h_cat_2 = h_cat_2 + self.stgat_2[layer](h_cat_2, fc_edge_index_sets, tc_edge_index_sets)
            h_cat_3 = h_cat_3 + self.stgat_3[layer](h_cat_3, fc_edge_index_sets, tc_edge_index_sets)

        h_cat = torch.cat([h_cat_1, h_cat_2, h_cat_3], dim=2)

        out_end = self.bilstm(h_cat)
        h_end = out_end.view(x.shape[0], -1)   # Hidden state for last timestamp

        recons = self.recon_model(h_end)

        return recons
