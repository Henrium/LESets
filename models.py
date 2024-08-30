import torch
from torch_geometric.nn.conv import GraphConv, GCNConv, GATConv, SAGEConv, CGConv
from torch_geometric.nn.norm import LayerNorm, BatchNorm
from torch_geometric.nn.pool import global_mean_pool
import torch.nn.functional as F

class GraphEmbedding(torch.nn.Module):
    def __init__(self, n_node_features, hidden_dim, conv, n_conv_layers, after_readout, activation, norm=None):
        super(GraphEmbedding, self).__init__()

        self.conv_op = conv
        match conv:      
            case 'SAGEConv': ConvLayer = SAGEConv
            case 'GCNConv': ConvLayer = GCNConv
            case 'GATConv': ConvLayer = GATConv
            case 'GraphConv': ConvLayer = GraphConv
            case 'CGConv': ConvLayer = CGConv
            case _: raise NotImplementedError
        
        match norm:
            case 'ln': NormLayer = LayerNorm
            case 'bn': NormLayer = BatchNorm
            case _: NormLayer = torch.nn.Identity
        
        self.act = activation
        self.after_readout = after_readout

        self.convs = torch.nn.ModuleList()

        # For CGConv, layer dim needs to match input dim
        if self.conv_op == 'CGConv':
            hidden_dim = n_node_features
            batch_norm = (norm == 'bn')
            for _ in range(n_conv_layers):
                self.convs.append(ConvLayer(channels=hidden_dim, dim=1, batch_norm=batch_norm))
        
        # Other conv layers
        else:
            self.convs.append(ConvLayer(n_node_features, hidden_dim))
            for _ in range(n_conv_layers-1):
                self.convs.append(ConvLayer(hidden_dim, hidden_dim))

        self.norm = NormLayer(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim)
        self.tanh = torch.nn.Tanh()
        
    def forward(self, graph):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr # .unsqueeze(1)
        for conv in self.convs:
            if self.conv_op == 'CGConv':
                x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
            elif self.conv_op == 'GATConv':
                x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
                x = self.norm(x)
            elif self.conv_op in {'GraphConv', 'GCNConv'}:
                x = conv(x=x, edge_index=edge_index, edge_weight=edge_attr)
                x = self.norm(x)
            else:
                x = conv(x=x, edge_index=edge_index)
                x = self.norm(x)
            x = self.act(x)
        x = global_mean_pool(x, graph.batch)

        match self.after_readout:
            case 'tanh': x = self.tanh(self.fc(x))
            case 'norm': x = F.normalize(self.fc(x), dim=1)
            case _: raise NotImplementedError
        return x

class LESets(torch.nn.Module):
    def __init__(self, n_node_features, gnn_dim, lesets_dim, conv, n_conv_layers, after_readout='tanh', activation='relu', norm=None):
        super().__init__()

        match activation:
            case 'relu': self.act = torch.nn.ReLU()
            case 'silu': self.act = torch.nn.SiLU()
            case 'gelu': self.act = torch.nn.GELU()
            case 'lrelu': self.act = torch.nn.LeakyReLU()
            case _: raise NotImplementedError
        
        emb_dim = n_node_features if conv == 'CGConv' else gnn_dim
        self.phi = GraphEmbedding(n_node_features, emb_dim, conv, n_conv_layers, after_readout, self.act, norm=norm)

       
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, lesets_dim),
            self.act,
            torch.nn.Linear(lesets_dim, lesets_dim),
            self.act,
            torch.nn.Linear(lesets_dim, 1)
        )

    def forward(self, graph_list):
        # graph_list is a batch of graphs in one datapoint
        embeddings = self.phi(graph_list)  # n_graphs * emb_dim

        frac = graph_list.y.unsqueeze(0)  # n_graphs * 1

        x = torch.matmul(frac, embeddings).squeeze()
        # Representation of a mixture

        x = self.rho(x)
        return x

class LESetsAtt(torch.nn.Module):
    def __init__(self, n_node_features, gnn_dim, lesets_dim, conv, n_conv_layers, after_readout='tanh', activation='relu', norm=None):
        super().__init__()

        match activation:
            case 'relu': self.act = torch.nn.ReLU()
            case 'silu': self.act = torch.nn.SiLU()
            case 'gelu': self.act = torch.nn.GELU()
            case 'lrelu': self.act = torch.nn.LeakyReLU()
            case _: raise NotImplementedError
        
        emb_dim = n_node_features if conv == 'CGConv' else gnn_dim
        self.phi = GraphEmbedding(n_node_features, emb_dim, conv, n_conv_layers, after_readout, self.act, norm=norm)

        self.att_q_net = torch.nn.Linear(emb_dim, lesets_dim)
        self.att_k_net = torch.nn.Linear(emb_dim, lesets_dim)
        self.att_v_net = torch.nn.Linear(emb_dim, emb_dim)
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, lesets_dim),
            self.act,
            torch.nn.Linear(lesets_dim, lesets_dim),
            self.act,
            torch.nn.Linear(lesets_dim, 1)
        )

    def forward(self, graph_list):
        # graph_list is a batch of graphs in one datapoint
        embeddings = self.phi(graph_list)  # n_graphs * emb_dim

        frac = graph_list.y.unsqueeze(0)  # n_graphs * 1

        # Permutation-invariant aggregation using attention mechanism
        att_queries = self.att_q_net(embeddings)  # n_graphs * att_dim
        att_values = self.att_v_net(embeddings)  # n_graphs * emb_dim
        att_keys = self.att_k_net(embeddings)  # n_graphs * att_dim
        att_scores = torch.matmul(att_queries, att_keys.transpose(0, 1)) / (att_keys.size(1) ** 0.5) # n_graphs * n_graphs
        att_outputs = torch.matmul(torch.softmax(att_scores, dim=0), att_values) # n_graphs * emb_dim
        x = torch.matmul(frac, att_outputs).squeeze()
        # Representation of a mixture

        x = self.rho(x)
        return x
