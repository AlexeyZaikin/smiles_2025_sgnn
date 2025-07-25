import torch
from torch_geometric.data import Data
from torch_geometric.nn import (
    GATv2Conv,
    GINEConv,
    global_mean_pool,
    TransformerConv,
    GCNConv,
)
import torch.nn as nn
import warnings
from omegaconf import DictConfig

warnings.filterwarnings("ignore")


class GNNModel(nn.Module):
    """Enhanced GNN model with automatic configuration handling"""

    def __init__(
        self,
        cfg: DictConfig,
        in_channels: int = 1,
        out_channels: int = 2,
        edge_dim: int = 1,
    ):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList()
        self.edge_encoders = nn.ModuleList()

        # Initialize from configuration
        self.model_type = cfg.model.type
        activation = cfg.model.activation
        self.hidden_channels = cfg.model.hidden_channels
        self.num_layers = cfg.model.num_layers
        self.dropout = cfg.model.dropout
        self.residual = cfg.model.get("residual", False)
        self.heads = cfg.model.get("heads", 1)
        self.concat = cfg.model.get("concat", True)
        self.use_edge_encoders = cfg.model.get("use_edge_encoders", False)
        self.edge_encoder_channels = cfg.model.get("edge_encoder_channels", 16)
        self.edge_encoder_layers = cfg.model.get("edge_encoder_layers", 1)
        self.use_classifier_mlp = cfg.model.get("use_classifier_mlp", False)
        self.classifier_mlp_channels = cfg.model.get("classifier_mlp_channels", 16)
        self.classifier_mlp_layers = cfg.model.get("classifier_mlp_layers", 1)

        # Activation function setup
        self.activation = self._get_activation(activation)

        current_edge_dim = edge_dim
        # Build GNN layers
        for i in range(self.num_layers):
            # Edge dimension handling
            if self.use_edge_encoders:
                per_layer_edge_encoders = []
                for i in range(self.edge_encoder_layers):
                    edge_encoder = nn.Sequential(
                        nn.Linear(current_edge_dim, self.edge_encoder_channels),
                        nn.Dropout(self.dropout),
                        self.activation,
                    )
                    current_edge_dim = self.edge_encoder_channels
                    per_layer_edge_encoders.append(edge_encoder)
                self.edge_encoders.append(nn.Sequential(*per_layer_edge_encoders))

            if self.model_type == "GINE":
                if self.residual:
                    self.res = nn.Linear(in_channels, self.hidden_channels, bias=False)
                else:
                    self.res = None

                # GINE convolution
                gin_nn = nn.Sequential(
                    nn.Linear(
                        in_channels if i == 0 else self.hidden_channels,
                        self.hidden_channels,
                    ),
                    nn.Dropout(self.dropout),
                    self.activation,
                )

                self.layers.append(GINEConv(gin_nn, edge_dim=current_edge_dim))
            elif self.model_type == "GATv2":
                self.layers.append(
                    GATv2Conv(
                        in_channels
                        if i == 0
                        else self.hidden_channels * self.heads
                        if self.concat
                        else self.hidden_channels,
                        self.hidden_channels,
                        concat=self.concat,
                        heads=self.heads,
                        edge_dim=current_edge_dim,
                        dropout=self.dropout,
                        residual=self.residual,
                    )
                )
            elif self.model_type == "Transformer":
                self.layers.append(
                    TransformerConv(
                        in_channels
                        if i == 0
                        else self.hidden_channels * self.heads
                        if self.concat
                        else self.hidden_channels,
                        self.hidden_channels,
                        self.heads,
                        self.concat,
                        dropout=self.dropout,
                        edge_dim=current_edge_dim,
                    )
                )
            elif self.model_type == "GCN":
                if self.residual:
                    self.res = nn.Linear(in_channels, self.hidden_channels, bias=False)
                else:
                    self.res = None
                self.layers.append(
                    GCNConv(
                        in_channels if i == 0 else self.hidden_channels,
                        self.hidden_channels,
                        improved=True,
                    )
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        # Classifier configuration
        classifier_input_dim = (
            self.hidden_channels * self.heads if self.concat else self.hidden_channels
        )

        if self.use_classifier_mlp:
            classifier_layers = []
            current_dim = classifier_input_dim
            for _ in range(self.classifier_mlp_layers):
                classifier_layer = nn.Sequential(
                    nn.Linear(current_dim, self.classifier_mlp_channels),
                    nn.Dropout(self.dropout),
                    self.activation,
                )
                classifier_layers.append(classifier_layer)
                current_dim = self.classifier_mlp_channels
            classifier_layers.append(nn.Linear(current_dim, out_channels))
            self.classifier = nn.Sequential(*classifier_layers)
        else:
            self.classifier = nn.Linear(classifier_input_dim, out_channels)

    def _get_activation(self, name: str) -> nn.Module:
        """Automatically select activation function"""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
            "prelu": nn.PReLU(),
            "selu": nn.SELU(),
            "tanh": nn.Tanh(),
        }
        name = name.lower()
        return activations.get(name, nn.ReLU())

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass with residual connections"""
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        if hasattr(self, "res") and self.res is not None:
            x_res = self.res(x)
        else:
            x_res = None

        for i, layer in enumerate(self.layers):
            if self.use_edge_encoders:
                edge_attr = self.edge_encoders[i](edge_attr)
            x = layer(x, edge_index, edge_attr)
            if x_res is not None:
                x = x + x_res
            # Apply activation
            x = self.activation(x)
        # Global pooling and classification
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x
