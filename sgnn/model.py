import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GINEConv, global_mean_pool
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
        self.per_layer_edge_encoders = nn.ModuleList()

        # Initialize from configuration
        model_type = cfg.model.type
        hidden_channels = cfg.model.hidden_channels
        num_layers = cfg.model.num_layers
        heads = cfg.model.get("heads", 1)
        dropout = cfg.model.dropout
        activation = cfg.model.activation
        residual = cfg.model.get("residual", False)
        use_edge_encoders = cfg.model.get("use_edge_encoders", False)
        use_classifier_mlp = cfg.model.get("use_classifier_mlp", False)
        classifier_mlp_dims = cfg.model.get("classifier_mlp_dims", [])

        # Activation function setup
        self.activation = self._get_activation(activation)

        # Edge dimension handling
        current_edge_dim = edge_dim

        # Build GNN layers
        for i in range(num_layers):
            if use_edge_encoders:
                edge_encoder = nn.Linear(current_edge_dim, hidden_channels)
                self.per_layer_edge_encoders.append(edge_encoder)
                current_edge_dim = hidden_channels
            else:
                self.per_layer_edge_encoders.append(nn.Identity())

            if model_type == "GINE":
                if residual:
                    self.res = nn.Linear(in_channels, hidden_channels, bias=False)
                else:
                    self.register_parameter("res", None)

                # GINE convolution
                gin_nn = nn.Sequential(
                    nn.Linear(
                        in_channels if i == 0 else hidden_channels, hidden_channels
                    ),
                    nn.Dropout(dropout),
                    self.activation,
                    nn.Linear(hidden_channels, hidden_channels),
                )
                self.layers.append(GINEConv(gin_nn, edge_dim=hidden_channels))
            elif model_type == "GATv2":
                self.layers.append(
                    GATv2Conv(
                        in_channels if i == 0 else hidden_channels * heads,
                        hidden_channels,
                        heads=heads,
                        edge_dim=edge_dim,
                        dropout=dropout,
                        residual=residual,
                    )
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        # Classifier configuration
        classifier_input_dim = hidden_channels * heads

        if use_classifier_mlp and classifier_mlp_dims:
            classifier_layers = []
            current_dim = classifier_input_dim
            for dim in classifier_mlp_dims:
                classifier_layers.append(nn.Linear(current_dim, dim))
                classifier_layers.append(self.activation)
                current_dim = dim
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
        if hasattr(self, "res"):
            res = self.res(x)
        else:
            res = None

        for i, layer in enumerate(self.layers):
            # Process layer
            edge_attr = self.per_layer_edge_encoders[i](edge_attr)
            x = layer(x, edge_index, edge_attr)
            if res is not None:
                x = x + res

            # Apply activation
            x = self.activation(x)

        # Global pooling and classification
        x = global_mean_pool(x, batch)
        return self.classifier(x)
