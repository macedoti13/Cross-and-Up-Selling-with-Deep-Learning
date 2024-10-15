from torch_geometric.nn import GATv2Conv, to_hetero
from torch_geometric.data import HeteroData
from torch.nn import Linear, LeakyReLU
import torch


class ProductEncoder(torch.nn.Module):
    """
    The ProductEncoder is designed to process the feature vectors of product nodes
    in a heterogeneous graph. It employs a simple multi-layer perceptron (MLP) with
    LeakyReLU activation and three linear layers to reduce the dimensionality of the
    product feature vectors from their original dimension (in_channels) to a smaller
    output dimension (out_channels).

    This transformation is necessary because product nodes begin with 769-dimensional
    feature vectors, whereas customer nodes have only 15 dimensions. Without this
    dimensionality reduction, it would be difficult to balance the feature size for
    both node types while ensuring the output dimension is not too large for customer
    nodes or too small for product nodes.

    Args:
        in_channels (int): Dimensionality of the original product feature vectors.
        hidden_channels (int): Dimensionality of the hidden layers in the encoder.
        out_channels (int): Dimensionality of the final output feature vectors.

    Example usage:
        encoder = ProductEncoder(in_channels=769, hidden_channels=512, out_channels=128)
        product_embeddings = encoder(product_features)
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        Initializes the ProductEncoder by defining three linear layers that gradually
        reduce the dimensionality of product feature vectors, along with LeakyReLU
        activations applied after the first two layers.

        Args:
            in_channels (int): Initial dimensionality of product feature vectors (e.g., 769).
            hidden_channels (int): Dimensionality of the intermediate layers.
            out_channels (int): Final dimensionality of the product embeddings after encoding.
        """
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x):
        """
        Forward pass through the ProductEncoder. The input feature vectors are passed through
        three linear layers with LeakyReLU activations after the first two, resulting in
        encoded product features of a smaller, more manageable dimensionality.

        Args:
            x (torch.Tensor): Input tensor of product feature vectors. Shape [num_products, in_channels].

        Returns:
            torch.Tensor: Encoded product feature vectors with dimensionality [num_products, out_channels].
        """
        x = self.lin1(x)
        x = LeakyReLU()(x)
        x = self.lin2(x)
        return x


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, edge_dim):
        """
        The GNNEncoder generates the embeddings for all nodes in the graph.
        The hidden_channels is the dimensionality of the embeddings inside
        the GNN. The out_channels is the dimensionality of the final node
        embeddings.
        """
        super().__init__()
        # The first layer transforms the input node features from their original dimension to hidden_channels dimension.
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, edge_dim=edge_dim, add_self_loops=False, dropout=0.2)  # Aggregates information from the node's 1-hop neighbors
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_dim, add_self_loops=False, dropout=0.2)  # Aggregates information from the node's 2-hop neighbors
        self.conv3 = GATv2Conv(hidden_channels, out_channels, edge_dim=edge_dim, add_self_loops=False, dropout=0.2)  # Aggregates information from the node's 3-hop neighbors

    def forward(self, x, edge_index, edge_attr):
        """
        Performs message passing through the graph to update node embeddings.

        Args:
            x (torch.Tensor): The feature matrix of all nodes in the graph.
                              Shape [num_nodes, num_node_features].
            edge_index (torch.Tensor): The edge list representing connections in the graph.
                                       Shape [2, num_edges] (COO format).
            edge_attr (torch.Tensor): The edge feature matrix for all edges in the graph.
                                      Shape [num_edges, edge_dim].

        Returns:
            torch.Tensor: The updated node embeddings for all nodes in the graph.
                          Shape [num_nodes, out_channels].
        """
        # Apply 5 layers of message passing, updating node embeddings based on neighbors at increasing distances.
        x = self.conv1(x, edge_index, edge_attr).relu()  # Update embeddings with 1-hop neighbors
        x = self.conv2(x, edge_index, edge_attr).relu()  # Update embeddings with 2-hop neighbors
        x = self.conv3(x, edge_index, edge_attr)  # Update embeddings with 3-hop neighbors
        return x


class GNNDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        """
        A decoder that takes two node embeddings and predicts the probability of an edge between them.

        Args:
            hidden_channels (int): The size of the hidden dimension and also the size of the input node embeddings.
        """
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)  # Receives concatenated embeddings, outputs hidden_channels
        self.lin2 = torch.nn.Linear(hidden_channels, 1)  # Outputs the probability of an edge

    def forward(self, embeddings_dict, edge_index):
        """
        Forward pass that computes edge probabilities for all specified node pairs.

        Args:
            z_dict (dict): Node embeddings for all node types (e.g., 'customer' and 'product').
            edge_index (torch.Tensor): Edge indices for which to compute probabilities.

        Returns:
            torch.Tensor: Predicted edge probabilities for the specified edges.
        """
        # Concatenate customer and product embeddings for each edge
        row, col = edge_index
        z = torch.cat([embeddings_dict['customer'][row], embeddings_dict['product'][col]], dim=-1)

        # Pass concatenated embeddings through the linear layers
        z = self.lin1(z).relu()
        z = self.lin2(z)

        # Apply sigmoid to get probabilities
        return torch.sigmoid(z.view(-1))  # Output edge probabilities for all pairs

class GNN(torch.nn.Module):
    def __init__(
        self,
        product_encoder_in_channels: int,
        product_encoder_hidden_channels: int,
        product_encoder_out_channels: int,
        gnn_encoder_hidden_channels: int,
        gnn_encoder_out_channels: int,
        graph_edge_dim: int,
        graph: HeteroData,
    ):
        super().__init__()

        # Initialize product encoder
        self.product_encoder = ProductEncoder(
            in_channels=product_encoder_in_channels,
            hidden_channels=product_encoder_hidden_channels,
            out_channels=product_encoder_out_channels,
        )

        # Initialize GNN encoder
        gnn_encoder = GNNEncoder(
            hidden_channels=gnn_encoder_hidden_channels,
            out_channels=gnn_encoder_out_channels,
            edge_dim=graph_edge_dim,
        )

        # Convert GNN encoder to a heterogeneous GNN
        self.gnn_encoder = to_hetero(gnn_encoder, graph.metadata(), aggr="sum")

        # Initialize GNN decoder
        self.gnn_decoder = GNNDecoder(gnn_encoder_out_channels)

    def forward(
        self,
        products: torch.Tensor,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        edge_attr_dict: dict[tuple, torch.Tensor],
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
    
        # Encode product features and update node dictionary
        x_dict["product"] = self.product_encoder(products)

        # Encode node features using the heterogeneous GNN encoder
        node_embeddings = self.gnn_encoder(x_dict, edge_index_dict, edge_attr_dict)

        # Decode edge probabilities
        return self.gnn_decoder(node_embeddings, edge_label_index)
