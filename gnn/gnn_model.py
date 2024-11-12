from torch_geometric.nn import GATv2Conv, to_hetero
from torch.nn import Linear, LeakyReLU, Dropout
from torch_geometric.data import HeteroData
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
        self.drop1 = Dropout(0.2)
        self.lin2 = Linear(hidden_channels, hidden_channels//2)
        self.drop2 = Dropout(0.2)
        self.lin3 = Linear(hidden_channels//2, out_channels)

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
        x = self.drop1(x)
        x = self.lin2(x)
        x = LeakyReLU()(x)
        x = self.drop2(x)
        x = self.lin3(x)
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
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, edge_dim=edge_dim, add_self_loops=False, dropout=0.2, heads=5)  # Aggregates information from the node's 1-hop neighbors
        self.conv2 = GATv2Conv(hidden_channels * 5, hidden_channels, edge_dim=edge_dim, add_self_loops=False, dropout=0.2, heads=5)  # Aggregates information from the node's 2-hop neighbors
        self.conv3 = GATv2Conv(hidden_channels * 5, out_channels, edge_dim=edge_dim, add_self_loops=False, dropout=0.2, heads=5)  # Aggregates information from the node's 3-hop neighbors

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
        self.lin1 = torch.nn.Linear(2 * hidden_channels * 5, hidden_channels)  # Adjusted for multi-head attention
        self.drop1 = Dropout(0.2)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.drop2 = Dropout(0.2)
        self.lin3 = torch.nn.Linear(hidden_channels // 2, 1)  # Outputs the probability of an edge

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
        z = self.drop1(z)
        z = self.lin2(z).relu()
        z = self.drop2(z)
        z = self.lin3(z)

        # Apply sigmoid to get probabilities
        return torch.sigmoid(z.view(-1))  # Output edge probabilities for all pairs

class GNN(torch.nn.Module):
    """
    The GNN class defines a complete graph neural network architecture designed to operate on a
    heterogeneous graph with customer and product nodes. It consists of three main components:

    - A ProductEncoder, which reduces the dimensionality of product node features to make them
      compatible with the rest of the GNN architecture.
    - A GNNEncoder, which performs message passing to generate embeddings for each node in the
      graph, aggregating information from neighboring nodes.
    - A GNNDecoder, which takes pairs of node embeddings and predicts the probability of an
      edge existing between them.

    This architecture is particularly suited for link prediction tasks on heterogeneous graphs
    where edges indicate relationships between different types of nodes (e.g., customer buying a
    product).

    Args:
        product_encoder_in_channels (int): Initial dimensionality of the product feature vectors.
        product_encoder_hidden_channels (int): Dimensionality of hidden layers in the product encoder.
        product_encoder_out_channels (int): Output dimensionality of the product encoder.
        gnn_encoder_hidden_channels (int): Dimensionality of embeddings within the GNN encoder.
        gnn_encoder_out_channels (int): Dimensionality of final node embeddings output by the GNN encoder.
        graph_edge_dim (int): Dimensionality of the edge features in the graph.
        graph (HeteroData): The heterogeneous graph object with metadata about node and edge types.

    Example usage:
        gnn = GNN(
            product_encoder_in_channels=770,
            product_encoder_hidden_channels=512,
            product_encoder_out_channels=128,
            gnn_encoder_hidden_channels=128,
            gnn_encoder_out_channels=64,
            graph_edge_dim=13,
            graph=graph_data
        )
        edge_probabilities = gnn(products=product_features, x_dict=node_features, edge_index_dict=edges, edge_attr_dict=edge_features, edge_label_index=edge_labels)
    """

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
        """
        Initializes the GNN architecture by setting up the product encoder, GNN encoder, and
        GNN decoder.

        Args:
            product_encoder_in_channels (int): Dimensionality of initial product features.
            product_encoder_hidden_channels (int): Dimensionality of hidden layers in the product encoder.
            product_encoder_out_channels (int): Output dimensionality of the product encoder.
            gnn_encoder_hidden_channels (int): Dimensionality of embeddings within the GNN encoder.
            gnn_encoder_out_channels (int): Final output dimensionality of node embeddings.
            graph_edge_dim (int): Dimensionality of edge features in the graph.
            graph (HeteroData): Heterogeneous graph containing metadata for node and edge types.
        """
        super().__init__()

        # Initialize the product encoder to reduce dimensionality of product features
        self.product_encoder = ProductEncoder(
            in_channels=product_encoder_in_channels,
            hidden_channels=product_encoder_hidden_channels,
            out_channels=product_encoder_out_channels,
        )

        # Initialize the GNN encoder for message passing across heterogeneous graph nodes
        gnn_encoder = GNNEncoder(
            hidden_channels=gnn_encoder_hidden_channels,
            out_channels=gnn_encoder_out_channels,
            edge_dim=graph_edge_dim,
        )

        # Convert the GNN encoder into a heterogeneous model that can handle multiple node types
        self.gnn_encoder = to_hetero(gnn_encoder, graph.metadata(), aggr="sum")

        # Initialize the GNN decoder to predict edge probabilities between nodes
        self.gnn_decoder = GNNDecoder(gnn_encoder_out_channels)

    def forward(
        self,
        products: torch.Tensor,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        edge_attr_dict: dict[tuple, torch.Tensor],
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Defines the forward pass of the GNN. This involves encoding product features, performing
        message passing to generate node embeddings, and predicting edge probabilities.

        Args:
            products (torch.Tensor): Input tensor of product features, to be encoded.
            x_dict (dict[str, torch.Tensor]): Dictionary of initial node features for each node type.
                                              Example key-value pairs: {"customer": customer_features, "product": product_features}.
            edge_index_dict (dict[tuple, torch.Tensor]): Dictionary containing edge indices for each relation in the graph.
            edge_attr_dict (dict[tuple, torch.Tensor]): Dictionary containing edge features for each relation.
            edge_label_index (torch.Tensor): Edge indices for which to compute link predictions.

        Returns:
            torch.Tensor: Predicted probabilities of edges existing between specified pairs of nodes.
        """
        # Encode product features and update product node embeddings in x_dict
        x_dict["product"] = self.product_encoder(products)

        # Apply the heterogeneous GNN encoder to generate node embeddings
        node_embeddings = self.gnn_encoder(x_dict, edge_index_dict, edge_attr_dict)

        # Decode the node embeddings to predict edge probabilities
        return self.gnn_decoder(node_embeddings, edge_label_index)
