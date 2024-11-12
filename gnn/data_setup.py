from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch

# Define file paths
GRAPH_PATH = "../data/transformed/graph.pth"
SAVE_DIR = "../data/transformed"
TRAIN_DATA_PATH = f"{SAVE_DIR}/train_data.pth"
VAL_DATA_PATH = f"{SAVE_DIR}/val_data.pth"
TEST_DATA_PATH = f"{SAVE_DIR}/test_data.pth"


def normalize_graph_features(train_data, val_data, test_data, save_path):
    """
    Normalizes customer and product node features in the train, validation, and test splits.
    Saves the customer and product normalizers to disk.

    Args:
    train_data, val_data, test_data: Heterogeneous graph data splits from PyTorch Geometric.
    save_path: Path where the normalizers will be saved.

    Returns:
    train_data, val_data, test_data: Normalized graph data splits.
    """
    # Step 1: Normalize customer features
    customer_scaler = MinMaxScaler()

    # Fit the scaler on the training customer features
    customer_train_features = train_data['customer'].x
    customer_train_features_scaled = torch.tensor(customer_scaler.fit_transform(customer_train_features), dtype=torch.float32)

    # Apply the same transformation to the validation and test sets
    customer_val_features_scaled = torch.tensor(customer_scaler.transform(val_data['customer'].x), dtype=torch.float32)
    customer_test_features_scaled = torch.tensor(customer_scaler.transform(test_data['customer'].x), dtype=torch.float32)

    # Update the normalized customer features in the graph
    train_data['customer'].x = customer_train_features_scaled
    val_data['customer'].x = customer_val_features_scaled
    test_data['customer'].x = customer_test_features_scaled

    # Save the customer normalizer
    customer_scaler_path = save_path + "/customer_scaler.pkl"
    joblib.dump(customer_scaler, customer_scaler_path)

    # Step 2: Normalize product features (only price and units purchased)
    product_scaler = MinMaxScaler()

    # Product features: Only normalize the first two features (price and units sold)
    product_train_features = train_data['product'].x[:, :2]  # Extract price and units purchased (first two features)
    product_train_features_scaled = torch.tensor(product_scaler.fit_transform(product_train_features), dtype=torch.float32)

    # Normalize validation and test product features
    product_val_features_scaled = torch.tensor(product_scaler.transform(val_data['product'].x[:, :2]), dtype=torch.float32)
    product_test_features_scaled = torch.tensor(product_scaler.transform(test_data['product'].x[:, :2]), dtype=torch.float32)

    # Combine the normalized price/units with the unchanged embeddings (remaining 768 dimensions)
    train_data['product'].x = torch.cat([product_train_features_scaled, train_data['product'].x[:, 2:]], dim=1)
    val_data['product'].x = torch.cat([product_val_features_scaled, val_data['product'].x[:, 2:]], dim=1)
    test_data['product'].x = torch.cat([product_test_features_scaled, test_data['product'].x[:, 2:]], dim=1)

    # Save the product normalizer
    product_scaler_path = save_path + "/product_scaler.pkl"
    joblib.dump(product_scaler, product_scaler_path)

    # Step 3: Normalize edge features
    edge_scaler = MinMaxScaler()

    # Fit the scaler on the training edge features
    edge_train_features = train_data["customer", "bought", "product"].edge_attr
    edge_train_features_scaled = torch.tensor(edge_scaler.fit_transform(edge_train_features), dtype=torch.float32)

    # Apply the same transformation to the validation and test sets
    edge_val_features_scaled = torch.tensor(edge_scaler.transform(val_data["customer", "bought", "product"].edge_attr), dtype=torch.float32)
    edge_test_features_scaled = torch.tensor(edge_scaler.transform(test_data["customer", "bought", "product"].edge_attr), dtype=torch.float32)

    # Update the normalized edge features in the graph
    train_data["customer", "bought", "product"].edge_attr = edge_train_features_scaled
    val_data["customer", "bought", "product"].edge_attr = edge_val_features_scaled
    test_data["customer", "bought", "product"].edge_attr = edge_test_features_scaled

    # Save the edge features normalizer
    edge_scaler_path = save_path + "/edge_scaler.pkl"
    joblib.dump(edge_scaler, edge_scaler_path)

    # Step 4: Normalize rev edge features
    rev_edge_scaler = MinMaxScaler()

    # Fit the scaler on the training rev edge features
    rev_edge_train_features = train_data["product", "rev_bought", "customer"].edge_attr
    rev_edge_train_features_scaled = torch.tensor(rev_edge_scaler.fit_transform(rev_edge_train_features), dtype=torch.float32)

    # Apply the same transformation to the validation and test sets
    rev_edge_val_features_scaled = torch.tensor(rev_edge_scaler.transform(val_data["product", "rev_bought", "customer"].edge_attr), dtype=torch.float32)
    rev_edge_test_features_scaled = torch.tensor(rev_edge_scaler.transform(test_data["product", "rev_bought", "customer"].edge_attr), dtype=torch.float32)

    # Update the normalized edge features in the graph
    train_data["product", "rev_bought", "customer"].edge_attr = rev_edge_train_features_scaled
    val_data["product", "rev_bought", "customer"].edge_attr = rev_edge_val_features_scaled
    test_data["product", "rev_bought", "customer"].edge_attr = rev_edge_test_features_scaled

    # Save the rev edge features normalizer
    rev_edge_scaler_path = save_path + "/rev_edge_scaler.pkl"
    joblib.dump(rev_edge_scaler, rev_edge_scaler_path)

    return train_data, val_data, test_data


# Load the preprocessed graph data
data = torch.load(GRAPH_PATH, weights_only=False)

# Split data into training, validation, and test sets with RandomLinkSplit transformation
transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=3.0,
    add_negative_train_samples=True,
    edge_types=("customer", "bought", "product"),
    rev_edge_types=("product", "rev_bought", "customer"),
)

train_data, val_data, test_data = transform(data)

# Normalize the features of the train, validation, and test data
train_data, val_data, test_data = normalize_graph_features(train_data, val_data, test_data, save_path=SAVE_DIR)

# Save processed train, validation, and test data
torch.save(train_data, TRAIN_DATA_PATH)
torch.save(val_data, VAL_DATA_PATH)
torch.save(test_data, TEST_DATA_PATH)


def create_train_dataloader(batch_size: int):
    """
    Creates a LinkNeighborLoader dataloader for the training dataset. This dataloader 
    creates batches with 'batch_size' number of links, the two nodes connected by the link 
    and 5 neighbors in 3 hops for each node.
    """
    edge_label_index = (("customer", "bought", "product"), train_data[("customer", "bought", "product")].edge_label_index)
    edge_label = train_data[("customer", "bought", "product")].edge_label
    return LinkNeighborLoader(
        data=train_data,
        num_neighbors=[5, 5, 5],
        neg_sampling_ratio=0,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=True,
    )


def create_val_dataloader(batch_size: int):
    """
    Creates a LinkNeighborLoader dataloader for the validation dataset. This dataloader
    creates batches with 'batch_size' number of links, the two nodes connected by the link
    and 5 neighbors in 3 hops for each node.
    """
    edge_label_index = (("customer", "bought", "product"), val_data[("customer", "bought", "product")].edge_label_index)
    edge_label = val_data[("customer", "bought", "product")].edge_label
    return LinkNeighborLoader(
        data=val_data,
        num_neighbors=[5, 5, 5],
        neg_sampling_ratio=0,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=True,
    )

def create_test_dataloader(batch_size: int):
    """
    Creates a LinkNeighborLoader dataloader for the test dataset. This dataloader
    creates batches with 'batch_size' number of links, the two nodes connected by the link
    and 5 neighbors in 3 hops for each node.
    """
    edge_label_index = (("customer", "bought", "product"), test_data[("customer", "bought", "product")].edge_label_index)
    edge_label = test_data[("customer", "bought", "product")].edge_label
    return LinkNeighborLoader(
        data=test_data,
        num_neighbors=[5, 5, 5],
        neg_sampling_ratio=0,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=True,
    )
