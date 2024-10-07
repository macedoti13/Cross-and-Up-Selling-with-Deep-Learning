from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
import torch

graph_path = "../data/transformed/graph.pth"
data = torch.load(graph_path, weights_only=False)

transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=True,
    edge_types=('customer', 'bought', 'product'),
    rev_edge_types=('product', 'rev_bought', 'customer')
)

# Train, validation and test splits
train_data, val_data, test_data = transform(data)

def create_train_dataloader(batch_size: int):
    edge_label_index = (("customer", "bought", "product"), train_data[("customer", "bought", "product")].edge_label_index)
    edge_label = train_data[("customer", "bought", "product")].edge_label
    return LinkNeighborLoader(
        data=train_data,
        num_neighbors=[3, 2, 1, 1, 1],
        neg_sampling_ratio=0,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=True,
    )

    
def create_val_dataloader(batch_size: int):
    edge_label_index = (("customer", "bought", "product"), val_data[("customer", "bought", "product")].edge_label_index)
    edge_label = val_data[("customer", "bought", "product")].edge_label
    return LinkNeighborLoader(
        data=val_data,
        num_neighbors=[3, 2, 1, 1, 1],
        neg_sampling_ratio=0,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=True,
    )
    
def create_test_dataloader(batch_size: int):
    edge_label_index = (("customer", "bought", "product"), test_data[("customer", "bought", "product")].edge_label_index)
    edge_label = test_data[("customer", "bought", "product")].edge_label
    return LinkNeighborLoader(
        data=test_data,
        num_neighbors=[3, 2, 1, 1, 1],
        neg_sampling_ratio=0,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=True,
    )
