from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
import torch

graph_path = "../data/transformed/graph.pth"
data = torch.load(graph_path, weights_only=False)

transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=False,
    edge_types=('customer', 'bought', 'product'),
    rev_edge_types=('product', 'rev_bought', 'customer')
)

# Train, validation and test splits
train_data, val_data, test_data = transform(data)

def create_train_dataloader(batch_size: int):
    return LinkNeighborLoader(
        data=train_data, 
        num_neighbors=[15, 15, 15, 10, 5],
        batch_size=batch_size,  
        edge_label_index=('customer', 'bought', 'product'), 
        edge_label=train_data['customer', 'bought', 'product'].edge_label  
    )
    
def create_val_dataloader(batch_size: int):
    return LinkNeighborLoader(
        data=val_data,
        num_neighbors=[15, 15, 15, 10, 5],
        batch_size=batch_size,
        edge_label_index=('customer', 'bought', 'product'),
        edge_label=val_data['customer', 'bought', 'product'].edge_label
    )
    
def create_test_dataloader(batch_size: int):
    return LinkNeighborLoader(
        data=test_data,
        num_neighbors=[15, 15, 15, 10, 5],
        batch_size=batch_size,
        edge_label_index=('customer', 'bought', 'product'),
        edge_label=test_data['customer', 'bought', 'product'].edge_label
    )
