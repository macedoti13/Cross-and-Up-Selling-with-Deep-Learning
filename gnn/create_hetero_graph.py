import os
import torch
import pandas as pd
from torch.nn import Embedding
from torch_geometric.transforms import ToUndirected
from torch_geometric.data import HeteroData

# Directory paths
DATA_DIR = "../data"
CUSTOMERS_PATH = os.path.join(DATA_DIR, "cleaned", "customers.parquet")
SALES_PATH = os.path.join(DATA_DIR, "cleaned", "sales.parquet")
PRODUCTS_PATH = os.path.join(DATA_DIR, "cleaned", "products.parquet")
PRODUCTS_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "transformed", "products_embeddings.parquet")
GRAPH_SAVE_PATH = os.path.join(DATA_DIR, "transformed", "graph.pth")


def read_datasets():
    """Load data from parquet files into Pandas DataFrames."""
    customers = pd.read_parquet(CUSTOMERS_PATH)
    sales = pd.read_parquet(SALES_PATH)
    products = pd.read_parquet(PRODUCTS_PATH)
    products_embeddings = pd.read_parquet(PRODUCTS_EMBEDDINGS_PATH)
    return customers, sales, products, products_embeddings


def update_customers(customers: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    """
    Update and preprocess customer data for graph construction.
    
    - Creates a mapping for customer IDs to graph-compatible indices.
    - Generates a new DataFrame with required customer features.

    Args:
        customers (pd.DataFrame): Original customer data.

    Returns:
        tuple: Updated customer DataFrame and a mapping dictionary of original IDs to graph indices.
    """
    customer_id_mapping = {id_: idx for idx, id_ in enumerate(customers['customer_id'].values)}
    customers['customer_id_for_graph'] = customers['customer_id'].map(customer_id_mapping)
    
    customers = customers[[
        "customer_id_for_graph", "customer_id", "customer_age", "customer_gender",
        "purchases", "total_gross_sum", "total_gross_mean", "total_gross_max", "total_gross_min",
        "total_discount_sum", "total_discount_mean", "total_discount_max", "total_discount_min",
        "total_net_sum", "total_net_mean", "total_net_max", "total_net_min"
    ]]
    return customers, customer_id_mapping


def update_products_and_embeddings(products: pd.DataFrame, products_embeddings: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    """
    Update and preprocess product data and embeddings for graph construction.
    
    - Creates a mapping for product IDs to graph-compatible indices.
    - Merges product information with their embeddings.

    Args:
        products (pd.DataFrame): Original product data.
        products_embeddings (pd.DataFrame): Embedding data for each product.

    Returns:
        tuple: Updated product DataFrame with embeddings and a mapping dictionary.
    """
    products_id_mapping = {id_: idx for idx, id_ in enumerate(products['product_id'].values)}
    products["product_id_for_graph"] = products["product_id"].map(products_id_mapping)
    products = products[["product_id_for_graph", "product_id", "product_price", "units_sold"]]

    final_products = products.merge(products_embeddings, on="product_id")
    embedding_df = pd.DataFrame(final_products['embedding'].tolist(), index=final_products.index)
    final_products = pd.concat([final_products.drop('embedding', axis=1), embedding_df], axis=1)
    
    return final_products, products_id_mapping


def update_sales(sales: pd.DataFrame, customer_id_mapping: dict, products_id_mapping: dict) -> pd.DataFrame:
    """
    Update and preprocess sales data for graph construction.
    
    - Maps customer and product IDs to graph-compatible indices.
    - Encodes store IDs as embeddings to represent store attributes.
    
    Args:
        sales (pd.DataFrame): Original sales data.
        customer_id_mapping (dict): Mapping of customer IDs to graph indices.
        products_id_mapping (dict): Mapping of product IDs to graph indices.
    
    Returns:
        pd.DataFrame: Updated sales DataFrame with necessary features and embeddings.
    """
    sales["sale_id_for_graph"] = sales["sale_id"].rank(method='dense').astype(int) - 1
    sales["customer_id_for_graph"] = sales["customer_id"].map(customer_id_mapping)
    sales["product_id_for_graph"] = sales["product_id"].map(products_id_mapping)

    sales = sales[[
        "sale_id_for_graph", "customer_id_for_graph", "product_id_for_graph", "sale_id", "customer_id",
        "product_id", "store_id", "week_of_year", "day_of_week", "hour", "units", "gross_total",
        "was_in_promotion", "total_discount", "net_total"
    ]]
    
    # Encode store ID as embeddings
    sales['store_id_code'] = sales['store_id'].astype('category').cat.codes
    num_stores = sales['store_id_code'].nunique()
    store_embedding_layer = Embedding(num_embeddings=num_stores, embedding_dim=5)
    store_embeddings = store_embedding_layer(torch.tensor(sales['store_id_code'].values, dtype=torch.long))
    embedding_cols = [f'store_embedding_{i}' for i in range(5)]
    sales = pd.concat([sales, pd.DataFrame(store_embeddings.detach().numpy(), columns=embedding_cols)], axis=1)
    
    return sales


def create_graph(customers: pd.DataFrame, sales: pd.DataFrame, products: pd.DataFrame) -> HeteroData:
    """
    Construct a heterogeneous graph from customer, sales, and product data.
    
    - Adds customer and product node features.
    - Creates edges with attributes between customers and products.
    
    Args:
        customers (pd.DataFrame): Updated customer data.
        sales (pd.DataFrame): Updated sales data with embeddings.
        products (pd.DataFrame): Updated product data with embeddings.
    
    Returns:
        HeteroData: The heterogeneous graph data object for PyTorch Geometric.
    """
    data = HeteroData()
    data["customer"].x = torch.tensor(customers.drop(columns=["customer_id", "customer_id_for_graph"]).values, dtype=torch.float)
    data["product"].x = torch.tensor(products.drop(columns=["product_id_for_graph", "product_id"]).values, dtype=torch.float)
    
    edge_index = torch.tensor([
        sales["customer_id_for_graph"].values,
        sales["product_id_for_graph"].values
    ], dtype=torch.long)
    data[("customer", "bought", "product")].edge_index = edge_index
    
    edge_features = [
        "week_of_year", "day_of_week", "hour", "units", "gross_total", "was_in_promotion",
        "total_discount", "net_total", "store_embedding_0", "store_embedding_1",
        "store_embedding_2", "store_embedding_3", "store_embedding_4"
    ]
    edge_attr = torch.tensor(sales[edge_features].astype(float, errors='ignore').values, dtype=torch.float)
    data[("customer", "bought", "product")].edge_attr = edge_attr
    
    data = ToUndirected()(data)  # Make edges bidirectional
    return data


def main():
    """Main execution function for reading data, processing, and creating the graph."""
    # Load datasets
    customers, sales, products, products_embeddings = read_datasets()
    
    # Process datasets for graph compatibility
    updated_customers, customer_id_mapping = update_customers(customers)
    updated_products, products_id_mapping = update_products_and_embeddings(products, products_embeddings)
    updated_sales = update_sales(sales, customer_id_mapping, products_id_mapping)
    
    # Build the graph and save it
    graph = create_graph(updated_customers, updated_sales, updated_products)
    torch.save(graph, GRAPH_SAVE_PATH)


if __name__ == "__main__":
    main()
