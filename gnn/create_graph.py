from torch_geometric.transforms import ToUndirected
from torch_geometric.data import HeteroData
from torch.nn import Embedding
import pandas as pd
import torch

customers_path = "../data/cleaned/customers.parquet"
sales_path = "../data/cleaned/sales.parquet"
products_path = "../data/cleaned/products.parquet"
products_embeddings_path = "../data/transformed/products_embeddings.parquet"

customers = pd.read_parquet(customers_path)
sales = pd.read_parquet(sales_path)
products = pd.read_parquet(products_path)
products_embeddings = pd.read_parquet(products_embeddings_path)

def update_customers(customers):
    customer_id_mapping = {id_: idx for idx, id_ in enumerate(customers['customer_id'].values)}
    customers['customer_id_for_graph'] = customers['customer_id'].apply(lambda x: customer_id_mapping[x])
    customers = customers[[
        "customer_id_for_graph",
        "customer_id",
        "customer_age",
        "customer_gender",
        "purchases",
        "total_gross_sum", 
        "total_gross_mean", 
        "total_gross_max", 
        "total_gross_min",
        "total_discount_sum", 
        "total_discount_mean", 
        "total_discount_max", 
        "total_discount_min",
        "total_net_sum", 
        "total_net_mean", 
        "total_net_max", 
        "total_net_min",
    ]]
    
    return customers, customer_id_mapping

def update_products(products, products_embeddings):
    products_id_mapping = {id_: idx for idx, id_ in enumerate(products['product_id'].values)}
    products["product_id_for_graph"] = products["product_id"].apply(lambda x: products_id_mapping[x])
    products = products[["product_id_for_graph", "product_id", "product_price", "units_sold"]]

    final_products = products.merge(products_embeddings, on="product_id")
    embedding_df = pd.DataFrame(final_products['embedding'].tolist(), index=final_products.index)
    final_products = pd.concat([final_products.drop('embedding', axis=1), embedding_df], axis=1)
    
    return final_products, products_id_mapping

def update_sales(sales, customer_id_mapping, products_id_mapping) -> pd.DataFrame:

    # Mapping sale IDs to indices
    sales_id_mapping = {id_: idx for idx, id_ in enumerate(sales['sale_id'].values)}
    sales["sale_id_for_graph"] = sales["sale_id"].apply(lambda x: sales_id_mapping[x])
    sales["customer_id_for_graph"] = sales["customer_id"].apply(lambda x: customer_id_mapping[x])
    sales["product_id_for_graph"] = sales["product_id"].apply(lambda x: products_id_mapping[x])

    # Ensure modifications are applied to the dataframe without triggering a SettingWithCopyWarning
    sales = sales.loc[:, [
        "sale_id_for_graph", "customer_id_for_graph", "product_id_for_graph", "sale_id", "customer_id",
        "product_id", "store_id", "week_of_year", "day_of_week", "hour", "units", "gross_total", "was_in_promotion",
        "total_discount", "net_total"]]

    # Instead of one hot encoding, create embeddings to encode the store id
    sales.loc[:, 'store_id_code'] = sales['store_id'].astype('category').cat.codes  # Use .loc to avoid the warning

    # Create the embeddings for store_id
    num_stores = sales['store_id_code'].nunique()
    store_embedding = Embedding(num_embeddings=num_stores, embedding_dim=5)
    store_ids_tensor = torch.tensor(sales['store_id_code'].values, dtype=torch.long)
    store_embeddings = store_embedding(store_ids_tensor)

    # Add embeddings to dataframe
    sales['store_embeddings'] = store_embeddings.detach().numpy().tolist()
    embedding_cols = [f'store_embedding_{i}' for i in range(5)]
    embedding_df = pd.DataFrame(sales['store_embeddings'].tolist(), columns=embedding_cols)

    # Concatenate the embeddings with the original dataframe
    sales = pd.concat([sales, embedding_df], axis=1)

    # Drop the temporary 'store_embeddings' column
    sales = sales.drop(columns=['store_embeddings'])
    
    return sales

def create_graph(customers, final_products, sales):
    data = HeteroData()
    data["customer"].x = torch.tensor(customers.drop(["customer_id", "customer_id_for_graph"], axis=1).values, dtype=torch.float)
    data["product"].x = torch.tensor(final_products.drop(["product_id_for_graph", "product_id"], axis=1).values, dtype=torch.float)

    edge_index = torch.stack([
        torch.tensor(sales["customer_id_for_graph"].values, dtype=torch.long),
        torch.tensor(sales["product_id_for_graph"].values, dtype=torch.long)
    ], dim=0)
    data[("customer", "bought", "product")].edge_index = edge_index

    edge_attr_features = [
        "week_of_year", "day_of_week", "hour", "units", "gross_total", "was_in_promotion",
        "total_discount", "net_total", "store_embedding_0", "store_embedding_1", 
        "store_embedding_2", "store_embedding_3", "store_embedding_4"
    ]
    edge_attr = torch.tensor(sales[edge_attr_features].astype(float, errors='ignore').values, dtype=torch.float)
    data[("customer", "bought", "product")].edge_attr = edge_attr

    data = ToUndirected()(data)
    torch.save(data, "../data/transformed/graph.pth")

def main():
    
    customers_path = "../data/cleaned/customers.parquet"
    sales_path = "../data/cleaned/sales.parquet"
    products_path = "../data/cleaned/products.parquet"
    products_embeddings_path = "../data/transformed/products_embeddings.parquet"

    customers = pd.read_parquet(customers_path)
    customers, customer_id_mapping = update_customers(customers)
    
    products = pd.read_parquet(products_path)
    products_embeddings = pd.read_parquet(products_embeddings_path)
    final_products, products_id_mapping = update_products(products, products_embeddings)
    
    sales = pd.read_parquet(sales_path)
    sales = update_sales(sales, customer_id_mapping, products_id_mapping)
    
    create_graph(customers, final_products, sales)

if __name__ == "__main__":
    main()
