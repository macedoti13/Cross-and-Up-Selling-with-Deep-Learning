import os
from loguru import logger
import pandas as pd

def create_customers(df):
    logger.info("Creating customers dataset.")
    # Group by customer-related columns and aggregate sales data
    grouped = df.groupby(
        ["COD_CLIENTE", "CLIENTE_FISICO_JURIDICO", "SEXO_CLIENTE", "DTNASCIMENTO_CLIENTE"]
    ).agg(
        total_gross_sum=("TOTAL_BRUTO", "sum"),
        total_gross_mean=("TOTAL_BRUTO", "mean"),
        total_gross_max=("TOTAL_BRUTO", "max"),
        total_gross_min=("TOTAL_BRUTO", "min"),
        total_discount_sum=("TOTAL_DESCONTO", "sum"),
        total_discount_mean=("TOTAL_DESCONTO", "mean"),
        total_discount_max=("TOTAL_DESCONTO", "max"),
        total_discount_min=("TOTAL_DESCONTO", "min"),
        total_spent_sum=("TOTAL_LIQUIDO", "sum"),
        total_spent_mean=("TOTAL_LIQUIDO", "mean"),
        total_spent_max=("TOTAL_LIQUIDO", "max"),
        total_spent_min=("TOTAL_LIQUIDO", "min"),
        purchases=("COD_CLIENTE", "size")  # Number of purchases
    ).reset_index()

    # Rename columns for clarity
    grouped.rename(columns={
        "COD_CLIENTE": "customer_id",
        "CLIENTE_FISICO_JURIDICO": "customer_type",
        "SEXO_CLIENTE": "gender",
        "DTNASCIMENTO_CLIENTE": "birth_date",
    }, inplace=True)

    # Drop unnecessary columns
    grouped.drop(columns=["customer_type"], inplace=True)

    # Calculate age from birth_date
    today = pd.Timestamp("today")
    grouped["birth_date"] = pd.to_datetime(grouped["birth_date"], errors='coerce')  # Handle invalid dates
    grouped["age"] = (today - grouped["birth_date"]).dt.days // 365

    # Filter out invalid ages (negative values)
    grouped = grouped[grouped["age"] >= 0]

    return grouped

def create_products(df):
    logger.info("Creating products dataset.")
    # Group by product-related columns and calculate mean regular price
    products = df.groupby(
        ["COD_SKU", "SKU", "CATEGORIA_SKU", "SUBCATEGORIA_SKU"]
    )["PRECO_REGULAR"].mean().round(2).reset_index()

    # Rename columns for clarity
    products.rename(columns={
        "COD_SKU": "product_id",
        "SKU": "product_name",
        "CATEGORIA_SKU": "category",
        "SUBCATEGORIA_SKU": "subcategory",
        "PRECO_REGULAR": "price"
    }, inplace=True)

    # Drop duplicate products based on name, category, and subcategory
    products.drop_duplicates(subset=["product_name", "category", "subcategory"], inplace=True)

    # Exclude certain categories
    exclude_categories = ["SERVIÇOS", "DIVERSOS", "MANIPULADOS", "IMOBILIZADO"]
    products = products[~products["category"].isin(exclude_categories)]

    # Remove subcategories with fewer than 11 occurrences
    subcategory_counts = products["subcategory"].value_counts()
    under_11_subcategories = subcategory_counts[subcategory_counts < 11].index
    products = products[~products["subcategory"].isin(under_11_subcategories)]

    # Filter products by price range
    products = products[(products["price"] >= 1) & (products["price"] <= 9999)]

    return products

def create_sales(df, customers, products):
    logger.info("Creating sales dataset.")
    
    # Filter sales by customers and products that exist in the respective datasets
    sales = df[df.COD_CLIENTE.isin(customers.customer_id) & df.COD_SKU.isin(products.product_id)]
    
    # Filter customers and products based on sales
    customers = customers[customers.customer_id.isin(sales.COD_CLIENTE)]
    products = products[products.product_id.isin(sales.COD_SKU)]

    # Select relevant columns for sales
    sales = sales[[
        "COD_CUPOM", "COD_CLIENTE", "COD_SKU", "DATA_CUPOM", "COD_LOJA", 
        "UNIDADES", "IDENTIFICADOR_PROMOCIONAL", "PRECO_REGULAR", 
        "TOTAL_DESCONTO", "TOTAL_BRUTO", "TOTAL_LIQUIDO"
    ]]

    # Create sale_id by combining relevant columns
    sales["sale_id"] = sales[['COD_CLIENTE', 'COD_LOJA', 'DATA_CUPOM']].astype(str).agg('_'.join, axis=1)
    sales["sale_id"] = sales.groupby("sale_id").ngroup() + 1

    # Rename columns to more user-friendly names
    sales.rename(columns={
        "COD_CUPOM": "cod_cupon",
        "COD_CLIENTE": "customer_id",
        "COD_SKU": "product_id",
        "COD_LOJA": "store_id",
        "DATA_CUPOM": "date",
        "UNIDADES": "units",
        "IDENTIFICADOR_PROMOCIONAL": "promotion_id",
        "PRECO_REGULAR": "regular_price",
        "TOTAL_DESCONTO": "discount",
        "TOTAL_BRUTO": "gross_total",
        "TOTAL_LIQUIDO": "net_total",
    }, inplace=True)

    # Extract temporal features from the date column
    sales["week_of_year"] = sales["date"].dt.isocalendar().week
    sales["day_of_week"] = sales["date"].dt.dayofweek
    sales["hour"] = sales["date"].dt.hour
    sales["minute"] = sales["date"].dt.minute

    # Add a binary flag indicating if the sale was part of a promotion
    sales["was_in_promotion"] = sales["promotion_id"].notnull().astype(int)
    
    # Drop columns that are no longer needed
    sales.drop(columns=["promotion_id"], inplace=True)
    
    # Fill missing values in the discount column
    sales.fillna({'discount': 0.0}, inplace=True)
    
    # Aggregate duplicate rows by summing up the numeric values
    group_cols = [
        "sale_id", "customer_id", "product_id", "store_id", 
        "week_of_year", "day_of_week", "hour", "minute", "regular_price", "was_in_promotion"
    ]
    sales = sales.groupby(group_cols).agg({
        'units': 'sum',
        'discount': 'sum',
        'gross_total': 'sum',
        'net_total': 'sum'
    }).reset_index()
    
    # Convert units to integer type
    sales.units = sales.units.astype(int)

    # Reorder columns for final output
    sales = sales[[
        "sale_id", "customer_id", "product_id", "store_id", 
        "week_of_year", "day_of_week", "hour", "minute", "units", 
        "was_in_promotion", "regular_price", "discount", "gross_total", "net_total"
    ]].sort_values(by=["customer_id", "product_id", "sale_id"]).reset_index(drop=True)
    
    # reorder columns and reset indexes of customers and products
    customers = customers[customers.customer_id.isin(sales.customer_id)].sort_values(by="customer_id").reset_index(drop=True)
    products = products[products.product_id.isin(sales.product_id)].sort_values(by="product_id").reset_index(drop=True)

    return sales, customers, products

def create_cleaned_datasets(df):
    logger.info("Starting to create cleaned datasets.")
    
    # Create customers dataset
    customers = create_customers(df)
    
    # Create products dataset
    products = create_products(df)
    
    # Create sales dataset
    sales, customers, products = create_sales(df, customers, products)
    
    logger.info("Checking for data integrity.")
    assert sales.customer_id.isin(customers.customer_id).all()
    assert sales.product_id.isin(products.product_id).all()
    assert customers.customer_id.is_unique
    assert products.product_id.is_unique
        
    # Save cleaned datasets to parquet
    logger.info("Saving cleaned datasets.")
    cleaned_data_dir = os.path.join(os.path.dirname(os.getcwd()),'SJ_PCD_24-2', 'data', 'cleaned')
    os.makedirs(cleaned_data_dir, exist_ok=True)
    customers.to_parquet(os.path.join(cleaned_data_dir, "customers.parquet"))
    products.to_parquet(os.path.join(cleaned_data_dir, "products.parquet"))
    sales.to_parquet(os.path.join(cleaned_data_dir, "sales.parquet"))
    logger.info("Cleaned datasets saved successfully.")

def main():
    logger.info("Starting the main process to create cleaned datasets.")
    
    # Read original dataset
    raw_data_path = os.path.join(os.path.dirname(os.getcwd()), 'SJ_PCD_24-2', 'data', 'raw')
    df = pd.read_parquet(os.path.join(raw_data_path, 'puc_vendas.parquet'))
    logger.info("Successfully read puc_vendas dataset.")
    
    # small fix in df 
    df = df.drop_duplicates()

    # Create and save cleaned datasets
    create_cleaned_datasets(df)
    logger.info("Cleaned datasets created and saved successfully.")

if __name__ == "__main__":
    main()
