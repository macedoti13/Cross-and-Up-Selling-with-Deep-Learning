from loguru import logger
import pandas as pd
import os

def preprocess_dataframe(df):
    """
    Preprocesses the input dataframe by performing the following steps:
    
    1. Removes rows where COD_CLIENTE is 0.0, as those customers cannot be accurately identified.
    2. Removes rows where CLIENTE_FISICO_JURIDICO is "Pessoa Jurídica", as there are too few data points.
    3. Drops unnecessary location-related columns (UF_CIDADE, COD_CIDADE, NOME_CIDADE).
    4. Renames columns to have more readable and consistent names.
    
    Parameters:
        df (pd.DataFrame): The input dataframe containing raw customer and product data.
    
    Returns:
        pd.DataFrame: The cleaned and preprocessed dataframe.
    """

    # Drop rows where COD_CLIENTE is 0.0
    df = df[df["COD_CLIENTE"] != 0.0]

    # Drop rows where CLIENTE_FISICO_JURIDICO is "Pessoa Jurídica" and remove the column
    df = df[df["CLIENTE_FISICO_JURIDICO"] != "Pessoa Jurídica"]
    df.drop(columns=["CLIENTE_FISICO_JURIDICO"], inplace=True)

    # Drop the columns related to location, as they are redundant
    df.drop(columns=["UF_CIDADE", "COD_CIDADE", "NOME_CIDADE"], inplace=True)

    # Rename columns to have more consistent and readable names
    df.rename(
        columns={
            "COD_CUPOM": "cod_cupom",
            "COD_CLIENTE": "customer_id",
            "SEXO_CLIENTE": "customer_gender",
            "DTNASCIMENTO_CLIENTE": "customer_birthdate",
            "COD_SKU": "product_id",
            "SKU": "product_name",
            "CATEGORIA_SKU": "product_category",
            "SUBCATEGORIA_SKU": "product_subcategory",
            "COD_LOJA": "store_id",
            "DATA_CUPOM": "purchase_date",
            "UNIDADES": "units",
            "IDENTIFICADOR_PROMOCIONAL": "promotion_id",
            "PRECO_REGULAR": "product_price", 
            "TOTAL_DESCONTO": "total_discount",
            "TOTAL_BRUTO": "gross_total",
            "TOTAL_LIQUIDO": "net_total"
        },
        inplace=True
    )
    
    return df

def create_products(df):
    """
    Processes a dataframe to extract distinct product information and filter it based on several criteria.
    
    This function aggregates product data and outputs two dataframes: one for distinct products and one for
    the original dataframe without product-specific columns.

    The steps involved:
    1. Group by product_id, product_name, product_category, and product_subcategory, calculating the total units sold and average product price.
    2. Round the product price to 2 decimal places and convert units_sold to integers.
    3. Resolve product duplicates by retaining the product with the highest units_sold.
    4. Remove products from specific categories and subcategories with fewer than 11 occurrences.
    5. Filter products based on price (between 1R$ and 9999R$) and keep only those sold at least 20 times.
    6. Drop product-related columns from the original dataframe.

    Parameters:
        df (pd.DataFrame): The input dataframe containing product information.
    
    Returns:
        tuple:
            pd.DataFrame: The original dataframe with product-related columns removed.
            pd.DataFrame: A new dataframe containing distinct products, filtered and aggregated based on the criteria.
    """

    # Group by product-related columns and calculate the mean price and total units sold
    products_df = df.groupby(["product_id", "product_name", "product_category", "product_subcategory"]).agg(
        product_price=("product_price", "mean"),
        units_sold=("units", "sum")
    ).reset_index()

    # Round product_price to 2 decimal places and convert units_sold to integer
    products_df["product_price"] = products_df["product_price"].round(2)
    products_df["units_sold"] = products_df["units_sold"].astype(int)

    # Sort by units_sold to keep the product with the highest units sold in case of duplicates
    products_df = products_df.sort_values(by="units_sold", ascending=False)
    products_df = products_df.drop_duplicates(subset=["product_name", "product_category", "product_subcategory"], keep="first")

    # Exclude specific product categories
    exclude_categories = ["SERVIÇOS", "DIVERSOS", "MANIPULADOS", "IMOBILIZADO"]
    products_df = products_df[~products_df["product_category"].isin(exclude_categories)]

    # Remove subcategories with fewer than 11 occurrences
    subcategory_counts = products_df["product_subcategory"].value_counts()
    under_11_subcategories = subcategory_counts[subcategory_counts < 11].index
    products_df = products_df[~products_df["product_subcategory"].isin(under_11_subcategories)]

    # Filter products based on the price range (between 1R$ and 9999R$)
    products_df = products_df[(products_df["product_price"] >= 1) & (products_df["product_price"] <= 9999)]

    # Keep only products sold at least 20 times
    products_df = products_df[products_df["units_sold"] >= 20]

    # Sort by product_id and reset the index
    products_df = products_df.sort_values(by="product_id", ascending=True).reset_index(drop=True)
    
    # delete from the original dataframe the columns that should be exclusive to the products dataframe
    df.drop(columns=["product_name", "product_category", "product_subcategory", "product_price"], inplace=True)

    return df, products_df
def create_customers(df):
    """
    Processes customer-related data by aggregating sales and discount information, calculating the age, and converting 
    gender to a numerical format. It also removes customer-specific columns from the original dataframe.

    Steps:
    1. Group by customer-related columns (customer_id, customer_gender, customer_birthdate) and aggregate sales data.
    2. Calculate customer age based on the birthdate and filter out invalid ages.
    3. Convert the gender column from "M"/"F" to 1/0.
    4. Drop customer-related columns from the original dataframe.

    Parameters:
        df (pd.DataFrame): The input dataframe containing customer and sales information.

    Returns:
        tuple:
            pd.DataFrame: The modified dataframe with customer sales metrics and age.
            pd.DataFrame: The original dataframe with customer-related columns removed.
    """

    # Group by customer-related columns and aggregate sales data
    customers = df.groupby(["customer_id", "customer_gender", "customer_birthdate"]).agg(
        total_gross_sum=("gross_total", "sum"),
        total_gross_mean=("gross_total", "mean"),
        total_gross_max=("gross_total", "max"),
        total_gross_min=("gross_total", "min"),
        total_net_sum=("net_total", "sum"),
        total_net_mean=("net_total", "mean"),
        total_net_max=("net_total", "max"),
        total_net_min=("net_total", "min"),
        total_discount_sum=("total_discount", "sum"),
        total_discount_mean=("total_discount", "mean"),
        total_discount_max=("total_discount", "max"),
        total_discount_min=("total_discount", "min"),
        purchases=("customer_id", "size")
    ).reset_index()

    # Calculate client age based on the birthdate
    today = pd.Timestamp("today")
    customers["customer_birthdate"] = pd.to_datetime(customers["customer_birthdate"], errors="coerce")
    customers.loc[:, "customer_age"] = (today - customers["customer_birthdate"]).dt.days // 365

    # Filter out customers with age less than 0
    customers = customers.loc[customers["customer_age"] >= 0]

    # Convert age column to an integer
    customers.loc[:, "customer_age"] = customers["customer_age"].astype(int)

    # Drop the client_birthdate column
    customers.drop(columns=["customer_birthdate"], inplace=True)

    # Convert gender column from "M" or "F" to 1 or 0
    customers.loc[:, "customer_gender"] = customers["customer_gender"].apply(lambda x: 1 if x == "M" else 0)
    
    # keep customers with at least 5 purchases
    customers = customers[customers.purchases >= 5]

    # reorder customers columns
    customers = customers[[
        "customer_id", 
        "customer_age",
        "customer_gender",
        "purchases",
        "total_gross_sum", 
        "total_gross_mean",
        "total_gross_max", 
        "total_gross_min", 
        "total_net_sum", 
        "total_net_mean",
        "total_net_max", 
        "total_net_min", 
        "total_discount_sum",
        "total_discount_mean", 
        "total_discount_max", 
        "total_discount_min"
        
    ]]

    # Drop customer-related columns from the original dataframe
    df.drop(columns=["customer_gender", "customer_birthdate"], inplace=True)
    
    return df, customers

def create_sales(df, customers, products):
    """
    Processes the sales dataframe by filtering it to contain only customers and products that exist in the customers and 
    products dataframes, generates unique sale IDs, cleans the data, and aggregates sales information.
    
    The function also refilters the customers and products dataframes to only contain entries with sales, and returns the 
    modified dataframes.
    
    Parameters:
        df (pd.DataFrame): The sales dataframe containing sales transactions.
        customers (pd.DataFrame): The dataframe containing customer information.
        products (pd.DataFrame): The dataframe containing product information.
    
    Returns:
        tuple: The processed sales, customers, and products dataframes.
    """
    
    # Filter sales dataframe to only include relevant customers and products
    sales = df[df.customer_id.isin(customers.customer_id) & df.product_id.isin(products.product_id)]
    
    # Filter customers and products to only include those with sales
    products = products[products.product_id.isin(sales.product_id)]
    customers = customers[customers.customer_id.isin(sales.customer_id)]
    
    # Ensure purchase_date is in datetime format
    sales = sales.copy()  # Make a copy to avoid SettingWithCopyWarning
    sales["purchase_date"] = pd.to_datetime(sales["purchase_date"])
    
    # Create a unique sale_id by combining relevant columns
    sales["sale_id_str"] = sales['customer_id'].astype(str) + '_' + sales['store_id'].astype(str) + '_' + sales['purchase_date'].dt.strftime('%Y-%m-%d')
    sales["sale_id"] = sales.groupby("sale_id_str").ngroup()
    sales.drop(columns=["sale_id_str"], inplace=True)
    
    # Drop sales related to donations from the floods of Porto Alegre in 2024
    donations = sales[(sales.purchase_date >= "2024-04-28") & (sales.purchase_date <= "2024-05-30")]
    donations = donations[donations.groupby("sale_id")["units"].transform('sum') >= 10]
    sales = sales[~sales.sale_id.isin(donations.sale_id.values)]
    
    # Make units column an integer
    sales["units"] = sales["units"].astype(int)
    
    # Extract temporal features from the purchase_date column
    sales["week_of_year"] = sales["purchase_date"].dt.isocalendar().week
    sales["day_of_week"] = sales["purchase_date"].dt.dayofweek
    sales["hour"] = sales["purchase_date"].dt.hour
    
    # Drop unnecessary columns
    sales.drop(columns=["purchase_date", "cod_cupom"], inplace=True)
    
    # Add a binary flag indicating if the sale was part of a promotion
    sales["was_in_promotion"] = sales["promotion_id"].notnull().astype(int)
    sales.drop(columns=["promotion_id"], inplace=True)
    
    # Fill missing values in the discount column
    sales.fillna({'total_discount': 0.0}, inplace=True)
    
    # Group by relevant columns and sum the necessary numeric columns
    columns_to_group_by = ['customer_id', 'product_id', 'store_id', 'was_in_promotion', 'sale_id', 'week_of_year', 'day_of_week', 'hour']
    sales = sales.groupby(columns_to_group_by, as_index=False).agg({
        'units': 'sum',
        'total_discount': 'sum',
        'gross_total': 'sum',
        'net_total': 'sum'
    })
    
    # Reorder the sales dataframe columns
    sales = sales[[
        "sale_id", 
        "customer_id",
        "product_id",
        "store_id",
        "week_of_year",
        "day_of_week",
        "hour",
        "units",
        "gross_total",
        "was_in_promotion",
        "total_discount",
        "net_total"
    ]].sort_values(by="sale_id").reset_index(drop=True)
    
    # Refilter customers and products based on sales
    products = products[products.product_id.isin(sales.product_id)]
    customers = customers[customers.customer_id.isin(sales.customer_id)]
    
    return sales, customers, products

def create_cleaned_datasets(df):
    logger.info("Starting to create cleaned datasets.")
    
    # Preprocess the dataframe
    df = preprocess_dataframe(df)
    
    # create products dataset
    df, products = create_products(df)
    
    # Create customers datasets
    df, customers = create_customers(df)
    
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
