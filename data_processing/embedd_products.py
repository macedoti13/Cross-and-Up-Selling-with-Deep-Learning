from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
import os

def main():
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    transformed_data_path = os.path.join(os.path.dirname(os.getcwd()), 'SJ_PCD_24-2', 'data', 'transformed')
    products = pd.read_parquet(os.path.join(transformed_data_path, 'products_descriptions.parquet'))
    
    products["embedding"] = None
    for index, product in tqdm(products.iterrows()):
        embedding = model.encode(product["description"])
        products.at[index, "embedding"] = embedding
        
    products.to_parquet(os.path.join(transformed_data_path, 'products_embeddings.parquet'), index=False)

if __name__ == "__main__":
    main()
