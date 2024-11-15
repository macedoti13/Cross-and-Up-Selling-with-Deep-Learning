from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


def main(ids: list[int]):
    df = read_data()

    # Generate GNN Features
    df = generate_gnn_features(df)

    # Generate Collaborative Filtering Features
    df = generate_collaborative_filtering_features(df)

    # Compute the average embedding of the purchased items
    purchase_embeddings = df[df["product_id"].isin(ids)].embedding.tolist()
    purchase_embedding = np.mean(purchase_embeddings, axis=0)

    # Common Settings
    column_names = ["product_id", "product_name", "product_price", "similarity"]
    n_recommendations = 10

    # Generate Up-selling Lists for each product
    for product_id in ids:
        upselling_list = generate_upselling_list(df, product_id)
        product = df[df["product_id"] == product_id].iloc[0]
        if not upselling_list.empty:
            print(f"\nUp-selling Recommendations for Product {product_id}:")
            print(f"{product['product_name']} - {product['product_price']}")
            print(upselling_list[column_names].head(n_recommendations))
        else:
            print(f"\nNo up-selling recommendations found for Product ID {product_id}")

    # Generate Cross-selling List
    cross_selling_list = generate_cross_selling_list(df, purchase_embedding, ids)

    # Generic Recommendations based on purchase embedding
    generic_recommendations = generate_generic_recommendations(
        df, purchase_embedding, ids
    )

    print("\nCross-selling Recommendations:")
    print(cross_selling_list[column_names].head(n_recommendations))

    print("\nGeneric Recommendations:")
    print(generic_recommendations[column_names].head(n_recommendations))


def read_data() -> pd.DataFrame:
    data_path = Path(__file__).parents[1] / "data"
    embeddings_path = data_path / "transformed" / "products_embeddings.parquet"
    products_path = data_path / "cleaned" / "products.parquet"

    embeddings = pd.read_parquet(embeddings_path)
    products = pd.read_parquet(products_path)

    df = products.merge(embeddings, on="product_id", how="inner")
    assert len(df) == len(products) and len(df) == len(embeddings)
    return df


def generate_gnn_features(df: pd.DataFrame) -> pd.DataFrame:
    # Placeholder function for generating GNN features
    # TODO: Implement GNN feature generation logic
    return df


def generate_collaborative_filtering_features(df: pd.DataFrame) -> pd.DataFrame:
    # Placeholder function for generating Collaborative Filtering features
    # TODO: Implement Collaborative Filtering feature generation logic
    return df


def generate_upselling_list(df: pd.DataFrame, product_id: int) -> pd.DataFrame:
    product = df[df["product_id"] == product_id].iloc[0]
    subcategory = product["product_subcategory"]
    price = product["product_price"]
    embedding = product["embedding"]

    # Calculate price range for upselling (15% to 30% higher)
    price_lower_bound = price * 1.15
    price_upper_bound = price * 1.30

    # Filter products in the same subcategory with higher price
    candidates = df[
        (df["product_subcategory"] == subcategory)
        & (df["product_price"] >= price_lower_bound)
        # & (df["product_price"] <= price_upper_bound)
        & (df["product_id"] != product_id)
    ].copy()

    if not candidates.empty:
        # Compute similarity
        candidates["similarity"] = candidates["embedding"].apply(  # type: ignore
            lambda x: cosine_similarity(x.reshape(1, -1), embedding.reshape(1, -1))[0][
                0
            ]
        )
        # Sort candidates by similarity
        candidates = candidates.sort_values("similarity", ascending=False)  # type: ignore
        return candidates
    else:
        return pd.DataFrame()  # Return empty DataFrame if no candidates found


def generate_cross_selling_list(
    df: pd.DataFrame, purchase_embedding: np.ndarray, ids: list[int]
) -> pd.DataFrame:
    # Exclude products already purchased and from the same subcategories
    purchased_subcategories = df[df["product_id"].isin(ids)][
        "product_subcategory"
    ].unique()  # type: ignore

    candidates = df[
        # (~df["product_subcategory"].isin(purchased_subcategories))  # type: ignore
        # & (~df["product_id"].isin(ids))
        (~df["product_id"].isin(ids))
    ].copy()

    # Compute similarity
    candidates["similarity"] = candidates["embedding"].apply(  # type: ignore
        lambda x: cosine_similarity(
            x.reshape(1, -1), purchase_embedding.reshape(1, -1)
        )[0][0]
    )

    # Sort candidates by similarity
    candidates = candidates.sort_values("similarity", ascending=False)  # type: ignore
    return candidates


def generate_generic_recommendations(
    df: pd.DataFrame, purchase_embedding: np.ndarray, ids: list[int]
) -> pd.DataFrame:
    # Exclude already purchased items
    candidates = df[~df["product_id"].isin(ids)].copy()

    # Compute similarity
    candidates["similarity"] = candidates["embedding"].apply(  # type: ignore
        lambda x: cosine_similarity(
            x.reshape(1, -1), purchase_embedding.reshape(1, -1)
        )[0][0]
    )

    # Sort candidates by similarity
    candidates = candidates.sort_values("similarity", ascending=False)  # type: ignore
    return candidates


if __name__ == "__main__":
    # Fixed IDs for testing
    items = {
        "YOPRO 250ML BANANA 15G PROTEINA DANONE": 10023281,
        "ISOTONICO POWERADE LIMAO 500ML": 10106063,
        "AMOXICILINA": 1000096,
        "PREDNISOLONA": 10005454,
        "PARACETAMOL": 10022402,
    }
    ids = list(items.values())
    main(ids)
