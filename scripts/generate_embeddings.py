#!/usr/bin/env python3
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os


def get_embedding(text, model):
    """Generate embedding using Sentence Transformers."""
    if pd.isna(text):
        return None
    return model.encode(text)


def main() -> None:
    """Main function to generate embeddings."""
    print("Loading Sentence Transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Reading CSV file...")
    df = pd.read_csv("data/ethical_dilemmas_cleaned.csv")

    columns_to_embed = [
        "selftext",
        "top_comment",
        "gpt3.5_reason_1",
        "gpt3.5_reason_2",
        "gpt3.5_reason_3",
        "gpt4_reason_1",
        "gpt4_reason_2",
        "claude_reason_1",
        "claude_reason_2",
        "claude_reason_3",
        "bison_reason_1",
        "bison_reason_2",
        "bison_reason_3",
        "gemma_reason_1",
        "gemma_reason_2",
        "gemma_reason_3",
        "gemma_reason_4",
        "gemma_reason_5",
        "mistral_reason_1",
        "mistral_reason_2",
        "mistral_reason_3",
        "llama_reason_1",
        "llama_reason_2",
        "llama_reason_3",
    ]

    checkpoint_file = "data/embeddings_checkpoint.csv"
    if os.path.exists(checkpoint_file):
        print("Found checkpoint file. Loading previous progress...")
        embeddings_df = pd.read_csv(checkpoint_file)
        processed_columns = [
            col.replace("_embedding", "")
            for col in embeddings_df.columns
            if col.endswith("_embedding")
        ]
        columns_to_embed = [
            col for col in columns_to_embed if col not in processed_columns
        ]
        print(
            f"Resuming from column: {columns_to_embed[0] if columns_to_embed else 'All columns completed.'}"
        )
    else:
        print("No checkpoint found. Starting from scratch...")
        embeddings_df = pd.DataFrame()
        embeddings_df["submission_id"] = df["submission_id"]

    print("Generating embeddings with Sentence Transformers...")
    for col in tqdm(columns_to_embed, desc="Processing columns"):
        if col in df.columns:
            embedding_col = f"{col}_embedding"
            embeddings = []
            for text in tqdm(df[col], desc=f"Processing rows in {col}", leave=False):
                embedding = get_embedding(text, model)
                embeddings.append(embedding)
            embeddings_df[embedding_col] = embeddings

            print(f"\nSaving checkpoint after completing {col}...")
            embeddings_df.to_csv(checkpoint_file, index=False)
            print(f"Checkpoint saved.")

    print("All columns processed.")
    embeddings_df.to_csv("data/embeddings.csv", index=False)
    print(
        "Sentence Transformers embeddings saved to 'data/embeddings.csv'"
    )


if __name__ == "__main__":
    main()
