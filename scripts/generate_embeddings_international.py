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


def process_dataset(language_code, model):
    """Process a single language dataset and generate embeddings."""
    print(f"\n=== Processing {language_code.upper()} dataset ===")

    input_file = f"data/moral_dilemmas_cleaned_{language_code}.csv"
    output_file = f"data/embeddings_{language_code}.csv"
    checkpoint_file = f"data/embeddings_{language_code}_checkpoint.csv"

    df = pd.read_csv(input_file)

    columns_to_embed = [
        "selftext",
        "top_comment",
        "gpt3.5_reason_1",
        "gpt3.5_reason_2",
        "gpt4_reason_1",
        "gpt4_reason_2",
        "claude_reason_1",
        "claude_reason_2",
        "gemini_reason_1",
        "gemini_reason_2",
        "llama_reason_1",
        "llama_reason_2",
        "mistral_reason_1",
        "mistral_reason_2",
        "gemma_reason_1",
        "gemma_reason_2",
    ]

    if os.path.exists(checkpoint_file):
        print(
            f"Found checkpoint file for {language_code}. Loading previous progress..."
        )
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
        print(f"No checkpoint found for {language_code}. Starting from scratch...")
        embeddings_df = pd.DataFrame()
        embeddings_df["submission_id"] = df["submission_id"]

    print(f"Generating embeddings for {language_code} dataset...")
    for col in tqdm(columns_to_embed, desc=f"Processing columns ({language_code})"):
        if col in df.columns:
            embedding_col = f"{col}_embedding"
            embeddings = []
            for text in tqdm(df[col], desc=f"Processing rows in {col}", leave=False):
                embedding = get_embedding(text, model)
                embeddings.append(embedding)
            embeddings_df[embedding_col] = embeddings

            print(f"\nSaving checkpoint after completing {col} for {language_code}...")
            embeddings_df.to_csv(checkpoint_file, index=False)
            print(f"Checkpoint saved for {language_code}.")

    print(f"All columns processed for {language_code}.")
    embeddings_df.to_csv(output_file, index=False)
    print(
        f"Sentence Transformers embeddings for {language_code} saved to '{output_file}'"
    )

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


def main() -> None:
    """Main function to generate embeddings for all international datasets."""
    print("Loading Sentence Transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    language_codes = ["br", "de", "es", "fr"]

    for lang_code in language_codes:
        process_dataset(lang_code, model)

    print("\n=== All international datasets processed successfully! ===")


if __name__ == "__main__":
    main()
