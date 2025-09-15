#!/usr/bin/env python3
import pandas as pd
from pathlib import Path


def process_csv_data(input_file: str, output_file: str) -> None:
    """
    Remove specified columns from CSV file and save to new file.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")

    # Remove columns
    columns_to_remove = ["gpt4_label_3", "gpt4_reason_3"]
    df_cleaned = df.drop(columns=columns_to_remove)

    # Drop rows with empty mistral_reason columns
    mistral_columns = ["mistral_reason_1", "mistral_reason_2", "mistral_reason_3"]
    df_cleaned = df_cleaned.dropna(subset=mistral_columns)

    print(f"Final shape: {df_cleaned.shape}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving cleaned data to {output_file}...")
    df_cleaned.to_csv(output_file, index=False)
    print("Data processing completed.")


def main() -> None:
    """Main function to process the data."""
    input_file = "data/normative_evaluation_everyday_dilemmas_dataset.csv"
    output_file = "data/normative_evaluation_everyday_dilemmas_dataset_cleaned.csv"

    try:
        process_csv_data(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"Error processing data: {e}")


if __name__ == "__main__":
    main()
