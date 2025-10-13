import pandas as pd
import os

COLUMNS_TO_CHECK = [
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

CSV_FILES = [
    "ethical_dilemmas_cleaned_br.csv",
    "ethical_dilemmas_cleaned_de.csv",
    "ethical_dilemmas_cleaned_es.csv",
    "ethical_dilemmas_cleaned_fr.csv",
]


def clean_dataframe(df, filename):
    """
    Clean a dataframe by removing rows where any reason column has a null value.

    Args:
        df: pandas DataFrame to clean
        filename: name of the file (for display purposes)

    Returns:
        Cleaned pandas DataFrame
    """
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    initial_rows = len(df)
    print(f"Initial number of rows: {initial_rows}")

    existing_columns = [col for col in COLUMNS_TO_CHECK if col in df.columns]

    df_cleaned = df.dropna(subset=existing_columns).copy()

    final_rows = len(df_cleaned)

    print(f"Final number of rows: {final_rows}")

    return df_cleaned


def main():
    """Main function to process all CSV files."""
    data_folder = "data"

    for csv_file in CSV_FILES:
        file_path = os.path.join(data_folder, csv_file)

        if not os.path.exists(file_path):
            print(f"\nWarning: File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)

        clean_dataframe(df, csv_file)

    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
