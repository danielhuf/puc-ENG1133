import pandas as pd
import os
import re

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
    "moral_dilemmas_br.csv",
    "moral_dilemmas_de.csv",
    "moral_dilemmas_es.csv",
    "moral_dilemmas_fr.csv",
]

COLUMNS_TO_CLEAN = [
    "llama_reason_1",
    "llama_reason_2",
    "mistral_reason_1",
    "mistral_reason_2",
    "gemma_reason_1",
    "gemma_reason_2",
]

REDDIT_LABEL_MAPPINGS = {
    "br": {
        "EOB": "YTA",
        "EOT": "YTA",
        "NEOB": "NTA",
        "NGM": "NAH",
        "TEOB": "ESH",
        "INFO": "INFO",
    },
    "de": {"BDA": "YTA", "NDA": "NTA", "KAH": "NAH", "ASA": "ESH", "INFO": "INFO"},
    "es": {},
    "fr": {"TTB": "YTA", "PTB": "NTA", "ATB": "NAH", "TLM": "ESH", "INFO": "INFO"},
}


def clean_text(text):
    """
    Clean text by removing unwanted words, symbols, and formatting.

    Args:
        text: string to clean

    Returns:
        Cleaned plain text string
    """
    if pd.isna(text) or not isinstance(text, str):
        return text

    # Remove brackets: {}, [], ()
    text = re.sub(r"[{}\[\]()]", "", text)

    # Remove markdown indicators: **, ##, ```, #
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"#+", "", text)

    # Remove "Objeto", "JSON" and similar prefixes (case-insensitive)
    text = re.sub(r"\bobje[ct]o\s*json\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bjson\b", "", text, flags=re.IGNORECASE)

    # Remove "verdict", "verdicte", "veredito" (case-insensitive - multiple languages)
    text = re.sub(r"\bvere?d[iu][ct]te?o?\b", "", text, flags=re.IGNORECASE)

    # Remove "reasoning" and variations like "reasonning" (case-insensitive)
    text = re.sub(r"\breason[ni]+ng?\b", "", text, flags=re.IGNORECASE)

    # Remove specific verdict labels: YTA, NTA, ESH, NAH, INFO (case-insensitive)
    text = re.sub(r"\b(YTA|NTA|ESH|NAH|INFO)\b", "", text, flags=re.IGNORECASE)

    # Remove all types of quotation marks (straight and curly) and colons
    text = re.sub(r'["\'\u201c\u201d\u2018\u2019:]', "", text)

    # Remove common punctuation artifacts at the beginning/end
    text = re.sub(r"^[\s,;]+", "", text)
    text = re.sub(r"[\s,;]+$", "", text)

    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def extract_reddit_label(comment_text, language_code):
    """
    Extract reddit label from comment text based on language-specific mappings.

    Args:
        comment_text: string containing the comment text
        language_code: language code ('br', 'de', 'es', 'fr')

    Returns:
        English reddit label or None if no match found
    """
    if pd.isna(comment_text) or not isinstance(comment_text, str):
        return None

    label_mapping = REDDIT_LABEL_MAPPINGS.get(language_code, {})

    # Spanish case
    if not label_mapping:
        return None

    comment_upper = comment_text.upper()

    for local_label, english_label in label_mapping.items():
        if local_label.upper() in comment_upper:
            return english_label

    return None


def clean_dataframe(df, filename):
    """
    Clean a dataframe by removing rows where any reason column has a null value,
    cleaning malformed text in specific columns, and adding reddit_label column.

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

    columns_to_clean = [col for col in COLUMNS_TO_CLEAN if col in df_cleaned.columns]
    for col in columns_to_clean:
        df_cleaned[col] = df_cleaned[col].apply(clean_text)

    language_code = filename.replace("moral_dilemmas_", "").replace(".csv", "")
    df_cleaned["reddit_label"] = df_cleaned["top_comment"].apply(
        lambda x: extract_reddit_label(x, language_code)
    )

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

        df_cleaned = clean_dataframe(df, csv_file)

        parts = csv_file.replace(".csv", "").rsplit("_", 1)
        output_filename = f"{parts[0]}_cleaned_{parts[1]}.csv"
        output_path = os.path.join(data_folder, output_filename)
        df_cleaned.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
