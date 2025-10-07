import pandas as pd
from pathlib import Path


def get_top_comments(df_comments):
    """Get the top comment (highest score) for each submission."""
    top_comments = (
        df_comments.sort_values("score", ascending=False)
        .groupby("submission_id")
        .first()
    )
    return top_comments


def process_language_data(
    language_code: str, language_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process data for a specific language."""
    submissions_file = f"data/aita_submissions_{language_code}.csv"
    comments_file = f"data/aita_comments_{language_code}.csv"

    print(f"\n{'='*60}")
    print(f"Processing {language_name} data")
    print(f"{'='*60}")

    # Check if files exist
    if not Path(submissions_file).exists():
        print(f"Warning: {submissions_file} not found, skipping...")
        return pd.DataFrame(), pd.DataFrame()

    if not Path(comments_file).exists():
        print(f"Warning: {comments_file} not found, skipping...")
        return pd.DataFrame(), pd.DataFrame()

    # Load data
    df_submissions = pd.read_csv(submissions_file)
    print(f"Total number of submissions: {len(df_submissions)}")

    df_comments = pd.read_csv(comments_file)
    print(f"Total number of comments: {len(df_comments)}")

    # Filter submissions by score
    df_submissions = df_submissions[df_submissions["score"] >= 25]
    print(f"Number of submissions with score >= 25: {len(df_submissions)}")

    # Filter submissions by text length
    df_submissions = df_submissions[df_submissions["text"].str.len() >= 300]
    print(f"Number of submissions with text length >= 300: {len(df_submissions)}")

    # Remove bot comments (language-specific bot names)
    bot_names = ["Judgement_Bot_AITA", "AutoModerator", "[deleted]", "[removed]"]
    df_comments = df_comments[~df_comments["author"].isin(bot_names)]
    print(f"Number of comments after removing bot comments: {len(df_comments)}")

    # Get top comments and filter by length
    top_comments = get_top_comments(df_comments)
    valid_submissions = top_comments[top_comments["text"].str.len() >= 25].index

    df_submissions = df_submissions[
        df_submissions["submission_id"].isin(valid_submissions)
    ]
    print(
        f"Number of submissions with top comments length >= 25: {len(df_submissions)}"
    )

    # Merge submissions with top comment IDs
    df_submissions = df_submissions.merge(
        top_comments[["comment_id"]],
        left_on="submission_id",
        right_index=True,
        how="left",
    )

    # Filter comments to only include those from valid submissions
    df_comments = df_comments[
        df_comments["submission_id"].isin(df_submissions["submission_id"])
    ]
    print(f"Number of comments after removing orphan comments: {len(df_comments)}")

    print(f"\nTotal number of submissions after filtering: {len(df_submissions)}")
    print(f"Total number of comments after filtering: {len(df_comments)}")

    return df_submissions, df_comments, top_comments


def create_dataset_file(
    df_submissions: pd.DataFrame, top_comments: pd.DataFrame, language_code: str
) -> None:
    """Create a dataset file with specified columns for a language."""
    # Create the dataset with required columns
    dataset = pd.DataFrame(
        {
            "submission_id": df_submissions["submission_id"],
            "title": df_submissions["title"],
            "selftext": df_submissions["text"],
            "created_utc": df_submissions["created_utc"],
            "permalink": df_submissions["permalink"],
            "score": df_submissions["score"],
            "top_comment": df_submissions["comment_id"].map(
                lambda x: (
                    top_comments.loc[top_comments["comment_id"] == x, "text"].iloc[0]
                    if x in top_comments["comment_id"].values
                    else None
                )
            ),
        }
    )

    # Save the dataset
    filename = f"data/ethical_dilemmas_cleaned_{language_code}.csv"
    dataset.to_csv(filename, index=False)
    print(f"Created dataset file: {filename} with {len(dataset)} rows")


def process_all_international_data():
    """Process data for all international languages separately."""
    language_configs = {
        "br": "Portuguese (Brazil)",
        "de": "German",
        "es": "Spanish",
        "fr": "French",
    }

    for language_code, language_name in language_configs.items():
        df_submissions, df_comments, top_comments = process_language_data(
            language_code, language_name
        )

        if not df_submissions.empty:
            # Create dataset file for this language
            create_dataset_file(df_submissions, top_comments, language_code)


if __name__ == "__main__":
    process_all_international_data()
