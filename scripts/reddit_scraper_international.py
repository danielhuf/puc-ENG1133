import praw
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=f"script:international_aita_scraper:v1.0 (by /u/{os.getenv('REDDIT_USERNAME')})",
)

# Configuration for each subreddit
SUBREDDIT_CONFIG = {
    "br": {
        "name": "EuSouOBabaca",
        "submissions_file": "data/aita_submissions_br.csv",
        "comments_file": "data/aita_comments_br.csv",
        "display_name": "Portuguese (Brazil)",
    },
    "de": {
        "name": "BinIchDasArschloch",
        "submissions_file": "data/aita_submissions_de.csv",
        "comments_file": "data/aita_comments_de.csv",
        "display_name": "German",
    },
    "es": {
        "name": "soyculero",
        "submissions_file": "data/aita_submissions_es.csv",
        "comments_file": "data/aita_comments_es.csv",
        "display_name": "Spanish",
    },
    "fr": {
        "name": "suisjeletroudeballe",
        "submissions_file": "data/aita_submissions_fr.csv",
        "comments_file": "data/aita_comments_fr.csv",
        "display_name": "French",
    },
}


def get_submission_data(submission: praw.models.Submission) -> Dict:
    """Extract relevant data from a submission."""
    return {
        "submission_id": submission.id,
        "title": submission.title,
        "text": submission.selftext,
        "score": submission.score,
        "permalink": submission.permalink,
        "created_utc": datetime.fromtimestamp(submission.created_utc),
    }


def get_comment_data(comment: praw.models.Comment, submission_id: str) -> Dict:
    """Extract relevant data from a comment."""
    return {
        "submission_id": submission_id,
        "comment_id": comment.id,
        "author": str(comment.author) if comment.author else "[deleted]",
        "text": comment.body,
        "score": comment.score,
        "created_utc": datetime.fromtimestamp(comment.created_utc),
    }


def load_existing_data(
    submissions_file: str, comments_file: str
) -> Tuple[Set[str], Set[str], Optional[datetime]]:
    """Load existing submission and comment IDs to avoid duplicates."""
    existing_submission_ids = set()
    existing_comment_ids = set()
    newest_submission_date = None

    if Path(submissions_file).exists():
        df_existing = pd.read_csv(submissions_file)
        if not df_existing.empty:
            existing_submission_ids = set(df_existing["submission_id"].tolist())
            # Get the newest submission date to know where to start fetching from
            df_existing["created_utc"] = pd.to_datetime(df_existing["created_utc"])
            newest_submission_date = df_existing["created_utc"].max()
            print(f"Found {len(existing_submission_ids)} existing submissions")
            print(f"Newest existing submission: {newest_submission_date}")

    if Path(comments_file).exists():
        df_existing = pd.read_csv(comments_file)
        if not df_existing.empty:
            existing_comment_ids = set(df_existing["comment_id"].tolist())
            print(f"Found {len(existing_comment_ids)} existing comments")

    return existing_submission_ids, existing_comment_ids, newest_submission_date


def fetch_submissions_and_comments(
    subreddit_name: str,
    max_posts: int = 10000,
    existing_submission_ids: Optional[Set[str]] = None,
    existing_comment_ids: Optional[Set[str]] = None,
    newest_submission_date: Optional[datetime] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Fetch submissions and their top-level comments from the subreddit."""
    if existing_submission_ids is None:
        existing_submission_ids = set()
    if existing_comment_ids is None:
        existing_comment_ids = set()

    try:
        subreddit = reddit.subreddit(subreddit_name)
        subreddit.id
        print(f"Successfully connected to r/{subreddit_name}")
    except Exception as e:
        print(f"Error accessing subreddit: {str(e)}")
        return [], []

    submissions = []
    comments = []
    new_submission_count = 0

    # Use a larger limit to get more posts, focusing on recent ones
    pbar = tqdm(
        total=max_posts, desc=f"Collecting from r/{subreddit_name}", unit="submissions"
    )

    try:
        # Get posts from multiple sorting methods to maximize coverage
        submission_sources = [
            ("new", subreddit.new(limit=max_posts // 3)),
            ("hot", subreddit.hot(limit=max_posts // 3)),
            ("top", subreddit.top(time_filter="month", limit=max_posts // 3)),
        ]

        seen_submission_ids = set()

        for source_name, submission_generator in submission_sources:
            print(f"\nFetching from {source_name} posts...")
            source_count = 0
            skipped_count = 0

            for submission in submission_generator:
                # Skip if we already have this submission
                if (
                    submission.id in existing_submission_ids
                    or submission.id in seen_submission_ids
                ):
                    skipped_count += 1
                    continue

                # Only apply date filtering to "new" posts
                # For "hot" and "top", we want popular/rated posts regardless of age
                if source_name == "new" and (
                    newest_submission_date
                    and datetime.fromtimestamp(submission.created_utc)
                    <= newest_submission_date
                ):
                    skipped_count += 1
                    continue

                seen_submission_ids.add(submission.id)
                submissions.append(get_submission_data(submission))
                new_submission_count += 1

                # Get comments for this submission
                submission.comments.replace_more(limit=None)
                comment_count = 0

                for comment in submission.comments:
                    # Only get top-level comments (not replies to comments)
                    if not comment.parent_id.startswith("t1_"):
                        # Skip if we already have this comment
                        if comment.id not in existing_comment_ids:
                            comments.append(get_comment_data(comment, submission.id))
                            comment_count += 1

                source_count += 1
                pbar.update(1)

                # Break if we've reached our target
                if new_submission_count >= max_posts:
                    break

            print(
                f"Collected {source_count} new submissions from {source_name} (skipped {skipped_count} existing)"
            )

            if new_submission_count >= max_posts:
                break

    except Exception as e:
        print(f"\nError while fetching data: {str(e)}")
        pbar.close()
        return submissions, comments

    pbar.close()
    print(
        f"\nTotal collected: {len(submissions)} new submissions, {len(comments)} new comments"
    )
    return submissions, comments


def save_data(
    submissions: List[Dict],
    comments: List[Dict],
    submissions_file: str,
    comments_file: str,
) -> None:
    """Save submissions and comments to CSV files."""
    if submissions:
        df_new_submissions = pd.DataFrame(submissions)

        # Check if file exists and load existing data
        if Path(submissions_file).exists():
            df_existing_submissions = pd.read_csv(submissions_file)
            # Combine new and existing data (no duplicates expected due to pre-filtering)
            df_combined_submissions = pd.concat(
                [df_new_submissions, df_existing_submissions], ignore_index=True
            )
        else:
            df_combined_submissions = df_new_submissions

        df_combined_submissions.to_csv(submissions_file, index=False)
        print(
            f"\nSaved {len(df_combined_submissions)} total submissions to {submissions_file}"
        )

        if comments:
            df_new_comments = pd.DataFrame(comments)

            if Path(comments_file).exists():
                df_existing_comments = pd.read_csv(comments_file)
                # Combine new and existing data (no duplicates expected due to pre-filtering)
                df_combined_comments = pd.concat(
                    [df_new_comments, df_existing_comments], ignore_index=True
                )
            else:
                df_combined_comments = df_new_comments

            df_combined_comments.to_csv(comments_file, index=False)
            print(
                f"Saved {len(df_combined_comments)} total comments to {comments_file}"
            )
        else:
            print("No new comments found.")
    else:
        print("No new submissions found.")


def scrape_subreddit(language_code: str, max_posts: int = 10000) -> None:
    """Scrape a specific subreddit."""
    config = SUBREDDIT_CONFIG[language_code]
    subreddit_name = config["name"]
    submissions_file = config["submissions_file"]
    comments_file = config["comments_file"]
    display_name = config["display_name"]

    print(f"\n{'='*60}")
    print(
        f"Starting scrape of r/{subreddit_name} ({display_name}) - max {max_posts} posts"
    )
    print(f"{'='*60}")

    # Load existing data to avoid duplicates
    existing_submission_ids, existing_comment_ids, newest_submission_date = (
        load_existing_data(submissions_file, comments_file)
    )

    # Fetch only new data
    new_submissions, new_comments = fetch_submissions_and_comments(
        subreddit_name=subreddit_name,
        max_posts=max_posts,
        existing_submission_ids=existing_submission_ids,
        existing_comment_ids=existing_comment_ids,
        newest_submission_date=newest_submission_date,
    )

    # Save the data
    save_data(new_submissions, new_comments, submissions_file, comments_file)


def main():
    """Main function to scrape all configured subreddits."""
    # Set a high limit to get maximum posts
    max_posts = 10000

    print("International AITA Subreddit Scraper")
    print("=" * 50)
    print(
        f"Configured subreddits: {', '.join([f'{config['display_name']} (r/{config['name']})' for config in SUBREDDIT_CONFIG.values()])}"
    )
    print(f"Max posts per subreddit: {max_posts}")
    print("=" * 50)

    # Scrape each subreddit
    for language_code in SUBREDDIT_CONFIG.keys():
        try:
            scrape_subreddit(language_code, max_posts)
        except Exception as e:
            print(f"\nError scraping {language_code}: {str(e)}")
            continue

    print(f"\n{'='*60}")
    print("Scraping completed for all subreddits!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
