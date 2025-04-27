import praw
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=f"script:aita_scraper:v1.0 (by /u/{os.getenv('REDDIT_USERNAME')})",
)


def get_submission_data(submission):
    """Extract relevant data from a submission."""
    return {
        "url": submission.url,
        "created_utc": datetime.fromtimestamp(submission.created_utc),
        "text": submission.selftext,
    }


def fetch_submissions(subreddit_name="AmItheAsshole"):
    """Fetch the latest 1000 submissions."""
    try:
        subreddit = reddit.subreddit(subreddit_name)
        subreddit.id
        print(f"Successfully connected to r/{subreddit_name}")
    except Exception as e:
        print(f"Error accessing subreddit: {str(e)}")
        return []

    submissions = []
    pbar = tqdm(total=1000, desc="Collecting submissions", unit="submissions")

    try:
        for submission in subreddit.new(limit=1000):
            submissions.append(get_submission_data(submission))
            pbar.update(1)

    except Exception as e:
        print(f"\nError while fetching submissions: {str(e)}")
        pbar.close()
        return submissions

    pbar.close()
    return submissions


def main():
    submissions = fetch_submissions()

    if submissions:
        df = pd.DataFrame(submissions)
        output_file = "aita_submissions.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved {len(submissions)} submissions to {output_file}")
    else:
        print("No submissions found.")


if __name__ == "__main__":
    main()
