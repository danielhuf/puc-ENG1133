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
        "submission_id": submission.id,
        "url": submission.url,
        "created_utc": datetime.fromtimestamp(submission.created_utc),
        "text": submission.selftext,
    }


def get_comment_data(comment, submission_id):
    """Extract relevant data from a comment."""
    return {
        "submission_id": submission_id,
        "comment_id": comment.id,
        "author": str(comment.author),
        "text": comment.body,
        "score": comment.score,
        "created_utc": datetime.fromtimestamp(comment.created_utc),
    }


def fetch_submissions_and_comments(subreddit_name="AmItheAsshole"):
    """Fetch the latest 1000 submissions and their top-level comments."""
    try:
        subreddit = reddit.subreddit(subreddit_name)
        subreddit.id
        print(f"Successfully connected to r/{subreddit_name}")
    except Exception as e:
        print(f"Error accessing subreddit: {str(e)}")
        return [], []

    submissions = []
    comments = []
    pbar = tqdm(
        total=1000, desc="Collecting submissions and comments", unit="submissions"
    )

    try:
        for submission in subreddit.new(limit=1000):
            submissions.append(get_submission_data(submission))
            submission.comments.replace_more(limit=None)

            for comment in submission.comments:
                if not comment.parent_id.startswith("t1_"):
                    comments.append(get_comment_data(comment, submission.id))

            pbar.update(1)

    except Exception as e:
        print(f"\nError while fetching data: {str(e)}")
        pbar.close()
        return submissions, comments

    pbar.close()
    return submissions, comments


def main():
    submissions, comments = fetch_submissions_and_comments()

    if submissions:
        df_submissions = pd.DataFrame(submissions)
        submissions_file = "aita_submissions.csv"
        df_submissions.to_csv(submissions_file, index=False)
        print(f"\nSaved {len(submissions)} submissions to {submissions_file}")

        if comments:
            df_comments = pd.DataFrame(comments)
            comments_file = "aita_comments.csv"
            df_comments.to_csv(comments_file, index=False)
            print(f"Saved {len(comments)} comments to {comments_file}")
        else:
            print("No comments found.")
    else:
        print("No submissions found.")


if __name__ == "__main__":
    main()

# https://www.reddit.com/r/pushshift/comments/1c2ndiu/confused_on_how_to_use_pushshift/
# https://www.reddit.com/r/pushshift/comments/14ei799/pushshift_live_again_and_how_moderators_can/
# https://www.reddit.com/r/redditdev/comments/17fksud/get_posts_from_certain_dates_praw/
# https://www.reddit.com/r/pushshift/comments/148fv2n/not_able_to_retrieve_reddit_submissions_and/
