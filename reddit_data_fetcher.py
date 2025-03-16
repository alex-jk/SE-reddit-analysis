import asyncpraw
import pandas as pd
import os
from datetime import datetime, timezone

class RedditFetcher:
    def __init__(self, client_id, client_secret, user_agent, subreddit_names, save_path='./data'):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.subreddit_names = subreddit_names
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    async def connect(self):
        self.reddit = asyncpraw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )

    async def close(self):
        await self.reddit.close()

    async def fetch_subreddit_posts(self, subreddit_name, post_limit=100):
        print(f"\nFetching posts from r/{subreddit_name}...")

        subreddit = await self.reddit.subreddit(subreddit_name)
        posts = []

        async for submission in subreddit.new(limit=post_limit):
            post_data = {
                'id': submission.id,
                'title': submission.title,
                'author': str(submission.author),
                'score': submission.score,
                'created_utc': datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                'num_comments': submission.num_comments,
                'url': submission.url,
                'selftext': submission.selftext
            }
            posts.append(post_data)

        return pd.DataFrame(posts)

    async def fetch_and_append_all(self, post_limit=100):
        await self.connect()

        for subreddit_name in self.subreddit_names:
            new_posts_df = await self.fetch_subreddit_posts(subreddit_name, post_limit)

            file_path = os.path.join(self.save_path, f'{subreddit_name}_posts.csv')

            if os.path.exists(file_path):
                existing_df = pd.read_csv(file_path)
                combined_df = pd.concat([existing_df, new_posts_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset='id')
            else:
                combined_df = new_posts_df

            combined_df.to_csv(file_path, index=False)
            print(f"Saved {len(combined_df)} posts to {file_path}")

        await self.close()

    def combine_subreddit_files(self, combined_file_name='reddit_posts.csv'):
        dfs = []

        for subreddit_name in self.subreddit_names:
            file_path = os.path.join(self.save_path, f'{subreddit_name}_posts.csv')

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                dfs.append(df)
            else:
                print(f"No file found for {subreddit_name} at {file_path}")

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset='id')

            combined_path = os.path.join(self.save_path, combined_file_name)
            combined_df.to_csv(combined_path, index=False)

            print(f"\n✅ Combined dataset saved as: {combined_path} ({len(combined_df)} unique posts)")
        else:
            print("No subreddit files to combine.")