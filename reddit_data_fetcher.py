import asyncpraw
import pandas as pd
import os
from datetime import datetime, timezone, timedelta
from collections import deque
import asyncio

class RedditFetcher:
    def __init__(self, client_id, client_secret, user_agent, subreddit_names, save_path='./data'):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.subreddit_names = subreddit_names
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        # Rate limiting setup
        self.api_call_limit = 100  # Reddit OAuth limit: 100 calls per minute
        self.rate_limit_window = timedelta(seconds=60)
        self.api_call_times = deque()
        self.pause_indexes = deque()
        self.api_sleep_time = 1  # Small delay between calls

        # Data fetching settings
        self.new_limit_num = 1000  # Max Reddit API allows per endpoint
        self.df_columns = [
            'id', 'title', 'author', 'score', 'created_utc', 'num_comments',
            'url', 'selftext', 'created_at', 'scrape_time', 'search_term'
        ]
        self.dtformat = "%Y-%m-%d %H:%M:%S"

    async def connect(self):
        self.reddit = asyncpraw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )

    async def close(self):
        await self.reddit.close()

    async def __manage_api_call_rate(self):
        now = datetime.now()
        self.api_call_times.append(now)

        if len(self.api_call_times) >= self.api_call_limit * (len(self.pause_indexes) + 1):
            self.pause_indexes.append(len(self.api_call_times) - 1)

            if len(self.pause_indexes) == 0:
                wait_until = self.api_call_times[0] + self.rate_limit_window
            else:
                wait_until = self.api_call_times[self.pause_indexes[-1]] + self.rate_limit_window

            wait_time = (wait_until - now).total_seconds()

            if wait_time > 0:
                print(f"🔴 Rate limit reached! Pausing for {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
        else:
            await asyncio.sleep(self.api_sleep_time)

    async def fetch_subreddit_posts(self, subreddit_name, post_limit=200):
        print(f"\nFetching posts from r/{subreddit_name}...")

        subreddit = await self.reddit.subreddit(subreddit_name)
        posts = []
        post_counter = 0

        # Prepare history file path
        history_file = f'{subreddit_name}_posts.csv'
        file_path = os.path.join(self.save_path, history_file)

        # Load existing post history if available
        try:
            subreddit_df = pd.read_csv(file_path)
            seen_submission_ids = set(subreddit_df['id'])
            print(f"🗂️ Loaded {len(subreddit_df)} existing posts from {history_file}")
        except FileNotFoundError:
            subreddit_df = None
            seen_submission_ids = set()
            print(f"🆕 No history found for {subreddit_name}. Starting fresh...")

        # Begin fetching posts
        async for submission in subreddit.new(limit=None):  # Let asyncpraw handle paging
            await self.__manage_api_call_rate()

            # Skip duplicates
            if submission.id in seen_submission_ids:
                continue

            # Ensure full data is loaded
            await submission.load()

            # Build post data dictionary
            post_data = {
                'id': submission.id,
                'title': submission.title,
                'author': str(submission.author),
                'score': submission.score,
                'created_utc': datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                'num_comments': submission.num_comments,
                'url': submission.url,
                'selftext': submission.selftext,
                'created_at': datetime.utcfromtimestamp(submission.created_utc),
                'scrape_time': datetime.now().strftime(self.dtformat),
                'search_term': 'all_new_posts'
            }

            posts.append(post_data)
            seen_submission_ids.add(submission.id)
            post_counter += 1

            # Stop if we reached the post limit
            if post_counter >= post_limit:
                print(f"✅ Reached post limit ({post_limit}) for {subreddit_name}")
                break

        # Combine old + new data
        if posts:
            print(f"💾 Fetched {len(posts)} new posts from r/{subreddit_name}")

            new_posts_df = pd.DataFrame(posts)

            if subreddit_df is not None:
                combined_df = pd.concat([subreddit_df, new_posts_df], ignore_index=True)
            else:
                combined_df = new_posts_df

            # Drop any duplicates and save
            combined_df = combined_df.drop_duplicates(subset='id')
            combined_df.to_csv(file_path, index=False)

            print(f"✅ Saved {len(combined_df)} total posts to {file_path}")
        else:
            print(f"⚠️ No new posts found for r/{subreddit_name}")

    async def fetch_and_append_all(self, post_limit=200):
        await self.connect()

        for subreddit_name in self.subreddit_names:
            await self.fetch_subreddit_posts(subreddit_name, post_limit)

        await self.close()

    def combine_subreddit_files(self, combined_file_name='reddit_combined_posts.csv'):
        dfs = []

        for subreddit_name in self.subreddit_names:
            file_path = os.path.join(self.save_path, f'{subreddit_name}_posts.csv')

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                dfs.append(df)
            else:
                print(f"⚠️ No file found for {subreddit_name} at {file_path}")

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset='id')

            combined_path = os.path.join(self.save_path, combined_file_name)
            combined_df.to_csv(combined_path, index=False)

            print(f"\n✅ Combined dataset saved as: {combined_path} ({len(combined_df)} unique posts)")
        else:
            print("⚠️ No subreddit files to combine.")