import asyncpraw
import pandas as pd
import os
from datetime import datetime, timezone, timedelta
from collections import deque
import asyncio

class RedditFetcher:
    def __init__(self, client_id, client_secret, user_agent, subreddit_names, search_terms,
                 posts_file_path='./data', limit_num=100, search_time_filter='all', df_columns=None, dtformat="%Y-%m-%d %H:%M:%S"):
        
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.subreddit_names = subreddit_names
        self.search_terms = search_terms
        self.posts_file_path = posts_file_path
        self.limit_num = limit_num
        self.search_time_filter = search_time_filter
        self.df_columns = df_columns or ['id', 'title', 'author', 'score', 'num_comments', 'url', 'selftext',
                                         'created_at', 'scrape_time', 'search_term']
        self.dtformat = dtformat

        os.makedirs(self.posts_file_path, exist_ok=True)

        self.api_call_times = deque()
        self.pause_indexes = deque()

        self.api_call_limit = 60  # Max API calls per time window
        self.rate_limit_window = timedelta(minutes=1)
        self.api_sleep_time = 1  # Sleep between calls in seconds

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

    async def fetch_search_data(self):
        '''
        Fetch Reddit data using the search method.
        '''
        print("🚀 Starting Reddit search fetch...")

        # Initialize asyncpraw reddit object
        reddit = asyncpraw.Reddit(client_id=self.client_id,
                                  client_secret=self.client_secret,
                                  user_agent=self.user_agent)

        for subreddit_name in self.subreddit_names:
            print(f"\n🔎 Searching in r/{subreddit_name}")

            subreddit = await reddit.subreddit(subreddit_name)

            history_file = f'{subreddit_name}_posts_data.csv'
            file_path = os.path.join(self.posts_file_path, history_file)

            # Load previous posts if they exist
            try:
                subreddit_df = pd.read_csv(file_path)
                seen_submission_ids = set(subreddit_df['id'])
                print(f"🗂️ Loaded {len(subreddit_df)} existing posts from {history_file}")
            except FileNotFoundError:
                subreddit_df = None
                seen_submission_ids = set()
                print(f"🆕 No history file found. Starting fresh.")

            subreddit_data = []

            # Search for each term in the list
            for search_term in self.search_terms:
                print(f"\n🔍 Searching for: {search_term}")

                found_post_count = 0

                try:
                    async for submission in subreddit.search(search_term,
                                                             limit=self.limit_num,
                                                             time_filter=self.search_time_filter):
                        # Handle rate limit
                        await self.__manage_api_call_rate()

                        if submission.id in seen_submission_ids:
                            continue

                        await submission.load()

                        # Build data dict
                        sub_dict = {}
                        for col in self.df_columns:
                            if col == "created_at":
                                sub_dict[col] = datetime.utcfromtimestamp(int(getattr(submission, "created_utc")))
                            elif col == "scrape_time":
                                sub_dict[col] = datetime.now().strftime(self.dtformat)
                            elif col == "search_term":
                                sub_dict[col] = search_term
                            else:
                                sub_dict[col] = getattr(submission, col, None)

                        subreddit_data.append(sub_dict)
                        seen_submission_ids.add(submission.id)

                        # print(f"✅ Found post: {submission.title[:60]}...")
                        found_post_count += 1

                except Exception as e:
                    print(f"❌ Error fetching posts for search term '{search_term}': {e}")
                    raise RuntimeError(e)
                
                print(f"✅ Found {found_post_count} new posts for search term '{search_term}'.")

            # Save posts if we got new ones
            if subreddit_data:
                print(f"\n💾 Fetched {len(subreddit_data)} new posts for r/{subreddit_name}.")

                new_data_df = pd.DataFrame(subreddit_data)

                if subreddit_df is not None:
                    combined_df = pd.concat([subreddit_df, new_data_df], ignore_index=True)
                else:
                    combined_df = new_data_df

                combined_df = combined_df.drop_duplicates(subset='id')

                combined_df.to_csv(file_path, index=False)

                print(f"✅ Saved {len(combined_df)} total posts to {file_path} (including previous and new posts).")
            else:
                print(f"⚠️ No new posts found for r/{subreddit_name}.")

        print("\n✅ Search fetch complete.")
        await reddit.close()

    def combine_subreddit_files(self, combined_file_name='reddit_combined_posts.csv'):
        dfs = []

        # Use the same file naming convention as fetch_search_data()
        for subreddit_name in self.subreddit_names:
            file_path = os.path.join(self.posts_file_path, f'{subreddit_name}_posts_data.csv')

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                dfs.append(df)
                print(f"✅ Loaded {len(df)} posts from {file_path}")
            else:
                print(f"⚠️ No file found for {subreddit_name} at {file_path}")

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset='id')

            combined_path = os.path.join(self.posts_file_path, combined_file_name)
            combined_df.to_csv(combined_path, index=False)

            print(f"\n✅ Combined dataset saved as: {combined_path} ({len(combined_df)} unique posts)")
        else:
            print("⚠️ No subreddit files to combine.")
