import pandas as pd
import praw as pw
import datetime as dt
import os
from psaw import PushshiftAPI

api = PushshiftAPI()

start_epoch=int(dt.datetime(2022, 3, 1).timestamp())
end_epoch=int(dt.datetime(2022, 3, 27).timestamp())

df = pd.DataFrame(api.search_submissions(after=start_epoch, before=end_epoch,
                            filter=['author','title','selftext'],
                            subreddit='dpdr'))
                            
df.shape
df.columns
del df['d_']
del df['created']

os.makedirs('/dataset', exist_ok=True)
df.to_csv('dataset/dpdr.csv')

