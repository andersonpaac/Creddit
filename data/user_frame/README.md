# User Frame

1. Every file here is a pickle of a dataframe corresponding to a post. Example: `b4agza_user_frame.pkl` corresponds
to all unique authors of comments posted in [http://redd.it/b4agza`]
2. Each row in the dataframe corresponds to an author. Each row has the following features
3. There are no duplicates.

#### Dataset generated by: `Creddit.Ingest.extractor.UserExtractor`

| Features                  | dType          | Example                                             | Inference                                                                      |   |
|---------------------------|----------------|-----------------------------------------------------|--------------------------------------------------------------------------------|---|
| author                    | string         | HereWeGoAgainTJ                                     | author of the comment                                                          |   |
| subreddit_comment_count   | Dict[str: int] | {"worldnews": 38, "confession": 6, ...}             | An aggregation of the comments made in various subreddits (Count)              |   |
| subreddit_comment_karma   | Dict[str: int] | {"worldnews": 333, "conspiracy": 31, ...}           | An aggregation of the karma from the comments made in various subreddits (Sum) |   |
| subreddit_post_count      | Dict[str: int] | {"bestoflegaladvice": 1, "SubredditDrama": 1, ...}  | An aggregation of the posts made in various subreddits (Count)                 |   |
| subreddit_post_karma      | Dict[str: int] | {"SubredditDrama": 32, "bestoflegaladvice": 2, ...} | An aggregation of the karma from the posts made in various subreddits (Sum)    |   |
| user_account_creation_utc | datetime64     | 2018-08-11 12:02:37                                 | When the users' account was created                                            |   |
| user_email_verified       | bool           | True                                                | If the user verified his email                                                 |   |
| user_total_comment_count  | int            | 993                                                 | Total number of comments made by the author on Reddit                          |   |
| user_total_comment_karma  | int            | 54048                                               | Total karma accumulated by the author from their comments on Reddit            |   |
| user_total_post_count     | int            | 13                                                  | Total number of posts made by the author on Reddit                             |   |
| user_total_post_karma     | int            | 226                                                 | Total karma accumulated by the author from their posts on Reddit               |   |
