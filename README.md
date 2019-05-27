## Objective

This is a repository containing the tools to Ingest, Analyze and train ML models to predict the mis-information 
presented in comments on [Reddit](https://reddit.com/)

Included is the discourse on 3 threads on [r/politics](http://www.reddit.com/r/politics)

1. [Megathread: Mueller files final report with Attorney General](http://redd.it/b4agza)
2. [Megathread: Attorney General Releases Redacted Version of Special Counsel Report](https://www.reddit.com/comments/bempai)
3. [Michael Cohen Testifies Before House Oversight Committee](https://www.reddit.com/comments/avdne2)

We section the data required into 2 Sets

### Comment Frame
Extracted by the `Ingest.CommentExtractor` into `data/comment_frame/{post_id}_comment_frame.pkl`
The pickle is of type `pandas.core.frame.DataFrame`
Each comment is encapsulated in a row and there are no duplicates
Contains 12 below features
    
| Features                | dType      | Example                     | Inference                                                                                                       |   |
|-------------------------|------------|-----------------------------|-----------------------------------------------------------------------------------------------------------------|---|
| author                  | string     | santaKlaus                  | author of the comment                                                                                           |   |
| body                    | string     | Now it's up to Congress ... | comment body                                                                                                    |   |
| comment_created_utc     | datetime64 | 2019-03-22 22:35:58         | comment creation time                                                                                           |   |
| comment_deleted         | bool       | False                       | if comment was deleted later by the user                                                                        |   |
| comment_id              | string     | ej5l32l                     | ID of the comment                                                                                               |   |
| comment_removed_by_mods | bool       | True                        | if the comment was removed by moderators                                                                        |   |
| edited                  | bool       | True                        | If the comment was edited                                                                                       |   |
| golds                   | int        | 0                           | Number of "gold"/"awards" received by the author for the comment                                                |   |
| is_submitter            | bool       | False                       | If the author of the comment is also the author of the post                                                     |   |
| parent_id               | string     | t3_b4agza                   | The comment_id that the comment is a response to. If the parent_id is the post_id, it is a "top-level" comment. |   |
| post_id                 | string     | b4agza                      | The post to which the comment belongs                                                                           |   |
| score                   | int        | 19                          | The "karma" / score is defined by upvotes - downvotes for the comment                                           |   |
    

### User Frame

Extracted by the `Ingest.UserExtractor` into `data/user_frame/{post_id}_user_frame.pkl`
The pickle is of type `pandas.core.frame.DataFrame`
Each user is encapsulated in a row and there are no duplicates
Contains 11 below features

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


### Training Data Construction Pipeline

You can skip step 1 and 2 if you intend to work with the existing dataset.

1.  Generate your credentials from apps [here](http://reddit.com/prefs/apps). Select the app-type to be "Script for personal use".

2.  Insert your `secret-key` and `client-id` into `reddit_api_creds.json`

3.  Call `Ingest.extractors.CommentExtractor` with a valid `post_id`. This will first pull a list of `comment_ids` associated
    with the `post_id` from [PushShift.io](https://pushshift.io/elasticsearch-quick-reference/). We do this because reddit's API's
    does not give access to several deleted and removed comments. PushShift still hosts these datasets.
    A post is is the base 36 id of the reddit post submission. 

4.  The earlier step gets all the comments for the associated `post_id`. 3. We augment the predictors in the `user_frame` by leveraging the `Analysis.FeatureBuilder.UserFeatureBuilder`.
    This class aggregates the user's historical activity in partisan subreddits along with r/news and r/politics. The 
    `Analysis/partisan_resource.json` is used to determine the partisan bias in subreddits.
    It also aggregates the post, comment karma and count. Sample usage is below
    
    ```python
    post_id, post_creation_time = 'b4agza', pd.to_datetime(get_reddit_instance().submission('b4agza').created_utc, unit='s')
    user_frame = Common.load_user_frame(post_id)
    userFeatureBuilder = UserFeatureBuilder(post_creation_time)
    user_features = Df(list(user_frame.apply(userFeatureBuilder.get_features_for_row, axis=1)))
    ```

5.  Similarly, we augment the features in the 'comment_frame' with the `Analysis.FeatureBuilder.PostFeatureBuilder`.
    This will generate network based features for every comment that is passed in. Network features include 
    network_comment_thread_max_depth, network_comment_thread_top_level_count, network_comment_thread_size. See 
    `CommentNetworkFeatureBuilder` class for more documentation.
    Features on the sentiment of the comment body is capture by profanity, polarity, objectivity, spelling errors by the
    `CommentTextFeatureBuilder` class.
    Additional features such as `timedelta` between the comment creation and post creation are also captured.
    Sample usage is below

    ```python
    comment_frame = Common.load_comment_frame(post_id)
    top_level_frame = Common.get_top_level_comments(comment_frame)
    postFeatureBuilder = PostFeatureBuilder(comment_frame)
    post_features = Df(list(top_level_frame.progress_apply(postFeatureBuilder.get_features_for_row, axis=1)))
    ```

6.  We then stitch the user_frame, comment_frame along with all augmented features together to form a single dataset. 
    This dataset contains 59 columns. The code for generating this dataset is in `Analysis/featureGenScript.py`. The dataset
    can contain data from multiple posts. For example: the frame in `data/dataset/combined.pkl` is a dataset merging the 
    top level comments from the 3 posts mentioned above.

7.  The training and the inference of the models can be found in the jupyter notebook `Model.ipynb` 
    and in the conclusion of the paper `cReddit_Munchen`. The feature set we use for training is restricted to the below
    42 features.
    

| Feature                                | dType   |        Example |
|:---------------------------------------|:--------|---------------:|
| golds                                  | float64 |    1           |
| comment_char_count                     | int64   |  134           |
| comment_text_polarity                  | float64 |    -0.123      |
| network_comment_thread_max_depth       | int64   |    2           |
| network_user_total_comment_count       | int64   |    5           |
| news_subreddit_comment_karma           | int64   |    7           |
| politics_subreddit_post_karma          | int64   |    31231       |
| left_subreddit_comment_count           | int64   |    2981        |
| news_subreddit_comment_count           | int64   |    7           |
| comment_has_user_ref                   | bool    |    True        |
| user_email_verified                    | object  |    1           |
| user_total_comment_count               | int64   |  157           |
| comment_spelling_error_count           | int64   |    1           |
| right_subreddit_comment_count          | int64   |    1           |
| network_user_thread_comment_count      | int64   |    1           |
| user_account_age_seconds               | float64 |    5.96577e+06 |
| center_subreddit_comment_karma         | int64   |    210          |
| post_comment_timedelta_seconds         | float64 | 7730           |
| right_subreddit_post_karma             | int64   |    -2120       |
| comment_url_refer_count                | int64   |    0           |
| center_subreddit_comment_count         | int64   |    12           |
| comment_has_citation                   | bool    |    True        |
| center_subreddit_post_karma            | int64   |    0           |
| politics_subreddit_comment_count       | int64   |   50           |
| politics_subreddit_post_count          | int64   |    0           |
| user_total_post_karma                  | int64   |   15           |
| politics_subreddit_comment_karma       | int64   |   69           |
| left_subreddit_comment_karma           | int64   |    0           |
| user_total_post_count                  | int64   |    4           |
| comment_text_profanity                 | int64   |    7           |
| network_comment_thread_size            | int64   |    3           |
| user_total_comment_karma               | int64   |  753           |
| network_comment_thread_top_level_count | int64   |    2           |
| network_user_top_level_comment_count   | int64   |    2           |
| left_subreddit_post_count              | int64   |    3123        |
| news_subreddit_post_count              | int64   |    0           |
| right_subreddit_comment_karma          | int64   |    0           |
| left_subreddit_post_karma              | int64   |    0           |
| comment_text_subjectivity              | float64 |    0           |
| right_subreddit_post_count             | int64   |    0           |
| center_subreddit_post_count            | int64   |    9012        |
| news_subreddit_post_karma              | int64   |    18000       |

### Setting up the environment

1.  This codebase was developed on Python 3.7.3, there is a `requirements.txt` containing all dependencies. To setup the 
    environment, first install and setup a virtual environment, and then install the dependencies.

    ```bash
    python3 -m pip install --user virtualenv
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```

2.  One of the dependencies `textblob==0.15.3` requires the appropriate NLTK corpora to function. Run the below command
    to fetch them 

    ```bash
    python -m textblob.download_corpora
    ```

