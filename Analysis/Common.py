import pandas as pd
from pandas import DataFrame as Df


class Constants:
    SCORE_HIDDEN = -0.9
    EXCLUSION_WLIST = 'Analysis/known_spellings.txt'
    CONFIG_FNAME = 'reddit_api_creds.json'


def get_replies_to(comments_frame: Df, parent_id: str):
    return comments_frame[comments_frame.parent_id.str.contains(parent_id)]


def get_top_level_comments(comments_frame: Df):
    """
    Returns the top level comments to a post from the comment_frame
    """
    post_id = comments_frame.iloc[0].post_id
    return get_replies_to(comments_frame, post_id)


def get_hidden_scores(comments_frame: Df):
    """
    This method will filter the comments who's scores have been hidden.
    Some comments have their scores hidden by reddit, see
    https://www.reddit.com/r/modnews/comments/1dd0xw/moderators_new_subreddit_feature_comment_scores/
    """
    return comments_frame[comments_frame.score == Constants.SCORE_HIDDEN]


def get_unique_post_subreddits(user_frame: Df):
    """
    This will list all unique subreddits that any user has posted on.
    """
    all_subreddits = set()
    for _, row in user_frame.iterrows():
        all_subreddits = all_subreddits.union(set(row.subreddit_post_count.keys()))
    return list(all_subreddits)


def get_unique_comment_subreddits(user_frame: Df):
    """
    This will list all unique subreddits that any user has commented on.
    """
    all_subreddits = set()
    for _, row in user_frame.iterrows():
        all_subreddits = all_subreddits.union(set(row.subreddit_comment_count.keys()))
    return list(all_subreddits)


def load_comment_frame(post_id: str):
    return pd.read_pickle(f'data/comment_frame/{post_id}_comment_frame.pkl')


def load_user_frame(post_id: str):
    return pd.read_pickle(f'data/user_frame/{post_id}_user_frame.pkl')


def load_dataset_for_post(post_id):
    return pd.read_pickle(f'data/dataset/{post_id}_dataset.pkl')


def get_flattened_thread_under_parent_id(comment_frame: Df, parent_id: str) -> Df:
    """
    This will return all the comments that are children and grandchildren of parent_id.
    Example:
        For the below comment structure
            ebjkl
                -exalz
                    -lxasq
                -pladx
                    -pidas
            plloq
                -qvnjs
                    -uxyats

        get_flattened_thread_under_parent_id(ebjkl) will return Df([exalz, lxasq, pladx, pidas])

    :param comment_frame: Dataframe containing all comments
    :param parent_id:
    :return:
    """
    to_prune = list(get_replies_to(comment_frame, parent_id).comment_id)
    children_comment_ids = []
    while len(to_prune) > 0:
        new_parent_id = to_prune.pop()
        children_comment_ids.append(new_parent_id)
        to_prune += list(get_replies_to(comment_frame, new_parent_id).comment_id)
    return comment_frame[comment_frame.comment_id.isin(children_comment_ids)]