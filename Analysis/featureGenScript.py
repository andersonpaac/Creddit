"""
    The below code takes
    1. The user_frame generated by `UserExtractor`
    2. The comment_frame generated by `CommentExtractor`

    It then generates all the features for the dataset and outputs it to a pickle. The size of this dataset is less
    than equal to the number of top level comments.
"""
from Analysis.FeatureBuilder import *
from Analysis import Common
from tqdm import tqdm
tqdm.pandas()

post_id = 'b4agza'
reddit = get_reddit_instance()
post_creation_time = pd.to_datetime(reddit.submission(post_id).created_utc, unit='s')
user_frame = pd.read_pickle(f'{post_id}_user_frame.pkl')
comment_frame = pd.read_pickle(f'{post_id}_comment_frame.pkl')
top_level_frame = Common.get_replies_to(comment_frame, post_id)
user_frame.reset_index(drop=True, inplace=True)
top_level_frame.reset_index(drop=True, inplace=True)


def check_merged_succesfully(pre_merge, post_merge):
    """
    Sanity checks to ensure the merge hasn't lead to duplicate columns or new rows

    """
    if len(pre_merge) != len(post_merge):
        print(f"Error while merging frames")
        exit(-1)
    if len(set(post_merge.columns)) != len(post_merge.columns):  # Sanity check for duplicate columns
        print(f"Error while merging, one of the columns is duplicate")
        exit(-1)


#  user_feature_frame  has UserExtractorFeatures (in user_frame) and UserFeatureBuilder
# it's length is the size of the user_frame
userFeatureBuilder = UserFeatureBuilder(post_creation_time)
user_features = Df(list(user_frame.progress_apply(userFeatureBuilder.get_features_for_row, axis=1)))
user_feature_frame = user_frame.merge(user_features, on='author')
check_merged_succesfully(user_frame, user_feature_frame)

# post_feature_frame has CommentExtractor features (in comment_frame) and PostFeatureBuilder
# it's length is the size of the top_level_frame
postFeatureBuilder = PostFeatureBuilder(comment_frame)
post_features = Df(list(top_level_frame.progress_apply(postFeatureBuilder.get_features_for_row, axis=1)))
post_feature_frame = top_level_frame.merge(post_features, on='comment_id')
check_merged_succesfully(top_level_frame, post_feature_frame)

missing_users = set(post_feature_frame.author.unique()).difference(set(user_feature_frame.author.unique()))
print(f"Couldn't find {len(missing_users)} users in user_frame. Their accounts might have been deleted or banned.")
post_feature_frame = post_feature_frame[~post_feature_frame.author.isin(list(missing_users))]


def user_patch(author: str) -> Dict:
    return user_feature_frame[user_feature_frame.author == author].iloc[0].to_dict()

# user_post_feature_frame has all the features in user_feature_frame and post_feature_frame
user_post_features = Df(list(post_feature_frame.author.progress_apply(user_patch)))
user_post_feature_frame = post_feature_frame.merge(user_post_features)
user_post_feature_frame.drop_duplicates(subset='comment_id', inplace=True)
user_post_feature_frame.reset_index(drop=True, inplace=True)
check_merged_succesfully(user_post_features, user_post_feature_frame)

user_post_feature_frame.to_pickle(f'{post_id}_dataset.pkl')