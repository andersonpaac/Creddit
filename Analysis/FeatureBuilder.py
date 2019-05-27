import json
import re
import pandas as pd
from datetime import datetime
import profanity_check
from textblob import TextBlob
from typing import Dict, List
from collections import defaultdict
from pandas import DataFrame as Df
from spellchecker import SpellChecker
from Ingest.Reddit import get_reddit_instance
from Analysis import Common


class CaseInsensitiveDefaultDict(defaultdict):
    """
        This is a case insensitive default dictionary that can help us match keys that are case insenti
    """

    def __setitem__(self, key, value):
        super(CaseInsensitiveDefaultDict, self).__setitem__(key.lower(), value)

    def __getitem__(self, key):
        return super(CaseInsensitiveDefaultDict, self).__getitem__(key.lower())

    def __init__(self, *args, **kwargs):
        super(CaseInsensitiveDefaultDict, self).__init__(*args, **kwargs)
        self._convert_keys()

    def _convert_keys(self):
        for k in list(self.keys()):
            v = super(CaseInsensitiveDefaultDict, self).pop(k)
            self.__setitem__(k, v)


class UserFeatureBuilder:

    def __init__(self, post_creation_time: datetime):
        self.post_creation_time = post_creation_time

    def get_features_for_row(self, user_frame_row):
        user_creation_post_creation_timedelta = self.post_creation_time - user_frame_row.user_account_creation_utc
        subreddit_features = SubredditFeatureBuilder.get_features(
                subreddit_post_count=CaseInsensitiveDefaultDict(int, user_frame_row.subreddit_post_count),
                subreddit_post_karma=CaseInsensitiveDefaultDict(int, user_frame_row.subreddit_post_karma),
                subreddit_comment_count=CaseInsensitiveDefaultDict(int, user_frame_row.subreddit_comment_count),
                subreddit_comment_karma=CaseInsensitiveDefaultDict(int, user_frame_row.subreddit_comment_karma)
            )
        return {
            **subreddit_features,
            "user_account_age_seconds": user_creation_post_creation_timedelta.total_seconds(),
            "author": user_frame_row.author
        }


class PostFeatureBuilder:
    """
        This will get all the features from
            1. PostUserFeatureBuilder,
            2. CommentNetworkFeatureBuilder
            3. CommentTextFeatureBuilder

        It will also add:
            post_comment_timedelta_seconds:The number of seconds between when the post was made and the comment was made

    """
    def __init__(self, comment_frame: Df):
        """
        This needs the comment frame because it needs access to children of top level comments for analysis
        """
        self.comment_frame = comment_frame
        self.postUserFBuilder = PostUserFeatureBuilder(comment_frame)
        self.commentNetworkFBuilder = CommentNetworkFeatureBuilder(comment_frame)
        self.commentTextFeatureBuilder = CommentTextFeatureBuilder(
            exclusion_wordlist_fname=Common.Constants.EXCLUSION_WLIST
        )
        self.post_id = comment_frame.post_id.iloc[0]
        reddit = get_reddit_instance()
        submission = reddit.submission(id=self.post_id)
        self.post_timestamp = pd.to_datetime(submission.created_utc, unit='s')

    def get_features_for_row(self, top_level_comment_row):
        """
        Pass only top_level_comment_rows to this method.
        """
        time_between_post_comment = top_level_comment_row.comment_created_utc - self.post_timestamp
        return {
            **self.postUserFBuilder.get_features_for_row(top_level_comment_row),
            **self.commentNetworkFBuilder.get_features_for_row(top_level_comment_row),
            **self.commentTextFeatureBuilder.get_features_for_row(top_level_comment_row),
            "comment_id": top_level_comment_row.comment_id,
            "post_comment_timedelta_seconds": time_between_post_comment.total_seconds()
        }


class PostUserFeatureBuilder:
    """
        For a given post and a user, this Builder generates the below features:
            1.  network_user_top_level_comment_count    The number of top level comments by user in the post
            2.  network_user_thread_comment_count       The number of comments by user under the same thread
            3.  network_user_total_comment_count        The number of comments by the user under the entire post
    """
    def __init__(self, comment_frame: Df):
        self.post_id = comment_frame.post_id.iloc[0]
        self.comment_frame = comment_frame

    def get_features_for_row(self, top_level_comment_row):
        return {
            "network_user_top_level_comment_count":
                self.get_top_level_comment_count(top_level_comment_row.author),
            "network_user_thread_comment_count":
                self.get_flattened_level_comment_count_under_thread(
                    top_level_comment_row.author,
                    top_level_comment_row.comment_id
                ),
            "network_user_total_comment_count":
                self.get_total_comment_count_by_author(top_level_comment_row.author)
        }

    def get_top_level_comment_count(self, author: str) -> int:
        """
        Returns the number of top level comments by author.
        :param author: author by whom comments are selected
        :return:
        """
        top_level_comments = Common.get_replies_to(self.comment_frame, self.post_id)
        return len(top_level_comments[top_level_comments.author == author])

    def get_flattened_level_comment_count_under_thread(self, author: str, parent_id: str) -> int:
        """
        This will get the count of comments under the flattened version of the thread of `parent_id` who's author
        is the passed in author.
        :param author: Author
        :param parent_id: comment_id of the thread.
        :return:
        """
        flattened_thread_under_parent = Common.get_flattened_thread_under_parent_id(self.comment_frame, parent_id)
        return len(flattened_thread_under_parent[flattened_thread_under_parent.author == author])

    def get_total_comment_count_by_author(self, author) -> int:
        """
        Returns the number of comments by the author in the entire post.
        """
        return len(self.comment_frame[self.comment_frame.author == author])


class CommentNetworkFeatureBuilder:
    """
        Generates
            1. network_comment_thread_top_level_count   The number of top level comments that are children to the thread
            2. network_comment_thread_max_depth         The height of the tallest tree in the thread
            3. network_comment_thread_size              The total number of children in the thread
    """

    def __init__(self, comment_frame: Df):
        self.comment_frame = comment_frame

    def get_features_for_row(self, top_level_comment_row):
        return {
            "network_comment_thread_top_level_count":
                self.get_top_level_comment_count_under_thread(top_level_comment_row.comment_id),
            "network_comment_thread_max_depth":
                self.get_max_depth_under_thread(
                    top_level_comment_row.comment_id,
                ),
            "network_comment_thread_size":
                self.get_thread_size(top_level_comment_row.comment_id),
        }

    def get_top_level_comment_count_under_thread(self, parent_id: str) -> int:
        """
        Gets the number of top level comments under a given parent_id
        :param parent_id:
        :return:
        """
        return len(Common.get_replies_to(self.comment_frame, parent_id))

    def get_max_depth_under_thread(self, parent_id: str, curr_depth = 0) -> int:
        """
        Returns the maximum depth (longest path) under a parent_id/comment_id.
        :param parent_id: comment_id of the node
        :param curr_depth: Depth of parent, if parent is a top level comment depth is 0 and is default behavior.
        :return: Height of tallest tree
        """
        max_depth = curr_depth
        children_ids = list(Common.get_replies_to(self.comment_frame, parent_id).comment_id)
        for child_comment_id in children_ids:
            max_depth = max(max_depth, self.get_max_depth_under_thread(child_comment_id, curr_depth + 1))
        return max_depth

    def get_thread_size(self, parent_id: str) -> int:
        return len(Common.get_flattened_thread_under_parent_id(self.comment_frame, parent_id))


class CommentTextFeatureBuilder:
    def __init__(self, exclusion_wordlist_fname: str):
        self.spell_check = SpellChecker()
        self.excluded_wordlist = self.load_excluded_words(exclusion_wordlist_fname)

    def get_features_for_row(self, top_level_comment_row):
        body = top_level_comment_row.body
        try:
            return {
                **CommentTextFeatureBuilder.get_sentiment_features(body),
                **self.get_spelling_features(comment_body=body),
                **CommentTextFeatureBuilder.get_url_features(body),
                "comment_text_profanity": profanity_check.predict([body])[0],       # Profanity_check.predict takes an iterable
                "comment_has_citation":  ">" in body,     # Markdown citation
                "comment_has_user_ref": " u/" in body,
                "comment_char_count": len(body)
            }
        except Exception as e:
            print(e, top_level_comment_row.comment_id)

    def get_spelling_features(self, comment_body: str) -> Dict:
        """
        For given comment text, this will return the number of spelling mistakes and a list of words that were
        misspelled.
        :param comment_body:
        :return:
        """
        comment_body = str.lstrip(comment_body)
        comment_body = str.rstrip(comment_body)
        comment_body = CommentTextFeatureBuilder.remove_urls(comment_body)
        comment_body = ''.join(list(map(lambda char: char if char.isalpha() else " ", list(comment_body))))
        uncorrected_words = comment_body.split(' ')
        mistakes_count, mistaken_words = 0, []
        for uncorrected_word in uncorrected_words:
            if len(uncorrected_word) > 0 and not self.is_word_spelled_correctly(uncorrected_word):
                mistakes_count += 1
                mistaken_words.append(uncorrected_word)
        return {
            "comment_spelling_error_count": mistakes_count,
            "meta_comment_spelling_errors": mistaken_words
        }

    @classmethod
    def get_sentiment_features(cls, comment_body: str):
        blob = TextBlob(comment_body)
        return {
            "comment_text_polarity": blob.sentiment.polarity,
            "comment_text_subjectivity": blob.sentiment.subjectivity
        }

    @classmethod
    def get_url_features(cls, comment_body: str) -> Dict:
        """
        For a given comment body, it returns the number of URL's referenced
        :param comment_body:
        :return:
        """
        return {
            "comment_url_refer_count": len(list(filter(lambda word: cls.is_word_a_url(word), comment_body.split(' '))))
        }

    @classmethod
    def load_excluded_words(cls, exclusion_wordlist_fname: str) -> List[str]:
        """
        This will load the words form the excluded wordlist into memory. We need to do this along with whitelisting
        with pyspellchecker because if "mueller" is excluded, "mueller's" will still count as an error according to
        the usage of the pyspellcheck package
        :param exclusion_wordlist_fname:
        :return:
        """
        fd = open(exclusion_wordlist_fname, 'rt')
        excluded_words = fd.read()
        fd.close()
        return excluded_words.split('\n')

    @classmethod
    def is_word_a_url(cls, word: str) -> bool:
        """
        Returns true if there is a url contained in the word
        """
        url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', word)
        return len(url) > 0

    @classmethod
    def remove_urls(cls, text: str) -> str:
        """
        Removes any URLs from text.
        :param word:  text that may contain the URL
        :return: Text not contianing url
        """
        text = text.lower()
        text = text.split(" ")
        words = list(filter(lambda word: not cls.is_word_a_url(word), text))
        text = " ".join(words)
        return text

    def is_excluded(self, word) -> bool:
        for excluded_word in self.excluded_wordlist:
            if excluded_word in word:
                return True
        return False

    def is_word_spelled_correctly(self, word: str) -> bool:
        """
        Returns true if word is spelled correctly
        """
        wrong_words = self.spell_check.unknown([word])
        wrong_words = list(filter(lambda word: not self.is_excluded(word), wrong_words))
        return len(wrong_words) == 0


class SubredditFeatureBuilder:

    @classmethod
    def get_features(cls, **args):
        return {
            **cls.get_partisan_features(**args),
            **cls.get_specific_subreddit_features("politics", **args),   # Origin subreddit
            **cls.get_specific_subreddit_features("news", **args),
        }

    @classmethod
    def get_specific_subreddit_features(cls, subreddit_name, **args):
        origin_subreddit_meta = cls.get_aggregated_features([subreddit_name], **args)
        return {
            f"{subreddit_name}_subreddit_post_count": origin_subreddit_meta[0],
            f"{subreddit_name}_subreddit_post_karma": origin_subreddit_meta[1],
            f"{subreddit_name}_subreddit_comment_count": origin_subreddit_meta[2],
            f"{subreddit_name}_subreddit_comment_karma": origin_subreddit_meta[3]
        }


    @classmethod
    def get_partisan_features(cls, **args):
        left_leaning_meta = cls.get_aggregated_features(cls.get_subreddits_in_category("left_leaning"), **args)
        right_leaning_meta = cls.get_aggregated_features(cls.get_subreddits_in_category("right_leaning"), **args)
        center_leaning_meta = cls.get_aggregated_features(cls.get_subreddits_in_category("centrist"), **args)
        return {
            "left_subreddit_post_count": left_leaning_meta[0],
            "left_subreddit_post_karma": left_leaning_meta[1],

            "right_subreddit_post_count": right_leaning_meta[0],
            "right_subreddit_post_karma": right_leaning_meta[1],

            "center_subreddit_post_count": center_leaning_meta[0],
            "center_subreddit_post_karma": center_leaning_meta[1],

            "left_subreddit_comment_count": left_leaning_meta[2],
            "left_subreddit_comment_karma": left_leaning_meta[3],

            "right_subreddit_comment_count": right_leaning_meta[2],
            "right_subreddit_comment_karma": right_leaning_meta[3],

            "center_subreddit_comment_count": center_leaning_meta[2],
            "center_subreddit_comment_karma": center_leaning_meta[3]
        }

    @classmethod
    def get_aggregated_features(
            cls,
            subreddits_in_category: List[str],
            subreddit_post_count: defaultdict,
            subreddit_post_karma: defaultdict,
            subreddit_comment_count: defaultdict,
            subreddit_comment_karma: defaultdict
        ):
        post_count, post_karma, comment_count, comment_karma = 0, 0, 0, 0
        for subreddit in subreddits_in_category:
            subreddit = subreddit.lower()
            post_count += subreddit_post_count[subreddit]
            post_karma += subreddit_post_karma[subreddit]
            comment_count += subreddit_comment_count[subreddit]
            comment_karma += subreddit_comment_karma[subreddit]
        return post_count, post_karma, comment_count, comment_karma

    @staticmethod
    def get_subreddits_in_category(partisan_leaning) -> List[str]:
        with open('Analysis/partisan_resource.json') as json_data:
            partisan_json = json.load(json_data)
            json_data.close()
        return partisan_json[partisan_leaning]

