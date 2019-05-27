from praw.reddit import models
from prawcore.exceptions import *
from collections import Counter
from praw.models import Redditor
from Ingest.Reddit import *
from Ingest.pushShift import *


class Extractor:
    def __init__(self, no_caching: bool = False, checkpoint_interval: int = 100):
        """
        :param checkpoint_interval: Saves the data every interval count rows
        """
        self.cp_fname = "checkpoint_extractor_frame"    # Overriden by child class
        self.reddit = get_reddit_instance()
        self.last_saved_at_index = 0
        self.save_cp_interval = checkpoint_interval
        self.no_caching = no_caching

    def save_checkpoint_if_needed(self, cursor_index: int, rows: List[Dict]):
        if self.no_caching:
            return
        if cursor_index - self.last_saved_at_index >= self.save_cp_interval:
            Df(rows).to_pickle(f'{self.cp_fname}.pkl')
            self.last_saved_at_index = cursor_index


class CommentExtractor(Extractor):
    """
        The below code is used to extract all the comments along with metadata for a given post_id

        Author
            If author is empty (""):    comment is removed by mods (Body should be visible)
            If author is math.nan:      Possibly - Author deleted own comment

        Score
            -0.9 if comment.score_hidden

        If comment was removed by mods, you can still see the comment body but the author shows up as  an empty string
        If comment was removed by the user himself, the whole row is empty
    """
    def __init__(self, **kwargs):
        Extractor.__init__(self, **kwargs)
        self.cp_fname = "checkpoint_comment_frame"

    def extract_comments_for_submission_id(self, submission_ids: List[str]):
        """
        Given a submission id, this will extract all comments associated to that submission id (including potential deleted
        ones and removed ones) and also write to disk.
        :param submission_ids:
        :return:
        """
        submissions = [self.reddit.submission(id=submission_id) for submission_id in submission_ids]
        for submission in submissions:
            submission.comment_sort = 'controversial'
            pshift_comments = PushShift.get_comments_for_submission_id(submission.id)
            self.extract_comments(pshift_comments, submission).to_pickle(f"{submission.id}_comment_frame.pkl")

    def extract_comments(self, pshift_comments: List[Dict], praw_submission: models.Submission):
        rows = []
        for index, pshift_comment in enumerate(pshift_comments):
            progress = float(index / len(pshift_comments))
            progress = progress * 100
            print(progress)
            rows.append(self.merge_with_praw_comment(pshift_comment))
            self.save_checkpoint_if_needed(index, rows)
        frame = Df(rows)
        frame['is_submitter'] = frame['author'] == praw_submission.author
        frame = frame[~frame.body.isnull()]
        frame.reset_index(drop=True, inplace=True)
        return frame

    def merge_with_praw_comment(self, phsift_comment: Dict):
        pshift_body = phsift_comment['body']
        if self.is_comment_removed_by_mods(pshift_body) or self.is_comment_removed_by_mods(pshift_body):  #  If even pshift couldn't get the deleted version we get rid of it
            return {}
        praw_comment = self.reddit.comment(phsift_comment['id'])
        return {
            "body": phsift_comment['body'],
            "edited": praw_comment.edited is not False,  # if the comment is edited, the utc is given. We map it to a bool
            "author": phsift_comment['author'],
            "comment_id": phsift_comment['id'],
            "post_id": self.get_post_id(phsift_comment['link_id']),
            "golds": praw_comment.gilded,
            "score": praw_comment.score if not praw_comment.score_hidden else -0.9,
            "parent_id": praw_comment.parent_id,
            "comment_removed_by_mods": self.is_comment_removed_by_mods(praw_comment.body),
            "comment_deleted": self.is_comment_deleted(praw_comment.body),
            "comment_created_utc": pd.to_datetime(praw_comment.created_utc, unit='s')
        }

    @classmethod
    def get_post_id(cls, full_name):
        if "_" in full_name:
            return full_name.split("_")[1]
        return full_name

    @classmethod
    def is_comment_removed_by_mods(cls, body):
        return "removed" in body.lower()

    @classmethod
    def is_comment_deleted(cls, body):
        return "deleted" in body.lower()


class UserExtractor(Extractor):

    def __init_(self, **kwargs):
        Extractor.__init__(self, **kwargs)
        self.cp_fname = "checkpoint_user_frame"

    @classmethod
    def get_all_user_predictors(cls, redditor: Redditor) -> Dict:
        return {
            "author": redditor.name,
            "user_total_comment_karma": redditor.comment_karma,
            "user_total_post_karma": redditor.link_karma,
            "user_email_verified": redditor.has_verified_email,
            "user_account_creation_utc": pd.to_datetime(redditor.created_utc, unit='s'),
            **cls.get_subreddit_post_score_and_count(redditor),
            **cls.get_subreddit_comment_score_and_count(redditor)
        }

    @classmethod
    def get_subreddit_post_score_and_count(cls, redditor: Redditor) -> Dict:
        """
        Returns the link/post karma grouped by subreddit and also number of posts in each subreddit
        # @toDo: 1. Should we handlle only the posts before the time?
        :param redditor:
        :return:
        """
        subreddit_post_karma = Counter()
        subreddit_post_count = Counter()
        total_posts = 0
        for author_submission in redditor.submissions.top(limit=None):
            subreddit_post_karma[author_submission.subreddit.display_name] += author_submission.score
            subreddit_post_count[author_submission.subreddit.display_name] += 1
            total_posts += 1
        return {"subreddit_post_karma": dict(subreddit_post_karma),
                "subreddit_post_count": dict(subreddit_post_count),
                "user_total_post_count": total_posts
                }

    @classmethod
    def get_subreddit_comment_score_and_count(cls, redditor: Redditor) -> Dict:
        """
        Returns the comment karma grouped by subreddit and also number of comments in each subreddit
        # @toDo: 1. Should we handlle only the comments before the time?
        :param redditor:
        :return:
        """
        subreddit_comment_karma = Counter()
        subreddit_comment_count = Counter()
        total_comments = 0
        for author_comment in redditor.comments.hot(limit=None):
            subreddit_comment_karma[author_comment.subreddit.display_name] += 0 if author_comment.score_hidden else author_comment.score
            subreddit_comment_count[author_comment.subreddit.display_name] += 1
            total_comments += 1
        return {
            "subreddit_comment_karma": dict(subreddit_comment_karma),
            "subreddit_comment_count": dict(subreddit_comment_count),
            "user_total_comment_count": total_comments
        }

    def get_frame_for_authors(self, authors: List[str]) -> Df:
        """
        Method to get a dataframe of user predictors for a list of authors
        :param authors:
        :return:
        """
        rows, reddit = [], get_reddit_instance()
        for index, author in enumerate(authors):
            progress = float(index / len(authors))
            progress = progress * 100
            print(progress)
            if author == "":
                continue
            if type(author) == float and math.isnan(author):
                continue
            try:
                rows.append(self.get_all_user_predictors(reddit.redditor(name=author)))
            except Exception as e:
                print(f"Encountered error: {e} at index: {index}. Saving checkpoint....")
                print(f"Encountered error at index {index}")
                out_fname = f'checkpoint_users_exception.pkl'
                Df(rows).to_pickle(out_fname)
            except NotFound as e:
                print(f"Seems like {author} has deleted their account: {e}")
            self.save_checkpoint_if_needed(index, rows)
        frame = Df(rows)
        frame.reset_index(drop=True, inplace=True)
        return frame

    @classmethod
    def save_user_comments(cls, input_fname: str, start_index: int = 0):
        """
        Main method to extract all user predictors from the author column of the frame in input_fname pickle
        :param input_fname:
        :param start_index: To resume
        :return:
        """
        comment_frame = pd.read_pickle(input_fname + '.pkl')
        authors = comment_frame.author.unique()[start_index:]
        user_frame = cls.get_frame_for_authors(authors)
        output_fname = f'{input_fname}_authors_{start_index}_'
        out_fname = f'{output_fname}_all.pkl'
        user_frame.to_pickle(out_fname)

