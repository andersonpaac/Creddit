import json
from typing import List, Dict
import numpy as np
import math
from pandas import DataFrame as Df
import pandas as pd
import requests
from datetime import datetime


class PushShift:
    PATH = "https://api.pushshift.io/reddit"

    @classmethod
    def get_comment_ids_for_submission_id(cls, submission_id: str) -> List[str]:
        """
        For a given submission, this will get all the comment_ids
        :param submission_id:
        :return:
        """
        url = "/".join([cls.PATH, "submission/comment_ids", submission_id])
        return cls.get_request(url)['data']

    @classmethod
    def get_comments_for_comment_ids(cls, comment_ids: List[str]) -> List[Dict]:
        """
        For a list of comment ids this will get all the comments. It will handle however big the list is
        :return:
        """
        num_splits = math.ceil(len(comment_ids) / 99)
        comment_ids_sets = np.array_split(comment_ids, num_splits)
        comments = []
        for comment_ids_set in comment_ids_sets:
            comment_ids = ",".join(comment_ids_set)
            url = f"{cls.PATH}/comment/search?ids={comment_ids}"
            try:
                comments += cls.get_request(url)['data']
            except KeyError as e:
                print(f"SKipped {comment_ids}")
                continue
        return comments

    @classmethod
    def get_comments_for_submission_id(cls, submission_id: str) -> List[Dict]:
        comment_ids = cls.get_comment_ids_for_submission_id(submission_id)
        comments = cls.get_comments_for_comment_ids(comment_ids)
        missed = len(comment_ids) - len(comments)
        print(f"Missed {missed} comments out of {len(comment_ids)}.")
        return comments

    @classmethod
    def get_request(cls, url) -> dict:
        """
        Helper to make network requests
        """
        then = datetime.now()
        resp = requests.get(url=url)
        if resp.status_code != 200:
            raise ConnectionError(f"Unable to fulfill {url}, got {resp.content} with {resp.status_code}")
        now = datetime.now()
        resp_time = now - then
        print(f"Took {resp_time.total_seconds()} seconds to hit {url}")
        return json.loads(resp.content)

