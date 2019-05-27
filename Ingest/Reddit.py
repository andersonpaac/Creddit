import json
import praw
from Analysis.Common import Constants


def get_reddit_instance(config_json_fname: str = Constants.CONFIG_FNAME):
    """
    Given path to a file containing the credentials for reddit API's client_id, secret, user agent. This will return
    the praw instance.
    :param config_json_fname:
    :return:
    """
    with open(config_json_fname) as json_data:
        config_creds = json.load(json_data)
        json_data.close()
    reddit = praw.Reddit(client_id=config_creds['client_id'],
                         client_secret=config_creds['client_secret'],
                         user_agent=config_creds['user_agent'])
    return reddit
