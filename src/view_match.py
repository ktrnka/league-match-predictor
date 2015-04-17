from __future__ import unicode_literals
import ConfigParser
import logging
import pprint
import sys
import argparse
import requests
import urlparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config file")
    parser.add_argument("match_id", help="ID of the match to use")
    return parser.parse_args()

class RiotService(object):
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.params = {"api_key": self.api_key}

    def get_match(self, match_id):
        response = requests.get(urlparse.urljoin(self.base_url, "v2.2/match/{}".format(match_id)),
                            params=self.params)
        response.raise_for_status()
        data = response.json()
        return data


def main():
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)

    config = ConfigParser.RawConfigParser()
    config.read([args.config])

    connection = RiotService(config.get("riot", "base"), config.get("riot", "api_key"))
    data = connection.get_match(args.match_id)

    pprint.pprint(data)



if __name__ == "__main__":
    sys.exit(main())