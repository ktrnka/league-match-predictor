from __future__ import unicode_literals
import ConfigParser
import logging
import pprint
import sys
import argparse
import requests
import urlparse
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config file")
    parser.add_argument("match_id", help="ID of the match to use")
    return parser.parse_args()


class RiotService(object):
    def __init__(self, base_url, static_base_url, api_key):
        self.base_url = base_url
        self.static_base_url = static_base_url
        self.api_key = api_key
        self.params = {"api_key": self.api_key}
        self.champions = None

        self.most_recent_request = None
        self.delay_seconds = 1.
        self.logger = logging.getLogger("RiotService")

    def get_match(self, match_id):
        self.throttle()
        response = requests.get(urlparse.urljoin(self.base_url, "v2.2/match/{}".format(match_id)), params=self.params)
        response.raise_for_status()
        data = response.json()
        return data

    def get_champion_info(self, champion_id):
        if not self.champions:
            self.champions = self.get_champions()

        return self.champions[champion_id]

    def get_champions(self):
        self.throttle()
        response = requests.get(urlparse.urljoin(self.static_base_url, "v1.2/champion"), params=self.params)
        response.raise_for_status()
        data = response.json()
        data = data["data"]

        return {value["id"]: value for value in data.itervalues()}

    def throttle(self):
        if self.most_recent_request and time.clock() - self.most_recent_request < self.delay_seconds:
            self.logger.debug("Sleeping for %f seconds", self.delay_seconds)
            time.sleep(self.delay_seconds)
        self.most_recent_request = time.clock()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)

    config = ConfigParser.RawConfigParser()
    config.read([args.config])

    connection = RiotService(config.get("riot", "base"), config.get("riot", "static_base"), config.get("riot", "api_key"))
    data = connection.get_match(args.match_id)

    for participant, identity in zip(data["participants"], data["participantIdentities"]):
        assert identity["participantId"] == participant["participantId"]
        print identity["player"]["summonerName"], participant["teamId"], connection.get_champion_info(participant["championId"])["name"]

    # pprint.pprint(data)


if __name__ == "__main__":
    sys.exit(main())