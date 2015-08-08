from __future__ import unicode_literals
import logging
import sys
import argparse
import time
import urlparse
import requests


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    args = parse_args()

if __name__ == "__main__":
    sys.exit(main())


class RiotService(object):
    def __init__(self, base_url, static_base_url, api_key):
        self.base_url = base_url
        self.static_base_url = static_base_url
        self.api_key = api_key
        self.params = {"api_key": self.api_key}
        self.champions = None

        self.most_recent_request = None
        self.delay_seconds = 1.2
        self.logger = logging.getLogger("RiotService")

    def request(self, endpoint):
        self.throttle()
        response = requests.get(urlparse.urljoin(self.base_url, endpoint), params=self.params)
        response.raise_for_status()
        data = response.json()
        return data

    def get_match(self, match_id):
        return self.request("v2.2/match/{}".format(match_id))

    def get_champion_info(self, champion_id):
        if not self.champions:
            self.champions = self.get_champions()

        return self.champions[champion_id]

    def request_static(self, endpoint):
        self.throttle()
        response = requests.get(urlparse.urljoin(self.static_base_url, endpoint), params=self.params)
        response.raise_for_status()
        data = response.json()
        return data

    def get_champions(self):
        data = self.request_static("v1.2/champion")
        data = data["data"]

        return {value["id"]: value for value in data.itervalues()}

    def throttle(self):
        if self.most_recent_request and time.clock() - self.most_recent_request < self.delay_seconds:
            self.logger.debug("Sleeping for %.1f seconds", self.delay_seconds)
            time.sleep(self.delay_seconds)
        self.most_recent_request = time.clock()

    def get_team_name(self, team_id):
        if team_id == 100:
            return "Blue"
        elif team_id == 200:
            return "Red"
        else:
            return "Unknown, " + team_id

    def get_summoner_ranked_stats(self, summoner_id):
        return self.request("v1.3/stats/by-summoner/{summonerId}/ranked".format(summonerId=summoner_id))