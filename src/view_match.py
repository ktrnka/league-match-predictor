from __future__ import unicode_literals
import ConfigParser
import collections
import logging
import pprint
import sys
import argparse
import math
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


def main():
    args = parse_args()

    logging.basicConfig(level=logging.WARNING)

    config = ConfigParser.RawConfigParser()
    config.read([args.config])

    connection = RiotService(config.get("riot", "base"), config.get("riot", "static_base"), config.get("riot", "api_key"))
    data = connection.get_match(args.match_id)

    teams = data["teams"]
    for team in teams:
        team["players"] = []
    teams = {team["teamId"]: team for team in teams}

    for participant, identity in zip(data["participants"], data["participantIdentities"]):
        assert identity["participantId"] == participant["participantId"]
        teams[participant["teamId"]]["players"].append((participant, identity))

    z_score_target = 1.6

    for team in teams.itervalues():
        print "{} team".format(connection.get_team_name(team["teamId"])), "winner" if team["winner"] else "loser"

        for participant, identity in team["players"]:
            summoner_data = connection.get_summoner_ranked_stats(identity["player"]["summonerId"])

            overall = collections.Counter()
            this_champ = collections.Counter()
            for champion in summoner_data["champions"]:
                overall.update(champion["stats"])
                if champion["id"] == participant["championId"]:
                    this_champ.update(champion["stats"])

            win_rate = this_champ["totalSessionsWon"] / float(this_champ["totalSessionsPlayed"])
            conf_interval = z_score_target * math.sqrt(win_rate * (1. - win_rate) / (this_champ["totalSessionsPlayed"] + z_score_target ** 2))

            overall_win_rate = overall["totalSessionsWon"] / float(overall["totalSessionsPlayed"])
            overall_conf_interval = z_score_target * math.sqrt(overall_win_rate * (1. - overall_win_rate) / (overall["totalSessionsPlayed"] + z_score_target ** 2))



            champion_name = connection.get_champion_info(participant["championId"])["name"]

            print "\t", identity["player"]["summonerName"], champion_name, "{:.1f}% win rate +/- {:.1f}% ({:.1f}% +/- {:.1f}% overall)".format(100 * win_rate, 100 * conf_interval, 100 * overall_win_rate, 100 * overall_conf_interval)



if __name__ == "__main__":
    sys.exit(main())