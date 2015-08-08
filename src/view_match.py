from __future__ import unicode_literals
import ConfigParser
import collections
import logging
import sys
import argparse
import math

from src.riot_api import RiotService


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config file")
    parser.add_argument("match_id", help="ID of the match to use")
    return parser.parse_args()


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