from __future__ import unicode_literals
import ConfigParser
import argparse
import logging
import math
import sys

import collections
from src.riot_api import RiotService
from src.riot_data import Participant


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config file")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.WARNING)

    config = ConfigParser.RawConfigParser()
    config.read([args.config])

    connection = RiotService.from_config(config)

    games, refresh_interval = connection.get_featured_matches()

    z_score_target = 1.6

    for game in games:
        print "\n"

        for player in game["participants"]:
            # print "{} team: {}".format(connection.get_team_name(player["teamId"]), player["summonerName"])
            player = Participant(player)
            champion_name = connection.get_champion_info(player.champion_id)["name"]

            try:
                # get summoner id
                summoner = connection.get_summoner_by_name(player.name)
                summoner_data = connection.get_summoner_ranked_stats(summoner.id)

                # win rates
                overall = collections.Counter()
                this_champ = collections.Counter()
                for champion in summoner_data["champions"]:
                    overall.update(champion["stats"])
                    if champion["id"] == player.champion_id:
                        this_champ.update(champion["stats"])

                try:
                    win_rate = this_champ["totalSessionsWon"] / float(this_champ["totalSessionsPlayed"])
                    conf_interval = z_score_target * math.sqrt(win_rate * (1. - win_rate) / (this_champ["totalSessionsPlayed"] + z_score_target ** 2))
                except ZeroDivisionError:
                    win_rate = -1
                    conf_interval = -1

                overall_win_rate = overall["totalSessionsWon"] / float(overall["totalSessionsPlayed"])
                overall_conf_interval = z_score_target * math.sqrt(overall_win_rate * (1. - overall_win_rate) / (overall["totalSessionsPlayed"] + z_score_target ** 2))

                print connection.get_team_name(player.team_id), player.name, champion_name, "{:.1f}% win rate +/- {:.1f}% ({:.1f}% +/- {:.1f}% overall)".format(100 * win_rate, 100 * conf_interval, 100 * overall_win_rate, 100 * overall_conf_interval)
            except requests.exceptions.HTTPError:
                print connection.get_team_name(player.team_id), player.name, champion_name


if __name__ == "__main__":
    sys.exit(main())