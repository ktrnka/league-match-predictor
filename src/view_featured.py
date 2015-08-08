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
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.WARNING)

    config = ConfigParser.RawConfigParser()
    config.read([args.config])

    connection = RiotService.from_config(config)

    games, refresh_interval = connection.get_featured_matches()
    data = connection.get_match(args.match_id)

    z_score_target = 1.6

    for game in games:
        for player in game["participants"]:
            # print "{} team: {}".format(connection.get_team_name(player["teamId"]), player["summonerName"])

            # get summoner id
            summoner_id = connection.get_summoner_id(player["summonerName"])
            summoner_data = connection.get_summoner_ranked_stats(summoner_id)

            # win rates
            overall = collections.Counter()
            this_champ = collections.Counter()
            for champion in summoner_data["champions"]:
                overall.update(champion["stats"])
                if champion["id"] == player["championId"]:
                    this_champ.update(champion["stats"])

            win_rate = this_champ["totalSessionsWon"] / float(this_champ["totalSessionsPlayed"])
            conf_interval = z_score_target * math.sqrt(win_rate * (1. - win_rate) / (this_champ["totalSessionsPlayed"] + z_score_target ** 2))

            overall_win_rate = overall["totalSessionsWon"] / float(overall["totalSessionsPlayed"])
            overall_conf_interval = z_score_target * math.sqrt(overall_win_rate * (1. - overall_win_rate) / (overall["totalSessionsPlayed"] + z_score_target ** 2))



            champion_name = connection.get_champion_info(player["championId"])["name"]

            print connection.get_team_name(player["teamId"]), player["summonerName"], champion_name, "{:.1f}% win rate +/- {:.1f}% ({:.1f}% +/- {:.1f}% overall)".format(100 * win_rate, 100 * conf_interval, 100 * overall_win_rate, 100 * overall_conf_interval)


if __name__ == "__main__":
    sys.exit(main())