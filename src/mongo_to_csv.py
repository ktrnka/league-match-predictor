from __future__ import unicode_literals
import ConfigParser
import logging
import sys
import argparse
import io
from src.riot_api import RiotService

from src.riot_api_cache import ApiCache
from src.riot_data import Champion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Verbose logging")
    parser.add_argument("config", help="Config file")
    parser.add_argument("output_csv", help="Output file to use for machine learning")
    return parser.parse_args()


def champion_set_to_indicators(champion_ids):
    return [int(i in champion_ids) for i in Champion.known_ids]


def make_champion_indicator_names(riot_connection):
    return [riot_connection.get_champion_info(champion_id)["name"] for champion_id in Champion.known_ids]


def main():
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.captureWarnings(True)
    logger = logging.getLogger(__name__)

    # reduce connection spam
    logging.getLogger("requests.packages.urllib3.connectionpool").setLevel(logging.WARNING)

    config = ConfigParser.RawConfigParser()
    config.read([args.config])

    riot_cache = ApiCache(config)
    riot_connection = RiotService.from_config(config)

    champion_labels = make_champion_indicator_names(riot_connection)
    blue_labels = ["Blue_{}".format(i) for i in champion_labels]
    red_labels = ["Red_{}".format(i) for i in champion_labels]
    columns = ["IsSoloQueue", "Blue_Tier", "Red_Tier"] + blue_labels + red_labels + ["IsBlueWinner"]

    with io.open(args.output_csv, "w") as csv_out:
        csv_out.write(",".join(columns) + "\n")

        for match in riot_cache.get_matches():
            winner = match.get_winning_team_id()

            picks = match.get_picks()
            teams = sorted(picks.keys())
            tiers = match.get_team_tiers_numeric()

            blue_indicators = champion_set_to_indicators(picks[teams[0]])
            red_indicators = champion_set_to_indicators(picks[teams[1]])
            is_blue_winner = int(winner == teams[0])

            row = [int("SOLO" in match.queue_type)] + [tiers[t] for t in teams] + blue_indicators + red_indicators + [is_blue_winner]
            csv_out.write(",".join(str(x) for x in row) + "\n")


if __name__ == "__main__":
    sys.exit(main())