from __future__ import unicode_literals
import ConfigParser
import argparse
import sys
import datetime

from riot_api import *
from riot_api_cache import ApiCache
from riot_data import Participant, Match
import requests.exceptions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Verbose logging")
    parser.add_argument("config", help="Config file")
    return parser.parse_args()


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

    riot_connection = RiotService.from_config(config)
    riot_cache = ApiCache(config)

    try:
        success = collections.Counter()
        for player_id in riot_cache.get_player_ids():

            try:
                player_stats = riot_connection.get_summoner_ranked_stats(player_id)
                riot_cache.update_player_stats(player_id, player_stats)
                success["update ranked success"] += 1
            except SummonerNotFoundError:
                success["update ranked failure"] += 1

            try:
                player_stats = riot_connection.get_summoner_summary_stats(player_id)
                riot_cache.update_player_summary_stats(player_id, player_stats)
                success["update summary success"] += 1
            except SummonerNotFoundError:
                success["update summary failure"] += 1

        print "Player update outcomes: {}".format(success.most_common())

    except requests.exceptions.HTTPError as exc:
        logger.exception("Unhandled HTTPError, aborting")

    riot_cache.log_summary()
    logger.info("Database status: %s", riot_cache.summarize())


if __name__ == "__main__":
    sys.exit(main())