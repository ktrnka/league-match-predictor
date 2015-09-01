from __future__ import unicode_literals
import ConfigParser
import logging
import sys
import argparse
import time

import riot_api
import riot_api_cache


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

    riot_cache = riot_api_cache.ApiCache(config)
    riot_connection = riot_api.RiotService.from_config(config)

    # quickly load all player stats into RAM so we can join more quickly
    agg_stats, agg_champion_stats = riot_cache.aggregate_champion_stats()

    print "Aggregated stats cover {:,} games".format(agg_stats.played)




if __name__ == "__main__":
    sys.exit(main())