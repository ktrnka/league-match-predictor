from __future__ import unicode_literals
import ConfigParser
import logging
import sys
import argparse
import io
import time
import collections
from riot_api import RiotService

from riot_api_cache import ApiCache
from riot_data import Champion, Participant, PlayerStats, ChampionStats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Verbose logging")
    parser.add_argument("config", help="Config file")
    parser.add_argument("output_csv", help="Output file to use for machine learning")
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

    riot_cache = ApiCache(config)
    riot_connection = RiotService.from_config(config)

    columns = ["TotalPlayed", "TotalWinRate", "PredictedPlayed", "PredictedWinRate"]

    # quickly load all player stats into RAM so we can join more quickly
    previous_time = time.time()
    riot_cache.preload_player_stats()
    riot_cache.precompute_champion_damage()
    logger.info("Preloading player stats took %.1f sec", time.time() - previous_time)

    previous_time = time.time()
    with io.open(args.output_csv, "w") as csv_out:
        csv_out.write(",".join(columns) + "\n")

        for player_stats in riot_cache.local_stats_cache.itervalues():
            assert isinstance(player_stats, PlayerStats)

            # sum up the total stats
            total_stats = collections.Counter()
            for champion_stats in player_stats.champion_stats.itervalues():
                total_stats.update(champion_stats)

            total_stats = ChampionStats(total_stats)

            # each individual becomes one row
            for champion_id, champion_stats in player_stats.champion_stats.iteritems():
                champion_stats = ChampionStats(champion_stats)

                # remove the predicted fields
                total_played = total_stats.played - champion_stats.played
                total_won = total_stats.won - champion_stats.won

                row = [total_played, total_won / float(total_played), champion_stats.played, champion_stats.won / float(champion_stats.played)]
                csv_out.write(",".join(str(x) for x in row) + "\n")


    logger.info("Pulling and converting data took %.1f sec", time.time() - previous_time)



if __name__ == "__main__":
    sys.exit(main())