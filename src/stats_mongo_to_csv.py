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


def get_champion_roles(champion_rates, riot_connection):
    champion_best_tag = dict()
    champion_merged_tags = dict()
    for champion_id in champion_rates.iterkeys():
        try:
            champion_info = riot_connection.get_champion_info(champion_id)
            tags = champion_info["tags"]
        except KeyError:
            print "Missing champion id: {}".format(champion_id)
            tags = ["UnknownChampion"]

        champion_best_tag[champion_id] = tags[0]
        champion_merged_tags[champion_id] = ",".join(sorted(tags))
    return champion_best_tag, champion_merged_tags


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

    # quickly load all player stats into RAM so we can join more quickly
    previous_time = time.time()
    riot_cache.preload_player_stats()
    riot_cache.precompute_champion_damage()
    logger.info("Preloading player stats took %.1f sec", time.time() - previous_time)

    agg_stats, agg_champion_stats = riot_cache.aggregate_champion_stats()

    columns = [
        "OtherChampions_Played",
        "OtherChampions_WinRate",
        "OtherPlayers_PlayRate",
        "OtherPlayers_WinRate",
        "PredictedPlayed",
        "PredictedWinRate"]

    previous_time = time.time()
    with io.open(args.output_csv, "w") as csv_out:
        csv_out.write(",".join(columns) + "\n")

        for player_stats in riot_cache.local_stats_cache.itervalues():
            assert isinstance(player_stats, PlayerStats)

            total_stats = player_stats.totals

            # each individual becomes one row
            for champion_id, champion_stats in player_stats.champion_stats.iteritems():
                champion_stats = ChampionStats(champion_stats)

                row = [total_stats.get_played(remove_stats=champion_stats),
                       total_stats.get_win_rate(remove_stats=champion_stats),
                       (agg_champion_stats[champion_id].played - champion_stats.played) / float(agg_stats.played - champion_stats.played),
                       (agg_champion_stats[champion_id].won - champion_stats.won) / float(agg_champion_stats[champion_id].played - champion_stats.played),
                       champion_stats.get_played(),
                       champion_stats.get_win_rate()]
                csv_out.write(",".join(str(x) for x in row) + "\n")


    logger.info("Pulling and converting data took %.1f sec", time.time() - previous_time)



if __name__ == "__main__":
    sys.exit(main())