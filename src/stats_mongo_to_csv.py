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

    # quickly load all player stats into RAM so we can join more quickly
    previous_time = time.time()
    riot_cache.preload_player_stats()
    riot_cache.precompute_champion_damage()
    logger.info("Preloading player stats took %.1f sec", time.time() - previous_time)

    total_rates = collections.Counter()
    champion_rates = collections.defaultdict(collections.Counter)

    # compute average win rate per champion and average win rate overall
    for player_stats in riot_cache.local_stats_cache.itervalues():
        assert isinstance(player_stats, PlayerStats)

        for champion_id, champion_stats in player_stats.champion_stats.iteritems():
            champion_stats = ChampionStats(champion_stats)
            total_rates["played"] += champion_stats.played
            total_rates["won"] += champion_stats.won

            champion_rates[champion_id]["played"] += champion_stats.played
            champion_rates[champion_id]["won"] += champion_stats.won

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

    columns = [
        "OtherChampions_Played",
        "OtherChampions_WinRate",
        "IsMainPrimaryRole",
        "SamePriRole_Played",
        "SamePriRole_PlayRate",
        "SamePriRole_WinRate",
        "IsMainDualRole",
        "SameDualRole_Played",
        "SameDualRole_PlayRate",
        "SameDualRole_WinRate",
        "OtherPlayers_PlayRate",
        "OtherPlayers_WinRate",
        "PredictedPlayed",
        "PredictedWinRate"]

    previous_time = time.time()
    with io.open(args.output_csv, "w") as csv_out:
        csv_out.write(",".join(columns) + "\n")

        for player_stats in riot_cache.local_stats_cache.itervalues():
            assert isinstance(player_stats, PlayerStats)

            # sum up the total stats and per-role stats
            total_stats = collections.Counter()
            stats_by_best_tag = collections.defaultdict(collections.Counter)
            stats_by_merged_tag = collections.defaultdict(collections.Counter)
            for champion_id, champion_stats in player_stats.champion_stats.iteritems():
                total_stats.update(champion_stats)
                stats_by_best_tag[champion_best_tag[champion_id]].update(champion_stats)
                stats_by_merged_tag[champion_merged_tags[champion_id]].update(champion_stats)

            stats_by_best_tag = {k: ChampionStats(v) for k, v in stats_by_best_tag.iteritems()}
            stats_by_merged_tag = {k: ChampionStats(v) for k, v in stats_by_merged_tag.iteritems()}

            total_stats = ChampionStats(total_stats)

            main_pri_tag = max(stats_by_best_tag.keys(), key=lambda i: stats_by_best_tag[i].played)
            main_dual_tag = max(stats_by_merged_tag.keys(), key=lambda i: stats_by_merged_tag[i].played)

            # each individual becomes one row
            for champion_id, champion_stats in player_stats.champion_stats.iteritems():
                champion_stats = ChampionStats(champion_stats)

                # remove the predicted fields
                total_played = total_stats.played - champion_stats.played
                total_won = total_stats.won - champion_stats.won

                same_primary_played = stats_by_best_tag[champion_best_tag[champion_id]].played - champion_stats.played
                same_primary_won = stats_by_best_tag[champion_best_tag[champion_id]].won - champion_stats.won

                same_dual_played = stats_by_merged_tag[champion_merged_tags[champion_id]].played - champion_stats.played
                same_dual_won = stats_by_merged_tag[champion_merged_tags[champion_id]].won - champion_stats.won

                row = [total_played,
                       total_won / float(total_played),
                       main_pri_tag == champion_best_tag[champion_id],
                       same_primary_played,
                       same_primary_played / float(total_played),
                       same_primary_won / float(same_primary_played + 0.01),
                       main_dual_tag == champion_merged_tags[champion_id],
                       same_dual_played,
                       same_dual_played / float(total_played),
                       same_dual_won / float(same_dual_played + 0.01),
                       (champion_rates[champion_id]["played"] - champion_stats.played) / float(total_rates["played"] - champion_stats.played),
                       (champion_rates[champion_id]["won"] - champion_stats.won) / float(champion_rates[champion_id]["played"] - champion_stats.played),
                       champion_stats.played,
                       champion_stats.won / float(champion_stats.played)]
                csv_out.write(",".join(str(x) for x in row) + "\n")


    logger.info("Pulling and converting data took %.1f sec", time.time() - previous_time)



if __name__ == "__main__":
    sys.exit(main())