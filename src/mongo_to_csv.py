from __future__ import unicode_literals
import ConfigParser
import logging
import sys
import argparse
import io
import time
import collections
from src.riot_api import RiotService

from src.riot_api_cache import ApiCache
from src.riot_data import Champion, Participant


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

    player_features = []
    for team in ["Blue", "Red"]:
        for player in range(1, 6):
            player_features.append("{}_{}_Champ".format(team, player))
            player_features.append("{}_{}_WinRate".format(team, player))
            player_features.append("{}_{}_Played".format(team, player))
            for summoner_spell_id in [1, 2]:
                player_features.append("{}_{}_Spell_{}".format(team, player, summoner_spell_id))
        for damage_type in ["magic", "physical", "true"]:
            player_features.append("{}_Damage_{}".format(team, damage_type))
    columns = ["QueueType", "Blue_Tier", "Red_Tier"] + player_features + ["IsBlueWinner"]

    # quickly load all player stats into RAM so we can join more quickly
    previous_time = time.time()
    riot_cache.preload_player_stats()
    riot_cache.precompute_champion_damage()
    logger.info("Preloading player stats took %.1f sec", time.time() - previous_time)

    previous_time = time.time()
    with io.open(args.output_csv, "w") as csv_out:
        csv_out.write(",".join(columns) + "\n")

        for match in riot_cache.get_matches():
            winner = match.get_winning_team_id()

            picks = match.get_picks_role()
            teams = sorted(picks.keys())

            tiers = match.get_team_tiers_numeric()

            player_features = []
            for team in teams:
                damage_types = collections.Counter()

                for player in picks[team]:
                    assert isinstance(player, Participant)
                    player_features.append(riot_connection.get_champion_name(player.champion_id))

                    player_stats = riot_cache.get_player_stats(player.id, force_cache=True)

                    damage_types.update(riot_cache.get_champion_damage_types(player.champion_id))

                    remove_match=False
                    if player_stats.modify_date > match.creation_time:
                        remove_match=True

                    player_features.append(player_stats.get_win_rate(player.champion_id, remove=remove_match, won=(winner == team)))
                    player_features.append(player_stats.get_games_played(player.champion_id, remove=remove_match))

                    for summoner_spell_id in player.spells:
                        player_features.append(riot_connection.get_summoner_spell_name(summoner_spell_id))

                damage_sum = float(sum(damage_types.itervalues()))
                for damage_type in ["magic", "physical", "true"]:
                    player_features.append(damage_types[damage_type] / damage_sum)

            is_blue_winner = int(winner == teams[0])

            row = [match.queue_type] + [tiers[t] for t in teams] + player_features + [is_blue_winner]
            csv_out.write(",".join(str(x) for x in row) + "\n")

    logger.info("Pulling and converting data took %.1f sec", time.time() - previous_time)



if __name__ == "__main__":
    sys.exit(main())