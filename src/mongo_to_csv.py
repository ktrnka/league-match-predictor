from __future__ import unicode_literals
import ConfigParser
import logging
import sys
import argparse
import io
import time
import collections

import riot_api
import riot_api_cache
import riot_data
import utilities


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Verbose logging")
    parser.add_argument("--update", default=False, action="store_true", help="Update local cache before generating data")
    parser.add_argument("--no-updates", default=False, action="store_true", help="Disable even the most basic updating")
    parser.add_argument("--max-matches", default=-1, type=int, help="Max matches to output to file")
    parser.add_argument("config", help="Config file")
    parser.add_argument("output_csv", help="Output file to use for machine learning")
    return parser.parse_args()


def champion_set_to_indicators(champion_ids):
    return [int(i in champion_ids) for i in riot_data.Champion.known_ids]


def make_champion_indicator_names(riot_connection):
    return [riot_connection.get_champion_info(champion_id)["name"] for champion_id in riot_data.Champion.known_ids]

def update_stats(match_history_stats, match):
    """Update our running tally of stats from the match history"""
    for player in match.players:
        for conditional_key in [player.champion_id, (match.version, player.champion_id), player.get_champion_spells_key()]:
            if conditional_key not in match_history_stats:
                match_history_stats[conditional_key] = riot_data.ChampionStats.from_wins_played(0, 0)
            if player.team_id == match.get_winning_team_id():
                match_history_stats[conditional_key].won += 1
            match_history_stats[conditional_key].played += 1


def smooth_winrate(primary_champion_stats, secondary_champion_stats, crossover=20, remove_games=0, remove_wins=0):
    if not primary_champion_stats:
        if secondary_champion_stats:
            return secondary_champion_stats.get_win_rate(remove_games=remove_games, remove_wins=remove_wins)
        else:
            return 0.5

    assert isinstance(primary_champion_stats, riot_data.ChampionStats)
    assert isinstance(secondary_champion_stats, riot_data.ChampionStats)

    primary_weight = primary_champion_stats.get_played(remove_games=remove_games) / float(primary_champion_stats.get_played(remove_games=remove_games) + crossover)

    return primary_champion_stats.get_win_rate(remove_games=remove_games, remove_wins=remove_wins) * primary_weight + secondary_champion_stats.get_win_rate(remove_games=remove_games, remove_wins=remove_wins) * (1 - primary_weight)


def update_player_history(player_history_stats, match, player):
    assert isinstance(match, riot_data.Match)
    assert isinstance(player, riot_data.Participant)

    if player.id not in player_history_stats:
        player_history_stats[player.id] = riot_data.PlayerStats.from_id(player.id)

    if player.team_id == match.get_winning_team_id():
        player_history_stats[player.id].losing_streak_games = 0
        player_history_stats[player.id].winning_streak_games += 1
    else:
        player_history_stats[player.id].losing_streak_games += 1
        player_history_stats[player.id].winning_streak_games = 0


def get_player_streaks(player_histories, player_id):
    if player_id in player_histories:
        return player_histories[player_id].winning_streak_games, player_histories[player_id].losing_streak_games
    else:
        return 0, 0


def generate_dataset(riot_connection, riot_cache, agg_champion_stats, agg_stats, champion_damage_types, csv_out, max_matches=-1):
    logger = logging.getLogger()
    heartbeat_logger = logging.getLogger("generate_dataset.heartbeat")
    heartbeat_logger.addFilter(utilities.ThrottledFilter(delay_seconds=10))
    previous_time = time.time()

    player_features = []
    for team in ["Blue", "Red"]:
        for player in range(1, 6):
            player_features.append("{}_{}_Champ".format(team, player))
            player_features.append("{}_{}_WinRate(player champion season)".format(team, player))
            player_features.append("{}_{}_NumGames(player champion season)".format(team, player))
            player_features.append("{}_{}_PlayRate(champion season)".format(team, player))
            player_features.append("{}_{}_WinRate(champion season)".format(team, player))
            player_features.append("{}_{}_WinRate(player season)".format(team, player))
            player_features.append("{}_{}_NumGames(player season)".format(team, player))
            player_features.append("{}_{}_WinRate(champion recent)".format(team, player))
            player_features.append("{}_{}_WinRate(champion version recent)".format(team, player))
            player_features.append("{}_{}_WinRate(champion summoners recent)".format(team, player))
            player_features.append("{}_{}_WinningStreak(player)".format(team, player))
            player_features.append("{}_{}_LosingStreak(player)".format(team, player))
            for summoner_spell_id in [1, 2]:
                player_features.append("{}_{}_Spell_{}".format(team, player, summoner_spell_id))
        for damage_type in ["magic", "physical", "true"]:
            player_features.append("{}_DamagePercent({})".format(team, damage_type))
    columns = ["MatchId", "QueueType", "GameVersion", "Blue_Tier", "Red_Tier"] + player_features + ["IsBlueWinner"]

    match_history_stats = dict()
    player_history_stats = dict()

    csv_out.write(",".join(columns) + "\n")

    previous_creation_time = 0
    timer = utilities.EstCompletionTimer().start(riot_cache.matches.find({}).count())

    for match_num, match in enumerate(riot_cache.get_matches(chronological=True)):
        assert previous_creation_time <= match.timestamp
        winner = match.get_winning_team_id()

        picks = match.get_picks_role()
        teams = sorted(picks.keys())

        tiers = match.get_team_tiers_numeric()

        player_features = []
        for team in teams:
            damage_types = collections.Counter()

            for player in picks[team]:
                assert isinstance(player, riot_data.Participant)
                player_features.append(riot_connection.get_champion_name(player.champion_id))

                player_stats = riot_cache.get_player_stats(player.id)
                assert isinstance(player_stats, riot_data.PlayerStats)

                damage_types.update(champion_damage_types[player.champion_id])

                remove_match_player_stats = 0
                remove_win_player_stats = 0
                if player_stats.modify_date > match.timestamp and match.full_data["season"] == "SEASON2015":
                    remove_match_player_stats = 1
                    remove_win_player_stats = int(winner == team)
                assert team == player.team_id

                champion_stats = player_stats.get_champion_stats(player.champion_id)

                # win rate on this champion
                player_champ_winrate = smooth_winrate(champion_stats, agg_champion_stats[player.champion_id],
                                                      crossover=10, remove_games=remove_match_player_stats,
                                                      remove_wins=remove_win_player_stats)
                player_features.append(player_champ_winrate)
                player_features.append(champion_stats.get_played(remove_games=remove_match_player_stats))

                # play rate in general and win rate in general for all players
                player_features.append(agg_champion_stats[player.champion_id].played / float(agg_stats.played))
                player_features.append(agg_champion_stats[player.champion_id].won / float(
                    agg_champion_stats[player.champion_id].played))

                # win rate overall
                player_features.append(player_stats.totals.get_win_rate(remove_games=remove_match_player_stats,
                                                                        remove_wins=remove_win_player_stats))
                player_features.append(player_stats.totals.get_played(remove_games=remove_match_player_stats))

                # note that the games are processed chronologically so we don't need to remove the current game from the historical
                # win rates

                # win rate from match histories
                try:
                    player_features.append(match_history_stats[player.champion_id].get_win_rate())
                except KeyError:
                    player_features.append(0.5)

                # win rate from match histories on this patch
                patch_champ_stats_mh = match_history_stats.get((match.version, player.champion_id), None)
                champ_stats_mh = match_history_stats.get(player.champion_id, None)
                champion_version_winrate = smooth_winrate(patch_champ_stats_mh, champ_stats_mh, crossover=10)
                player_features.append(champion_version_winrate)

                # win rate of the champion with the specific summoners from match history with backoff
                champion_spells_winrate = match_history_stats.get(player.get_champion_spells_key(), None)
                champion_spells_smoothed = smooth_winrate(champion_spells_winrate, champ_stats_mh, crossover=10)
                player_features.append(champion_spells_smoothed)

                winning_streak, losing_streak = get_player_streaks(player_history_stats, player.id)
                player_features.append(winning_streak)
                player_features.append(losing_streak)

                for summoner_spell_id in player.spells:
                    player_features.append(riot_connection.get_summoner_spell_name(summoner_spell_id))

                update_player_history(player_history_stats, match, player)

            damage_sum = float(sum(damage_types.itervalues()))
            for damage_type in ["magic", "physical", "true"]:
                player_features.append(damage_types[damage_type] / damage_sum)

        is_blue_winner = int(winner == teams[0])

        row = [match.id, match.queue_type, match.version] + [tiers[t] for t in teams] + player_features + [is_blue_winner]

        # history columns are completely inaccurate until this point
        if match_num > 2000:
            csv_out.write(",".join(str(x) for x in row) + "\n")

            if 0 < max_matches <= match_num + 2000:
                break

        previous_creation_time = match.timestamp
        update_stats(match_history_stats, match)

        timer.update()
        heartbeat_logger.info(timer.log_info())

    logger.info("Pulling and converting data took %.1f sec", time.time() - previous_time)


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

    riot_connection = riot_api.RiotService.from_config(config)

    local_cache = riot_api_cache.MemoizeCache(config, riot_connection)

    if args.update:
        remote_cache = riot_api_cache.ApiCache(config)
        local_cache.load(remote_cache)
    elif not args.no_updates:
        local_cache.update_players()

    # quickly load all player stats into RAM so we can join more quickly
    previous_time = time.time()
    champion_damage_types = local_cache.compute_champion_damage_types()
    logger.info("Computing champion damage types took %.1f sec", time.time() - previous_time)

    previous_time = time.time()
    agg_stats, agg_champion_stats = local_cache.aggregate_champion_stats()
    logger.info("Computing champion win rates took %.1f sec", time.time() - previous_time)


    with io.open(args.output_csv, "w") as csv_out:
        generate_dataset(riot_connection, local_cache, agg_champion_stats, agg_stats, champion_damage_types, csv_out, max_matches=args.max_matches)



if __name__ == "__main__":
    sys.exit(main())