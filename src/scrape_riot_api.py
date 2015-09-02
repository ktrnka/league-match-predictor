from __future__ import unicode_literals
import ConfigParser
import argparse
import logging
import sys
import datetime
import collections

import riot_api
import riot_api_cache
import riot_data
import requests.exceptions
from utilities import DevReminderError

EPOCH = datetime.datetime.utcfromtimestamp(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Verbose logging")
    parser.add_argument("config", help="Config file")
    return parser.parse_args()


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def queue_featured(riot_cache, riot_connection, queued_counts):
    logger = logging.getLogger(__name__)
    logger.info("Fetching featured matches and queueing players in them")

    summoner_names = []

    games, _ = riot_connection.get_featured_matches()
    for game_data in games:
        match = riot_data.Match.from_featured(game_data)

        for player in match.players:
            summoner_names.append(player.name)

    for player_names in chunks(summoner_names, 40):
        players = riot_connection.get_summoners(names=player_names)
        new_count, updated_count = riot_cache.update_player_names(players)
        queued_counts["player"] += new_count


def update_summoner_names(riot_cache, riot_connection, queued_counts, min_players=100, chunk_size=40):
    logger = logging.getLogger(__name__)

    max_players = max(min_players, queued_counts["player"] * 2) * 40
    logger.info("Fetching summoner names, up to %d", max_players)

    ids = []
    for player in riot_cache.get_queued_players(riot_cache.players, max_players):
        ids.append(player["data"]["id"])

    for player_ids in chunks(ids, chunk_size):
        players = riot_connection.get_summoners(ids=player_ids)
        riot_cache.update_player_names(players)


def refresh_match_history(riot_connection, riot_cache, queued_counts, player):
    try:
        matches = list(riot_connection.get_match_history(player.id))
        for match in matches:
            if match.is_interesting() and riot_cache.queue_match(match):
                queued_counts["match"] += 1

        return matches
    except riot_api.InvalidIdError:
        logging.getLogger(__name__).error("Bad summoner ID for player %s, removing", player)
        riot_cache.remove_player(player)

    return None


def update_match_histories(riot_cache, riot_connection, queued_counts, min_players=200):
    logger = logging.getLogger(__name__)

    max_players = max(min_players, queued_counts["player"] * 2)
    logger.info("Updating match history, up to %d players", max_players)

    refresh_outcomes = collections.Counter()

    for player in riot_cache.get_players_recrawl(max_players):
        assert isinstance(player, riot_data.Summoner)

        matches = refresh_match_history(riot_connection, riot_cache, queued_counts, player)

        # set the recrawl date
        if matches:
            riot_cache.update_match_history_refresh(player, get_recrawl_date(matches))
            refresh_outcomes["matches found, set recrawl"] += 1
        else:
            riot_cache.update_match_history_refresh(player, get_recrawl_delay(7))
            refresh_outcomes["matches not found, set 7-day recrawl"] += 1

    logger.info("Outcomes from refreshing summoners: {}".format(sorted(refresh_outcomes.items())))


def update_matches(riot_cache, riot_connection, queued_counts, min_matches=200):
    logger = logging.getLogger(__name__)

    raise DevReminderError("Refactor update_matches to fetch some matches and just queue some new players")

    max_matches = min(min_matches * 3, max(min_matches, queued_counts["match"] * 2))
    logger.info("Fetching queued matches, up to %d", max_matches)

    outcomes = collections.Counter()

    for match in riot_cache.get_queued_matches(max_matches):
        match_id = match["data"]["matchId"]

        try:
            match_info = riot_connection.get_match(match_id)

            riot_cache.dequeue_match(match_info)

            try:
                parsed_match = riot_data.Match(match_info)
                for player in parsed_match.players:
                    if player.id:
                        if riot_cache.queue_player(player.to_summoner()):
                            queued_counts["player"] += 1
                outcomes["parsed match and added players"] += 1
            except KeyError:
                # not all games have player identity info so skip if it fails
                outcomes["failed to parse match data"] += 1
        except requests.exceptions.HTTPError as e:
            logger.exception("Something went wrong in updating %d", match_id)

            if e.response.status_code == 404:
                riot_cache.remove_match(match_id)

    logger.info("Outcomes from fetching queued matches: %s", outcomes.most_common())


def get_recrawl_date(matches, max_matches=15):
    raise DevReminderError("Update get_recrawl_date for larger number of matches in match-list")

    date_format = "%Y-%m-%d %H:%M:%S"

    dates = [match.get_creation_datetime() for match in matches]
    start_date = min(dates)
    end_date = datetime.datetime.now()

    rate = (end_date - start_date) / len(dates)
    recrawl = end_date + rate * max_matches
    logging.getLogger(__name__).debug("%d matches from %s to %s, setting recrawl date to %s",
                                     len(matches),
                                     start_date.strftime(date_format),
                                     end_date.strftime(date_format),
                                     recrawl.strftime(date_format)
    )
    return (recrawl - EPOCH).total_seconds()

def get_recrawl_delay(num_days):
    return (datetime.datetime.now() - EPOCH + datetime.timedelta(days=num_days)).total_seconds()


def queue_master_plus(riot_cache, riot_connection, queued_counts):
    logger = logging.getLogger(__name__)

    added_players = 0
    for player in riot_connection.get_master_plus_solo():
        if riot_cache.queue_player(player):
            added_players += 1
            queued_counts["player"] += 1

    logger.info("Queued %d new summoners from crawling masters and challenger", added_players)


def update_summoner_leagues(riot_cache, riot_connection, queued_counts):
    logger = logging.getLogger(__name__)

    max_players = max(100, queued_counts["player"] * 2) * 40
    logger.info("Fetching summoner names, up to %d", max_players)

    players = []
    for player in riot_cache.get_queued_players(max_players, type=riot_api_cache.TYPE_LEAGUE):
        players.append(player)

    for player_chunk in chunks(players, 10):
        player_ids = [p.id for p in player_chunk]
        leagues = riot_connection.get_leagues(player_ids)

        for player, league in zip(player_chunk, leagues):
            riot_cache.set_league(player, league)


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
    riot_cache = riot_api_cache.ApiCache(config)

    try:
        queued_counts = collections.Counter()

        # extract players from featured matches
        queue_featured(riot_cache, riot_connection, queued_counts)

        # find players from masters and challenger and add them
        queue_master_plus(riot_cache, riot_connection, queued_counts)

        # make sure we have summoner names (for convenience)
        update_summoner_names(riot_cache, riot_connection, queued_counts)

        # make sure we have their leagues
        # update_summoner_leagues(riot_cache, riot_connection, queued_counts)

        update_match_histories(riot_cache, riot_connection, queued_counts)
        update_matches(riot_cache, riot_connection, queued_counts)

        logger.info("Found %d new players, %d new matches", queued_counts["player"], queued_counts["match"])

        riot_cache.compact()
    except requests.exceptions.HTTPError as exc:
        logger.exception("Unhandled HTTPError, aborting")

    riot_cache.log_summary()
    logger.info("Database status: %s", riot_cache.summarize())


if __name__ == "__main__":
    sys.exit(main())