from __future__ import unicode_literals
import ConfigParser
import argparse
import pprint
import sys
import datetime

from riot_api import *
from riot_api_cache import ApiCache
from riot_data import Participant, Match
import requests.exceptions

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
    logger.info("Fetching featured matches and queueing matches and players")

    summoner_names = []

    games, _ = riot_connection.get_featured_matches()
    for game_data in games:
        match = Match.from_featured(game_data)

        if riot_cache.queue_match(match):
            queued_counts["match"] += 1

        for player in match.players:
            summoner_names.append(player.name)

    for player_names in chunks(summoner_names, 40):
        players = riot_connection.get_summoners(names=player_names)
        riot_cache.update_players(players)


def update_summoners(riot_cache, riot_connection, queued_counts):
    logger = logging.getLogger(__name__)

    max_players = max(100, queued_counts["player"] * 2) * 40
    logger.info("Fetching queued summoners, up to %d", max_players)

    ids = []
    for player in riot_cache.get_queued(riot_cache.players, max_players):
        ids.append(player["data"]["id"])

    for player_ids in chunks(ids, 40):
        players = riot_connection.get_summoners(ids=player_ids)
        riot_cache.update_players(players)


def update_summoner_stats(riot_cache, riot_connection):
    logger = logging.getLogger(__name__)

    max_players = 500
    logger.info("Updating summoner stats, up to %d", max_players)
    success = collections.Counter()
    for player in riot_cache.get_queued_players_stats(max_players):
        assert isinstance(player, Summoner)

        try:
            player_stats = riot_connection.get_summoner_ranked_stats(player.id)
            riot_cache.update_player_stats(player.id, player_stats)
            success[True] += 1
        except SummonerNotFoundError:
            # This actually happens quite often
            logger.debug("Player %d no stats, setting to empty stats", player.id)
            riot_cache.update_player_stats(player.id, {})
            success[False] += 1

    logger.info("Updating stats succeeded %d / %d times", success[True], success[True] + success[False])


def update_matches(riot_cache, riot_connection, queued_counts):
    logger = logging.getLogger(__name__)

    max_matches = max(600, queued_counts["match"] * 2)
    logger.info("Fetching queued matches, up to %d", max_matches)

    outcomes = collections.Counter()

    for match in riot_cache.get_queued_matches(max_matches):
        match_id = match["data"]["matchId"]

        try:
            match_info = riot_connection.get_match(match_id)
            riot_cache.update_match(match_info)

            try:
                parsed_match = Match(match_info)
                for player in parsed_match.players:
                    if player.id:
                        if riot_cache.queue_player(player):
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


def get_recrawl_date(matches):
    date_format = "%Y-%m-%d %H:%M:%S"

    dates = [match.get_creation_datetime() for match in matches]
    rate = (max(dates) - min(dates)) / len(dates)
    recrawl = max(dates) + rate * 15
    logging.getLogger(__name__).debug("%d matches from %s to %s, setting recrawl date to %s",
                                     len(matches),
                                     min(dates).strftime(date_format),
                                     max(dates).strftime(date_format),
                                     recrawl.strftime(date_format)
    )
    return (recrawl - EPOCH).total_seconds()


def queue_from_match_histories(riot_cache, riot_connection, queued_counts):
    logger = logging.getLogger(__name__)

    max_players = max(100, queued_counts["player"] * 2)
    logger.info("Fetching match history for queued summoners, up to %d", max_players)

    for player in riot_cache.get_players_recrawl(max_players):
        assert isinstance(player, Summoner)

        try:
            matches = list(riot_connection.get_match_history(player.id))
            for match in matches:
                if riot_cache.queue_match(match):
                    queued_counts["match"] += 1

            if matches:
                riot_cache.update_match_history_refresh(player, get_recrawl_date(matches))

        except InvalidIdError:
            logging.getLogger(__name__).error("Bad summoner ID for player %s, removing", player)
            riot_cache.remove_player(player)


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
        queued_counts = collections.Counter()

        update_summoner_stats(riot_cache, riot_connection)
        queue_from_match_histories(riot_cache, riot_connection, queued_counts)
        update_summoners(riot_cache, riot_connection, queued_counts)
        update_matches(riot_cache, riot_connection, queued_counts)

        # queue up featured matches last because they will automatically fail to get match data for a while
        queue_featured(riot_cache, riot_connection, queued_counts)

        logger.info("Found %d new players, %d new matches", queued_counts["player"], queued_counts["match"])

        riot_cache.compact()
    except requests.exceptions.HTTPError as exc:
        logger.exception("Unhandled HTTPError, aborting")

    riot_cache.log_summary()
    logger.info("Database status: %s", riot_cache.summarize())


if __name__ == "__main__":
    sys.exit(main())