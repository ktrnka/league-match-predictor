from __future__ import unicode_literals
import ConfigParser
import argparse
import sys

from riot_api import *
from riot_api_cache import ApiCache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Verbose logging")
    parser.add_argument("config", help="Config file")
    return parser.parse_args()


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def queue_featured(riot_cache, riot_connection):
    games, _ = riot_connection.get_featured_matches()
    for game in games:
        riot_cache.queue_match_id(game["gameId"])

        for player in game["participants"]:
            player = FeaturedParticipant.from_joined(player)
            summoner = riot_connection.get_summoner(name=player.name)

            riot_cache.queue_player_id(summoner.id)


def update_summoners(riot_cache, riot_connection):
    ids = []
    for player in riot_cache.get_queued(riot_cache.players):
        ids.append(player["data"]["id"])

    for player_ids in chunks(ids, 40):
        players = riot_connection.get_summoners(ids=player_ids)
        riot_cache.update_players(players)


def update_matches(riot_cache, riot_connection):
    logger = logging.getLogger("update_matches")
    for match in riot_cache.get_queued(riot_cache.matches):
        match_id = match["data"]["matchId"]
        # logger.info("Updating from match %d", match_id)

        try:
            match_info = riot_connection.get_match(match_id)
            riot_cache.update_match(match_info)

            try:
                parsed_match = Match(match_info)
                for player in parsed_match.players:
                    if player.id:
                        riot_cache.queue_player_id(player.id)
            except KeyError:
                # not all games have player identity info so skip if it fails
                pass
        except requests.exceptions.HTTPError:
            logger.exception("Something went wrong in updating %d", match_id)


def queue_from_match_histories(riot_cache, riot_connection):
    for player in riot_cache.get_players():
        assert isinstance(player, Summoner)

        try:
            matches = riot_connection.get_match_history(player.id)
            for match in matches:
                riot_cache.queue_match_id(match.id)
        except ValueError:
            logging.getLogger(__name__).error("Bad summoner ID for player %s", player)


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
        # queue_featured(riot_cache, riot_connection)
        # queue_from_match_histories(riot_cache, riot_connection)
        #
        # update_summoners(riot_cache, riot_connection)
        update_matches(riot_cache, riot_connection)
    except requests.exceptions.HTTPError as exc:
        logger.exception("Unhandled HTTPError, aborting")

    riot_cache.log_summary()
    logger.info("Database status: %s", riot_cache.summarize())


if __name__ == "__main__":
    sys.exit(main())