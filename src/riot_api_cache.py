from __future__ import unicode_literals
import collections
from datetime import datetime
import logging
import sys
import argparse
import pymongo
from riot_data import Summoner

_ENVELOPE_UPDATED_DATE = "updated"

_ENVELOPE_IS_QUEUED = "queued"

_ENVELOPE_DATA = "data"

_MONGO_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

MATCH_COLLECTION = "matches"

PLAYER_COLLECTION = "summoners"


class Envelope(object):
    """
    Wraps API data with metadata such as whether it's a queued identifier, when it was last updated, and so on.
    """
    def __init__(self, data, is_queued, last_updated):
        self.data = data
        self.is_queued = is_queued
        self.last_updated = last_updated

    @staticmethod
    def wrap(data, is_queued=True):
        return {_ENVELOPE_DATA: data, _ENVELOPE_UPDATED_DATE: datetime.now().strftime(_MONGO_DATE_FORMAT), _ENVELOPE_IS_QUEUED: is_queued}

    @staticmethod
    def unwrap(mongo_object):
        return Envelope(mongo_object[_ENVELOPE_DATA], mongo_object[_ENVELOPE_IS_QUEUED], datetime.strptime(mongo_object[_ENVELOPE_UPDATED_DATE], _MONGO_DATE_FORMAT))

    @staticmethod
    def query_queued(is_queued):
        return {_ENVELOPE_IS_QUEUED: is_queued}

    @staticmethod
    def query_data(data_query):
        return {_ENVELOPE_DATA: data_query}


class ApiCache(object):
    def __init__(self, config):
        self.mongo_client = pymongo.MongoClient(config.get("mongo", "uri"))
        self.mongo_db = self.mongo_client.get_default_database()

        self.players = self.mongo_db[PLAYER_COLLECTION]
        self.matches = self.mongo_db[MATCH_COLLECTION]

        self.logger = logging.getLogger(__name__)

        self.new_matches = collections.Counter()
        self.new_players = collections.Counter()

    def queue_match_id(self, id):
        match = self.matches.find_one(Envelope.query_data({"matchId": id}))

        if not match:
            self.logger.debug("Queueing match %d", id)
            self.new_matches[True] += 1
            self.matches.insert_one(Envelope.wrap({"matchId": id}))
            return True
        else:
            self.new_matches[False] += 1
            self.logger.debug("Already queued match %d", id)
            return False

    def queue_player(self, player):
        assert isinstance(player, Summoner)
        if not player.id:
            self.logger.error("ID is null: {}".format(player))
            return

        player_data = self.players.find_one(Envelope.query_data({"id": player.id}))

        if not player_data:
            self.new_players[True] += 1
            self.logger.debug("Queueing player %d", player.id)
            self.players.insert_one(Envelope.wrap(player.export()))
            return True
        else:
            self.new_players[False] += 1
            self.logger.debug("Already queued player %d", player.id)
            return False

    def get_queued(self, collection, max_records):
        for item in collection.find(Envelope.query_queued(True)).limit(max_records):
            yield item

    def get_players(self, max_records):
        for player_data in self.players.find(Envelope.query_queued(True)).limit(max_records):
            yield Summoner(Envelope.unwrap(player_data).data)

    def update_players(self, players):
        for player in players:
            assert isinstance(player, Summoner)
            self.logger.debug("Updating %d -> %s", player.id, player.name)
            self.players.update(Envelope.query_data({"id": player.id}), Envelope.wrap(player.export(), False))

    def update_match(self, match):
        self.matches.update(Envelope.query_data({"matchId": match["matchId"]}), Envelope.wrap(match, False))

    def remove_match(self, match_id):
        result = self.matches.delete_one(Envelope.query_data({"matchId": match_id}))
        self.logger.info("Removed %d objects for match id %d", result.deleted_count, match_id)

    def remove_player(self, player):
        assert isinstance(player, Summoner)

        result = self.players.delete_one(Envelope.query_data({"id": player.id}))
        self.logger.info("Removed %d objects for player {}".format(player), result.deleted_count)

    def log_summary(self):
        self.logger.info("New matches added: %.1f%% of queries (%d)",
                         100. * self.new_matches[True] / (self.new_matches[True] + self.new_matches[False]),
                         self.new_matches[True])
        self.logger.info("New players added: %.1f%% of queries (%d)",
                         100. * self.new_players[True] / (self.new_players[True] + self.new_players[False]),
                         self.new_players[True])

    def summarize(self):
        """Summarize the status of the database overall"""
        players_queued, players_total = self.players.find({_ENVELOPE_IS_QUEUED: True}).count(), self.players.find({}).count()
        matches_queued, matches_total = self.matches.find({_ENVELOPE_IS_QUEUED: True}).count(), self.matches.find({}).count()

        return """
        {:,} players queued ({:.1f}% of {:,})
        {:,} matches queued ({:.1f}% of {:,})
        """.format(players_queued, 100. * players_queued / players_total, players_total, matches_queued,
                   100. * matches_queued / matches_total, matches_total)


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    sys.exit(main())