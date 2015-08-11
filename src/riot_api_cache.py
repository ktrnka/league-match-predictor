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
    def __init__(self, data, is_queued, last_updated):
        self.data = data
        self.is_queued = is_queued
        self.last_updated = last_updated

    @staticmethod
    def wrap(data, is_queued=True):
        return Envelope(data, is_queued, datetime.now())

    @staticmethod
    def unwrap(mongo_object):
        return Envelope(mongo_object[_ENVELOPE_DATA], mongo_object[_ENVELOPE_IS_QUEUED], datetime.strptime(mongo_object[_ENVELOPE_UPDATED_DATE], _MONGO_DATE_FORMAT))

    def export(self):
        return {_ENVELOPE_DATA: self.data, _ENVELOPE_UPDATED_DATE: self.last_updated.strftime(_MONGO_DATE_FORMAT), _ENVELOPE_IS_QUEUED: self.is_queued}


class ApiCache(object):
    def __init__(self, config):
        self.mongo_client = pymongo.MongoClient(config.get("mongo", "uri"))
        self.mongo_db = self.mongo_client.get_default_database()

        self.players = self.mongo_db[PLAYER_COLLECTION]
        self.matches = self.mongo_db[MATCH_COLLECTION]

        self.logger = logging.getLogger(__name__)

        self.new_matches = collections.Counter()
        self.new_players = collections.Counter()

    @staticmethod
    def _wrap_data(data, is_queued=True):
        return Envelope.wrap(data, is_queued).export()

    def queue_match_id(self, id):
        match = self.matches.find_one({_ENVELOPE_DATA: {"matchId": id}})

        if not match:
            self.logger.debug("Queueing match %d", id)
            self.new_matches[True] += 1
            self.matches.insert_one(self._wrap_data({"matchId": id}))
        else:
            self.new_matches[False] += 1
            self.logger.debug("Already queued match %d", id)

    def queue_player_id(self, id):
        if not id:
            self.logger.error("ID is null")
            return
        player = self.players.find_one({_ENVELOPE_DATA: {"id": id}})

        if not player:
            self.new_players[True] += 1
            self.logger.debug("Queueing player %d", id)
            self.players.insert_one(self._wrap_data({"id": id}))
        else:
            self.new_matches[False] += 1
            self.logger.debug("Already queued player %d", id)

    def get_queued(self, collection):
        for item in collection.find({_ENVELOPE_IS_QUEUED: True}):
            yield item

    def update_players(self, players):
        for player in players:
            assert isinstance(player, Summoner)
            self.logger.info("Updating %d -> %s", player.id, player.name)
            self.players.update({_ENVELOPE_DATA: {"id": player.id}}, self._wrap_data(player.export(), False))

    def get_players(self):
        for player_data in self.players.find({_ENVELOPE_IS_QUEUED: True}):
            yield Summoner(player_data[_ENVELOPE_DATA])

    def update_match(self, match):
        self.matches.update({_ENVELOPE_DATA: {"matchId": match["matchId"]}}, self._wrap_data(match, False))

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