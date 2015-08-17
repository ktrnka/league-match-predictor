from __future__ import unicode_literals
import collections
from datetime import datetime
import logging
import sys
import argparse
import pymongo
from riot_data import Summoner, Match

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
        return {"data.{}".format(k): v for k, v in data_query.iteritems()}


class ApiCache(object):
    def __init__(self, config):
        self.mongo_client = pymongo.MongoClient(config.get("mongo", "uri"))
        self.mongo_db = self.mongo_client.get_default_database()

        self.players = self.mongo_db[PLAYER_COLLECTION]
        self.matches = self.mongo_db[MATCH_COLLECTION]

        self.logger = logging.getLogger(__name__)

        self.new_matches = collections.Counter()
        self.new_players = collections.Counter()

    def queue_match(self, match):
        assert isinstance(match, Match)

        match_data = self.matches.find_one(Envelope.query_data({"matchId": match.id}))

        if not match_data:
            self.logger.debug("Queueing match %d", match.id)
            self.new_matches[True] += 1
            self.matches.insert_one(Envelope.wrap(match.export()))
            return True
        else:
            self.logger.debug("Already queued match %d", match.id)
            self.new_matches[False] += 1
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

    def get_queued_matches(self, max_records):
        # ranked 5v5
        previous_max_records = max_records
        for match_data in self.matches.find({"queued": True, "data.queueType": Match.QUEUE_RANKED_5}).limit(max_records):
            yield match_data
            max_records -= 1
        self.logger.info("Retrieved %d queued %s matches", previous_max_records - max_records, Match.QUEUE_RANKED_5)

        # solo 5v5
        previous_max_records = max_records
        if max_records > 0:
            for match_data in self.matches.find({"queued": True, "data.queueType": Match.QUEUE_RANKED_SOLO}).limit(max_records):
                yield match_data
                max_records -= 1
        self.logger.info("Retrieved %d queued %s matches", previous_max_records - max_records, Match.QUEUE_RANKED_SOLO)

    def get_players_recrawl(self, max_records):
        """Get an iterable of players to recrawl for match history or other purposes"""
        previous_max_records = max_records
        for player_data in self.players.find({"recrawl_at": None}).limit(max_records):
            yield Summoner(Envelope.unwrap(player_data).data)
            max_records -= 1
        self.logger.info("%d players without recrawl specified", previous_max_records - max_records)

        if max_records > 0:
            previous_max_records = max_records
            for player_data in self.players.find({"recrawl_at": {"$ne": None}}).sort("recrawl_at", pymongo.ASCENDING).limit(max_records):
                yield Summoner(Envelope.unwrap(player_data).data)
                max_records -= 1
            self.logger.info("%d players selected from earliest recrawl dates", previous_max_records - max_records)



    def update_players(self, players):
        for player in players:
            assert isinstance(player, Summoner)
            self.logger.debug("Updating %d -> %s", player.id, player.name)
            self.players.update(Envelope.query_data({"id": player.id}), Envelope.wrap(player.export(), False))

    def update_match_history_refresh(self, player, recrawl_date):
        result = self.players.update(Envelope.query_data({"id": player.id}), {"$set": {"recrawl_at": recrawl_date}})
        self.logger.debug("Updated %d match hist refresh for id %d", result["nModified"], player.id)

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
                         100. * self.new_matches[True] / (self.new_matches[True] + self.new_matches[False] + 1),
                         self.new_matches[True])
        self.logger.info("New players added: %.1f%% of queries (%d)",
                         100. * self.new_players[True] / (self.new_players[True] + self.new_players[False] + 1),
                         self.new_players[True])

    def summarize(self):
        """Summarize the status of the database overall"""
        players_queued, players_total = self.players.find({_ENVELOPE_IS_QUEUED: True}).count(), self.players.find({}).count()
        matches_queued, matches_total = self.matches.find({_ENVELOPE_IS_QUEUED: True}).count(), self.matches.find({}).count()

        return """
        {:,} players queued ({:.1f}% of {:,})
        {:,} matches queued ({:.1f}% of {:,})
        """.format(players_queued,
                   100. * players_queued / players_total,
                   players_total,
                   matches_queued,
                   100. * matches_queued / matches_total,
                   matches_total)

    def compact(self):
        # remove any matches from queues we don't care about
        result = self.matches.remove({"data.queueType": {"$nin": [Match.QUEUE_RANKED_5, Match.QUEUE_RANKED_SOLO]}})
        self.logger.info("Result from removing matches from queues we don't care about: %s", result)

        result = self.matches.update(Envelope.query_queued(False), {"$unset": make_unset()}, multi=True)
        self.logger.info("Result from pruning detail fields: %s", result)

    def get_matches(self):
        for match_data in self.matches.find(Envelope.query_queued(False)):
            yield Match(match_data["data"])


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    sys.exit(main())


def make_unset():
    """Make the proper dict of fields to unset from the match collection"""
    fields = []
    for i in xrange(0, 10):
        for field in "stats runes masteries timeline".split():
            fields.append("data.participants.{}.{}".format(i, field))

    return {k: None for k in fields}