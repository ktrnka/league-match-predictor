from __future__ import unicode_literals
import collections
from datetime import datetime
import logging
import sys
import argparse
import pymongo
from riot_data import Summoner

MATCH_COLLECTION = "matches"

PLAYER_COLLECTION = "summoners"


class ApiCache(object):
    def __init__(self, config):
        self.mongo_client = pymongo.MongoClient(config.get("mongo", "uri"))
        self.mongo_db = self.mongo_client.get_default_database()

        self.players = self.mongo_db[PLAYER_COLLECTION]
        self.matches = self.mongo_db[MATCH_COLLECTION]

        self.logger = logging.getLogger(__name__)

        self.new_matches = collections.Counter()
        self.new_players = collections.Counter()

    def _wrap_data(self, data, is_queued=True):
        return {"data": data, "updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "queued": is_queued}

    def queue_match_id(self, id):
        match = self.matches.find_one({"data": {"matchId": id}})

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
        player = self.players.find_one({"data": {"id": id}})

        if not player:
            self.new_players[True] += 1
            self.logger.debug("Queueing player %d", id)
            self.players.insert_one(self._wrap_data({"id": id}))
        else:
            self.new_matches[False] += 1
            self.logger.debug("Already queued player %d", id)

    def get_queued(self, collection):
        for item in collection.find({"queued": True}):
            yield item

    def update_players(self, players):
        for player in players:
            assert isinstance(player, Summoner)
            self.logger.info("Updating %d -> %s", player.id, player.name)
            self.players.update({"data": {"id": player.id}}, self._wrap_data(player.export(), False))

    def get_players(self):
        for player_data in self.players.find({"queued": True}):
            yield Summoner(player_data["data"])

    def update_match(self, match):
        self.matches.update({"data": {"matchId": match["matchId"]}}, self._wrap_data(match, False))

    def log_summary(self):
        self.logger.info("New matches added: %.1f%% of queries (%d)", 100. * self.new_matches[True] / (self.new_matches[True] + self.new_matches[False]), self.new_matches[True])
        self.logger.info("New players added: %.1f%% of queries (%d)", 100. * self.new_players[True] / (self.new_players[True] + self.new_players[False]), self.new_players[True])

    def summarize(self):
        """Summarize the status of the database overall"""
        players_queued, players_total = self.players.find({"queued": True}).count(), self.players.find({}).count()
        matches_queued, matches_total = self.matches.find({"queued": True}).count(), self.matches.find({}).count()

        return """
        {:,} players queued ({:.1f}% of {:,})
        {:,} matches queued ({:.1f}% of {:,})
        """.format(players_queued, 100. * players_queued / players_total, players_total, matches_queued, 100. * matches_queued / matches_total, matches_total)


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    sys.exit(main())