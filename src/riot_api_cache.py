from __future__ import unicode_literals
from datetime import datetime
import logging
import sys
import argparse
import pymongo
from src.riot_data import Summoner


class ApiCache(object):
    def __init__(self, config):
        self.mongo_client = pymongo.MongoClient(config.get("mongo", "uri"))
        self.mongo_db = self.mongo_client.get_default_database()

        self.players = self.mongo_db["summoners"]
        self.matches = self.mongo_db["matches"]

        self.logger = logging.getLogger(__name__)

    def _wrap_data(self, data, is_queued=True):
        return {"data": data, "updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "queued": is_queued}

    def queue_match_id(self, id):
        match = self.matches.find_one({"data": {"matchId": id}})

        if not match:
            self.logger.info("Queueing match %d", id)
            self.matches.insert_one(self._wrap_data({"matchId": id}))
        else:
            self.logger.info("Already queued match %d", id)

    def queue_player_id(self, id):
        player = self.players.find_one({"data": {"id": id}})

        if not player:
            self.logger.info("Queueing player %d", id)
            self.players.insert_one(self._wrap_data({"id": id}))
        else:
            self.logger.info("Already queued player %d", id)

    def get_queued(self, collection):
        for item in collection.find({"queued": True}):
            yield item

    def update_players(self, players):
        for player in players:
            assert isinstance(player, Summoner)
            self.logger.info("Updating %d -> %s", player.id, player.name)
            self.players.update({"data": {"id": player.id}}, self._wrap_data(player.export(), False))

    def update_match(self, match):
        self.matches.update({"data": {"matchId": match["matchId"]}}, self._wrap_data(match, False))


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    sys.exit(main())