from __future__ import unicode_literals
import collections
from datetime import datetime, timedelta
import pprint
import time
import logging
import sys
import argparse
import functools32

import pymongo
import pymongo.errors
import requests
import requests.exceptions
import riot_api

import riot_data
import utilities


_ENVELOPE_UPDATED_DATE = "updated"

_ENVELOPE_IS_QUEUED = "queued"

_ENVELOPE_DATA = "data"

_ENVELOPE_KEY = "key"

_MONGO_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

MATCH_COLLECTION = "matches"

PLAYER_COLLECTION = "summoners"

TYPE_LEAGUE = "league"
TYPE_QUEUED = "queued"


class Envelope(object):
    """
    Wraps API data with metadata such as whether it's a queued identifier, when it was last updated, and so on.
    """

    def __init__(self, data, is_queued, last_updated, recrawl_start_time):
        self.data = data
        self.is_queued = is_queued
        self.last_updated = last_updated

        self.recrawl_start_time = None

    @staticmethod
    def wrap(data, is_queued=True):
        return {_ENVELOPE_DATA: data,
                _ENVELOPE_UPDATED_DATE: datetime.now().strftime(_MONGO_DATE_FORMAT),
                _ENVELOPE_IS_QUEUED: is_queued}

    @staticmethod
    def unwrap(mongo_object):
        return Envelope(mongo_object[_ENVELOPE_DATA],
                        mongo_object[_ENVELOPE_IS_QUEUED],
                        datetime.strptime(mongo_object[_ENVELOPE_UPDATED_DATE], _MONGO_DATE_FORMAT),
                        recrawl_start_time=mongo_object.get("recrawl_start_time", None))

    @staticmethod
    def query_queued(is_queued):
        return {_ENVELOPE_IS_QUEUED: is_queued}

    @staticmethod
    def query_data(data_query):
        return {"data.{}".format(k): v for k, v in data_query.iteritems()}

    @staticmethod
    def set_queued(queued):
        return {"$set": {_ENVELOPE_IS_QUEUED: queued}}


class ComputeResultEnvelope(object):
    """Wraps any dict-like structure that is the result of computation, like stat breakdowns by champion"""

    def __init__(self, name, data, created):
        self.key = name
        self.data = data
        self.updated = created

        assert isinstance(name, basestring)
        assert isinstance(created, datetime)

    @staticmethod
    def unwrap(mongo_object):
        return ComputeResultEnvelope(mongo_object[_ENVELOPE_KEY],
                                     mongo_object[_ENVELOPE_DATA],
                                     datetime.strptime(mongo_object[_ENVELOPE_UPDATED_DATE], _MONGO_DATE_FORMAT))

    def wrap(self):
        return {_ENVELOPE_KEY: self.key,
                _ENVELOPE_DATA: self.data,
                _ENVELOPE_UPDATED_DATE: self.updated.strftime(_MONGO_DATE_FORMAT)}

    def is_stale(self, max_age_days=7):
        return datetime.now() - self.updated > timedelta(max_age_days)

    @staticmethod
    def query_key(name):
        return {_ENVELOPE_KEY: name}


class ApiCache(object):
    def __init__(self, config, config_section="mongo"):
        self.outcomes = collections.Counter()

        # matches
        self.__matches_mongo_client = pymongo.MongoClient(config.get(config_section, "match_uri"))
        self.__matches_mongo_db = self.__matches_mongo_client.get_default_database()
        self.matches = self.__matches_mongo_db[MATCH_COLLECTION]

        # summoners
        self.__summoners_mongo_client = pymongo.MongoClient(config.get(config_section, "summoner_uri"))
        self.__summoners_mongo_db = self.__summoners_mongo_client.get_default_database()
        self.players = self.__summoners_mongo_db[PLAYER_COLLECTION]

        self.logger = logging.getLogger(__name__)

        self.new_matches = collections.Counter()
        self.new_players = collections.Counter()

        self._setup_mongo()

    def queue_match(self, match):
        assert isinstance(match, riot_data.MatchReference)

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
        assert isinstance(player, riot_data.Summoner)
        if not player.id:
            self.logger.error("ID is null: {}".format(player))
            return

        player_data = self.players.find_one(Envelope.query_data({"id": player.id}))

        if not player_data:
            self.new_players[True] += 1
            self.logger.debug("Queueing player %d", player.id)
            self.players.insert_one(Envelope.wrap(player.export(), is_queued=(player.name is not None)))
            return True
        else:
            self.new_players[False] += 1
            self.logger.debug("Already queued player %d", player.id)
            return False

    def get_queued_players(self, max_records, type=TYPE_QUEUED):
        if type == TYPE_QUEUED:
            cursor = self.players.find(Envelope.query_queued(True))
        elif type == TYPE_LEAGUE:
            cursor = self.players.find(Envelope.query_data({"league": {"$exists": False}}))
        else:
            raise ValueError("Type not recognized: {}".format(type))

        if max_records > 0:
            cursor = cursor.limit(max_records)

        for item in cursor:
            yield riot_data.Summoner(Envelope.unwrap(item).data)

    def get_queued_matches(self, max_records):
        # ranked 5v5
        previous_max_records = max_records
        for match_data in self.matches.find({"queued": True, "data.queue": riot_data.Match.QUEUE_RANKED_5}).limit(
                max_records):
            yield match_data
            max_records -= 1
        self.logger.info("Retrieved %d queued %s matches", previous_max_records - max_records,
                         riot_data.Match.QUEUE_RANKED_5)

        # solo 5v5
        previous_max_records = max_records
        if max_records > 0:
            for match_data in self.matches.find(
                    {"queued": True, "data.queue": riot_data.Match.QUEUE_RANKED_SOLO}).limit(max_records):
                yield match_data
                max_records -= 1
        self.logger.info("Retrieved %d queued %s matches", previous_max_records - max_records,
                         riot_data.Match.QUEUE_RANKED_SOLO)

    def get_players_recrawl(self, max_records):
        """Get an iterable of players to recrawl for match history or other purposes"""
        players = []
        previous_max_records = max_records
        for player_data in self.players.find({"recrawl_at": None}).limit(max_records):
            envelope = Envelope.unwrap(player_data)
            players.append((envelope, riot_data.Summoner(envelope.data)))
            max_records -= 1
        self.logger.info("%d players without recrawl specified", previous_max_records - max_records)

        if max_records > 0:
            previous_max_records = max_records
            for player_data in self.players.find({"recrawl_at": {"$ne": None}}).sort("recrawl_at",
                                                                                     pymongo.ASCENDING).limit(
                    max_records):
                envelope = Envelope.unwrap(player_data)
                players.append((envelope, riot_data.Summoner(envelope.data)))
                max_records -= 1
            self.logger.info("%d players selected from earliest recrawl dates", previous_max_records - max_records)

        return players

    def get_queued_players_stats(self, max_records):
        """Get an iterable of players that need their stats object updated"""
        players = []
        previous_max_records = max_records
        for player_data in self.players.find(Envelope.query_data({"stats": {"$exists": False}})).limit(max_records):
            players.append(riot_data.Summoner(Envelope.unwrap(player_data).data))
            max_records -= 1
        self.logger.info("%d players without stats data", previous_max_records - max_records)

        if max_records > 0:
            previous_max_records = max_records
            # TODO: This is hijacking the "recrawl_at" field to be used for both match history and summoner stats
            for player_data in self.players.find(Envelope.query_data({"stats": {"$exists": True}})).sort("recrawl_at",
                                                                                                         pymongo.ASCENDING).limit(
                    max_records):
                players.append(riot_data.Summoner(Envelope.unwrap(player_data).data))
                max_records -= 1
            self.logger.info("%d players selected with earliest recrawl dates", previous_max_records - max_records)

        return players

    def get_player_ids(self):
        player_ids = set()
        for player_data in self.players.find({}):
            player_ids.add(riot_data.Summoner(Envelope.unwrap(player_data).data).id)

        return player_ids

    def update_player_names(self, players):
        new_count = 0
        updated_count = 0

        for player in players:
            assert isinstance(player, riot_data.Summoner)
            self.logger.debug("Updating %d -> %s", player.id, player.name)

            # TODO: Convert this to a clever upsert.
            if self.players.find(Envelope.query_data({"id": player.id})):
                result = self.players.update(Envelope.query_data({"id": player.id}),
                                             {"$set": {"data.name": player.name}})
                self.logger.debug("Updated player name, result: %s", result)
                updated_count += 1
            else:
                result = self.players.insert_one(Envelope.wrap(player.export(), False))
                self.logger.debug("Inserted player and name, result: %s", result)
                new_count += 1

        return new_count, updated_count

    def update_player_stats(self, player_id, player_stats):
        assert isinstance(player_id, int)
        assert isinstance(player_stats, dict)

        player_stats["updateDate"] = int(time.time() * 1000)
        result = self.players.update(Envelope.query_data({"id": player_id}), {"$set": {"data.stats": player_stats}})
        if result["ok"] != 1 or result["n"] != 1:
            self.logger.error("Bad result in setting player stats: %s", result)
        elif result["nModified"] == 0:
            self.outcomes["ranked stats: identical update"] += 1

    def set_league(self, player_id, league):
        assert isinstance(player_id, int)
        assert isinstance(league, riot_data.LeagueEntry)

        result = self.players.update(Envelope.query_data({"id": player_id}),
                                     {"$set": {"data.league": league.to_mongo()}})
        if result["nModified"] != 1:
            self.logger.error("Updating %d to %s weird result: %s", player_id, league.to_mongo(), result)

    @functools32.lru_cache(500000)
    def get_league(self, player_id):
        assert isinstance(player_id, int)

        player_data = self.players.find_one(Envelope.query_data({"id": player_id}))
        if player_data and "league" in player_data["data"]:
            return riot_data.LeagueEntry.from_mongo(player_data["data"]["league"])

        return None

    def update_player_summary_stats(self, player_id, player_stats):
        assert isinstance(player_id, int)
        assert isinstance(player_stats, dict)

        result = self.players.update(Envelope.query_data({"id": player_id}),
                                     {"$set": {"data.summary_stats": player_stats}})
        if result["ok"] != 1 or result["n"] != 1:
            self.logger.error("Bad result in setting player stats: %s", result)
        elif result["nModified"] == 0:
            self.outcomes["summary stats: identical update"] += 1

    @functools32.lru_cache(5000)
    def get_player_stats(self, player_id, retries_remaining=3):
        assert isinstance(player_id, int)

        try:
            result = self.players.find_one(Envelope.query_data({"id": player_id}))
            return riot_data.PlayerStats(result["data"]["stats"])
        except (TypeError, KeyError):
            return riot_data.PlayerStats.make_blank()
        except pymongo.errors.AutoReconnect:
            if retries_remaining:
                return self.get_player_stats(player_id, retries_remaining - 1)
            else:
                self.logger.error("Failed to reconnect, giving up and setting empty stats")
                return riot_data.PlayerStats.make_blank()

    def update_match_history_refresh(self, player, recrawl_date, last_match_millis):
        result = self.players.update(Envelope.query_data({"id": player.id}), {
            "$set": {"recrawl_at": recrawl_date, "recrawl_begin_time": last_match_millis + 1}})
        self.logger.debug("Updated %d match hist refresh for id %d", result["nModified"], player.id)

    def update_match(self, match):
        """Update a match, overwriting all existing fields"""
        self.matches.update(Envelope.query_data({"matchId": match["matchId"]}), Envelope.wrap(match, False))

    def add_match(self, match):
        """Add a match to the database, will error if duplicate"""
        self.matches.insert_one(Envelope.wrap(match, False))

    def dequeue_match(self, match):
        """Update the match queued status to false"""
        self.matches.update(Envelope.query_data({"matchId": match["matchId"]}), Envelope.set_queued(False))

    def remove_match(self, match_id):
        result = self.matches.delete_one(Envelope.query_data({"matchId": match_id}))
        self.logger.info("Removed %d objects for match id %d", result.deleted_count, match_id)

    def remove_player(self, player):
        assert isinstance(player, riot_data.Summoner)

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
        players_queued, players_total = self.players.find({_ENVELOPE_IS_QUEUED: True}).count(), self.players.find(
            {}).count()
        matches_queued, matches_total = self.matches.find({_ENVELOPE_IS_QUEUED: True}).count(), self.matches.find(
            {}).count()

        return """
        {:,} players queued ({:.1f}% of {:,})
        {:,} matches queued ({:.1f}% of {:,})
        Outcomes: {}
        """.format(players_queued,
                   100. * players_queued / players_total,
                   players_total,
                   matches_queued,
                   100. * matches_queued / matches_total,
                   matches_total,
                   self.outcomes.most_common())

    def compact(self):
        pass

    def get_matches(self, chronological=False):
        c = self.matches.find({}).batch_size(100)
        if chronological:
            self.matches.ensure_index("data.matchCreation")
            c = c.sort("data.matchCreation", pymongo.ASCENDING)

        for match_data in c:
            yield riot_data.Match(match_data["data"])

    def get_num_matches(self):
        return self.matches.find().count()

    def get_match_refs(self):
        c = self.matches.find({}).batch_size(100).sort("data.matchId", pymongo.DESCENDING)

        for match_data in c:
            yield riot_data.MatchReference(match_data["data"])

    def compute_champion_damage_types(self):
        champion_damages = collections.defaultdict(collections.Counter)
        play_rates = collections.Counter()

        for player_stats_data in self.players.find(Envelope.query_data({"stats.champions": {"$exists": True}})):
            player_stats = riot_data.PlayerStats(player_stats_data["data"]["stats"])
            for champion_id, champion_stats in player_stats.champion_stats.iteritems():
                num_played = float(champion_stats["totalSessionsPlayed"])
                play_rates[champion_id] += num_played
                champion_damages[champion_id]["magic"] += champion_stats["totalMagicDamageDealt"]
                champion_damages[champion_id]["physical"] += champion_stats["totalPhysicalDamageDealt"]
                champion_damages[champion_id]["true"] += (champion_stats["totalDamageDealt"] - (
                    champion_stats["totalMagicDamageDealt"] + champion_stats["totalPhysicalDamageDealt"]))

        for champion_id, champion_damage_stats in champion_damages.iteritems():
            for damage_type in champion_damage_stats.iterkeys():
                champion_damage_stats[damage_type] /= play_rates[champion_id]

        return champion_damages

    def aggregate_champion_stats(self):
        """Aggregate all stats across all champions and return a ChampionStats, dict(id -> ChampionStats)"""
        total_data = collections.Counter()
        champion_data = collections.defaultdict(collections.Counter)

        for player_stats_data in self.players.find(Envelope.query_data({"stats.champions": {"$exists": True}})):
            player_stats = riot_data.PlayerStats(player_stats_data["data"]["stats"])

            for champion_id, champion_stats_dict in player_stats.champion_stats.iteritems():
                if champion_id == 0:
                    continue
                total_data.update(champion_stats_dict)
                champion_data[champion_id].update(champion_stats_dict)

        return riot_data.ChampionStats(total_data), {k: riot_data.ChampionStats(v) for k, v in
                                                     champion_data.iteritems()}

    def aggregate_match_stats(self):
        """Get aggregated win rates by champion id and (version, champ id)"""
        games_won = collections.Counter()
        games_played = collections.Counter()

        for match in self.get_matches():
            for player in match.players:
                for conditional_key in [player.champion_id, (match.version, player.champion_id)]:
                    if player.team_id == match.get_winning_team_id():
                        games_won[conditional_key] += 1
                    games_played[conditional_key] += 1

        win_stats = dict()
        for key in games_played.iterkeys():
            win_stats[key] = riot_data.ChampionStats.from_wins_played(games_won[key], games_played[key])

        return win_stats

    def _setup_mongo(self):
        result = self.players.ensure_index("data.id", unique=True)
        self.logger.debug("Player ensure index result: {}".format(result))
        result = self.matches.ensure_index("data.matchId", unique=True)
        self.logger.debug("Match ensure index result: {}".format(result))

    def get_players(self):
        """Get all player objects in database, parsed"""
        for player_data in self.players.find({}).batch_size(500):
            yield riot_data.Summoner(Envelope.unwrap(player_data).data)


class MemoizeCache(ApiCache):
    def __init__(self, config, riot_connection):
        super(MemoizeCache, self).__init__(config, config_section="mongo_local")

        assert isinstance(riot_connection, riot_api.RiotService)
        self.riot_connection = riot_connection
        self.logger = logging.getLogger("MemoizeCache")

        # cached computation like champion damage breakdowns
        self.__compute_mongo_client = pymongo.MongoClient(config.get("mongo_local", "match_uri"))
        self.__compute_mongo_db = self.__compute_mongo_client.get_default_database()
        self.compute_collection = self.__compute_mongo_db["computed_values"]

        self.heartbeat_logger = logging.getLogger("MemoizeCache.heartbeat")
        self.heartbeat_logger.addFilter(utilities.ThrottledFilter(delay_seconds=10))

    def compute_champion_damage_types(self):
        cached_data = self.compute_collection.find_one({"key": "compute_champion_damage_types"})

        if not cached_data:
            self.logger.info("compute_champion_damage_types: recomputing and storing in mongo")
            data = super(MemoizeCache, self).compute_champion_damage_types()
            cached_data = Envelope.wrap({str(k): v for k, v in data.iteritems()}, is_queued=False)
            cached_data["key"] = "compute_champion_damage_types"
            self.compute_collection.insert_one(cached_data)
        else:
            self.logger.info("compute_champion_damage_types: found cached in mongo")
            data = {int(k): v for k, v in cached_data["data"].iteritems()}

        return data

    def aggregate_champion_stats(self):
        # riot_data.ChampionStats(total_data), {k: riot_data.ChampionStats(v) for k, v in champion_data.iteritems()}
        cached_data = self.compute_collection.find_one({"key": "aggregate_champion_stats"})

        if not cached_data:
            self.logger.info("aggregate_champion_stats: recomputing and storing in mongo")
            data = super(MemoizeCache, self).aggregate_champion_stats()
            cached_data = Envelope.wrap(riot_data.ChampionStats.wrap(data), is_queued=False)
            cached_data["key"] = "aggregate_champion_stats"
            self.compute_collection.insert_one(cached_data)
        else:
            self.logger.info("aggregate_champion_stats: found cached in mongo")
            data = riot_data.ChampionStats.unwrap(Envelope.unwrap(cached_data).data)

        return data


    def get_ranked_stats(self, player_id):
        """Look up the player's ranked stats, preferring mongodb cache over Riot API"""
        assert isinstance(player_id, int)

        # cache lookup
        ranked_stats = self.get_player_stats(player_id)
        if ranked_stats:
            return ranked_stats

        # API lookup
        ranked_stats = self.riot_connection.get_summoner_ranked_stats(player_id)
        if ranked_stats:
            self.update_player_stats(player_id, ranked_stats)
            return riot_data.PlayerStats(ranked_stats)

        return None

    def get_match(self, match_id):
        assert isinstance(match_id, int)

        # cache lookup
        match_data = self.matches.find(Envelope.query_data({"matchId": match_id}))
        if match_data:
            return riot_data.Match(match_data)

        # API lookup
        match_data = self.riot_connection.get_match(match_id)
        if match_data:
            match = riot_data.Match(match_data)
            self.update_match(match)
            return match

        return None

    def __load_leagues(self):
        player_ids = list(player.id for player in self.get_queued_players(-1, TYPE_LEAGUE))

        timer = utilities.EstCompletionTimer().start(len(player_ids))

        self.logger.info("Updating league entries for {:,} players".format(len(player_ids)))
        for player_id_chunk in utilities.chunks(player_ids, 10):
            leagues = self.riot_connection.get_leagues(player_id_chunk)

            timer.update("league updated", len(leagues))
            timer.update("league update failed", len(player_id_chunk) - len(leagues))

            for player_id, league_entry in leagues.iteritems():
                self.set_league(int(player_id), league_entry)

            self.heartbeat_logger.info(timer.log_info())

    def __load_matches(self, remote_cache):
        timer = utilities.EstCompletionTimer().start(remote_cache.matches.find({}).count())

        self.logger.info("Loading matches from remote cache into local and fetching match details")
        for i, match_ref in enumerate(remote_cache.get_match_refs()):
            if not match_ref.is_interesting():
                timer.update("skipped not interesting")
                continue

            if self.matches.find(Envelope.query_data({"matchId": match_ref.id})).count() != 0:
                timer.update("already cached")
            else:
                try:
                    match = self.riot_connection.get_match(match_ref.id)

                    try:
                        _filter_match_data(match)
                    except KeyError as e:
                        self.logger.exception("Missing field {}".format(e.message))
                    self.add_match(match)
                    timer.update("updated")
                except requests.exceptions.HTTPError:
                    timer.update("match not found")

            self.heartbeat_logger.info(timer.log_info())

    def __load_players(self, remote_cache):
        """Load players from the remote cache"""
        timer = utilities.EstCompletionTimer().start(remote_cache.players.find({}).count())

        self.logger.info("Loading summoners from remote cache into local")

        for i, player in enumerate(remote_cache.get_players()):
            self.queue_player(player)

            timer.update()
            self.heartbeat_logger.info(timer.log_info())

    def __get_player_ids(self):
        timer = utilities.EstCompletionTimer().start(self.players.find({}).count())

        self.logger.info("Loading list of players from local")

        id_set = set()
        for player in self.get_players():
            id_set.add(player.id)
            timer.update()
            self.heartbeat_logger.info(timer.log_info())

        return id_set

    def __load_ranked_stats(self, stats_expiry_days):
        """Update ranked stats into local cache"""

        # TODO: Some stats aren't updated often so we waste a ton of time crawling them.
        stats_min_date = 1000. * (time.time() - timedelta(stats_expiry_days).total_seconds())
        self.logger.info("Updating ranked stats")

        # complex query because riot sometimes doesn't refresh their stats for some players
        # so if I use just modifyDate then I end up recrawling about 20,000 every single time
        riot_modified_date_query = {"$or": [{"data.stats.modifyDate": {"$lt": stats_min_date}},
                                            {"data.stats.modifyDate": {"$exists": False}}]}
        my_update_query = {"$or": [{"data.stats.updateDate": {"$lt": stats_min_date}},
                                   {"data.stats.updateDate": {"$exists": False}}]}
        query = {"$and": [riot_modified_date_query, my_update_query]}

        cursor = self.players.find(query).batch_size(100)
        timer = utilities.EstCompletionTimer().start(cursor.count())

        for player_data in cursor:
            player_envelope = Envelope.unwrap(player_data)
            player = riot_data.Summoner(player_envelope.data)

            try:
                self.update_player_stats(player.id, self.riot_connection.get_summoner_ranked_stats(player.id))

                if "stats" in player_envelope.data and "modifyDate" in player_envelope.data["stats"]:
                    timer.update("stats fetched, refresh")
                else:
                    timer.update("stats fetched, first time")
            except riot_api.SummonerNotFoundError:
                timer.update("stats not found")
            except requests.exceptions.HTTPError:
                timer.update("http error")

            self.heartbeat_logger.info(timer.log_info())

    def load(self, remote_cache, stats_expiry_days=12):
        """Load player and match data from the remote cache"""
        assert isinstance(remote_cache, ApiCache)

        self.__load_players(remote_cache)
        self.__load_ranked_stats(stats_expiry_days)
        self.__load_leagues()
        self.__load_matches(remote_cache)

    def add_match_players(self):
        """Add any players in the matches that aren't in the summoner database"""
        cached_player_ids = self.__get_player_ids()

        self.logger.info("Finding players that aren't in the database")
        timer = utilities.EstCompletionTimer().start(self.matches.find({}).count() * 10)
        for match in self.get_matches():
            for participant in match.players:
                if participant.id not in cached_player_ids:
                    self.queue_player(riot_data.Summoner.from_fields(participant.id, participant.name))
                    cached_player_ids.add(participant.id)
                    timer.update("added player")
                else:
                    timer.update("player found")
            self.heartbeat_logger.info(timer.log_info())

    def update_players(self):
        """Ensure that we have all summoner information for players in the match database"""
        # self.add_match_players()
        self.__load_ranked_stats(12)
        self.__load_leagues()



def _filter_match_data(data):
    """Remove some fields from a match dict to save space"""
    for team in data["teams"]:
        assert isinstance(team, dict)
        del team["vilemawKills"]
        del team["dominionVictoryScore"]

    for participant_identity in data["participantIdentities"]:
        del participant_identity["player"]["matchHistoryUri"]
        del participant_identity["player"]["profileIcon"]

    for participant in data["participants"]:
        if "runes" in participant:
            del participant["runes"]
        if "masteries" in participant:
            del participant["masteries"]
        del participant["stats"]

        for field in participant["timeline"].keys():
            if field.endswith("Deltas"):
                del participant["timeline"][field]

    return data


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

    # unset player-related fields
    for i in xrange(0, 10):
        for field in "stats runes masteries timeline".split():
            fields.append("data.participants.{}.{}".format(i, field))
        fields.append("data.participantIdentities.{}.player.matchHistoryUri".format(i))
        fields.append("data.participantIdentities.{}.player.profileIcon".format(i))

    # unset team-related fields
    for i in xrange(0, 2):
        fields.append("data.teams.{}.vilemawKills".format(i))
        fields.append("data.teams.{}.dominionVictoryScore".format(i))

    return {k: None for k in fields}


class NoStatsError(KeyError):
    pass