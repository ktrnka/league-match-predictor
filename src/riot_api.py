from __future__ import unicode_literals
import logging
import pprint
import time
import urlparse
import math

import requests
from riot_data import Summoner

# error codes that happen cause a server is temporarily down/etc
HTTP_OK = 200
HTTP_TRANSIENT_ERRORS = {500, 503}

# the same errors will happen again if tried
HTTP_HARD_ERRORS = {400, 401, 404, 422}

# hit rate limit
RATE_LIMIT_ERROR = 427

class RiotService(object):
    def __init__(self, base_url, static_base_url, observer_base_url, api_key):
        self.base_url = base_url
        self.static_base_url = static_base_url
        self.observer_base_url = observer_base_url
        self.api_key = api_key
        self.params = {"api_key": self.api_key}
        self.champions = None

        self.most_recent_request = None
        self.delay_seconds = 1.4
        self.logger = logging.getLogger("RiotService")

    @staticmethod
    def from_config(config_parser):
        return RiotService(config_parser.get("riot", "base"), config_parser.get("riot", "static_base"),
                           config_parser.get("riot", "observer_base"), config_parser.get("riot", "api_key"))

    def request(self, endpoint, base_url=None, tries_left=1):
        self.throttle()
        full_url = urlparse.urljoin(base_url or self.base_url, endpoint)
        response = requests.get(full_url, params=self.params)

        if response.status_code != HTTP_OK:
            self.logger.error("Request %s error code %d", full_url, response.status_code)

        if response.status_code in HTTP_TRANSIENT_ERRORS and tries_left > 0:
            return self.request(endpoint, base_url, tries_left=tries_left-1)

        for exponential_level in xrange(1, 4):
            if response.status_code == HTTP_OK:
                break

            self.throttle(exponential_level)
            self.logger.warning("Waiting for %.1f seconds", self.scale_delay(exponential_level))
            response = requests.get(full_url, params=self.params)

        response.raise_for_status()
        data = response.json()
        return data

    def get_match(self, match_id):
        return self.request("v2.2/match/{}".format(match_id))

    def get_champion_info(self, champion_id):
        if not self.champions:
            self.champions = self.get_champions()

        return self.champions[champion_id]

    def request_static(self, endpoint):
        self.throttle()
        response = requests.get(urlparse.urljoin(self.static_base_url, endpoint), params=self.params)
        response.raise_for_status()
        data = response.json()
        return data

    def get_champions(self):
        data = self.request_static("v1.2/champion")
        data = data["data"]

        return {value["id"]: value for value in data.itervalues()}

    def scale_delay(self, delay_level):
        return self.delay_seconds * 10 ** delay_level

    def throttle(self, delay_level=0):
        scaled_delay_seconds = self.scale_delay(delay_level)
        if self.most_recent_request and time.clock() - self.most_recent_request < scaled_delay_seconds:
            self.logger.debug("Sleeping for %.1f seconds", scaled_delay_seconds)
            time.sleep(scaled_delay_seconds)
        self.most_recent_request = time.clock()

    def get_team_name(self, team_id):
        if team_id == 100:
            return "Blue"
        elif team_id == 200:
            return "Red"
        else:
            return "Unknown, " + team_id

    def get_summoner_ranked_stats(self, summoner_id):
        return self.request("v1.3/stats/by-summoner/{summonerId}/ranked".format(summonerId=summoner_id))

    def get_summoner_id(self, summoner_name):
        return self.request("v1.4/summoner/by-name/{}".format(summoner_name)).values()[0]["id"]

    def get_summoner(self, name=None, id=None):
        if name:
            return Summoner(self.request("v1.4/summoner/by-name/{}".format(name)).values()[0])
        else:
            return None

    def get_summoners(self, ids=None, names=None):
        assert ids or names

        if ids:
            if None in ids:
                self.logger.warn("Empty IDs in list")
                ids = [i for i in ids if i]
                if not i:
                    return []

            self.logger.info("Requesting summoners {}".format(ids))
            data = self.request("v1.4/summoner/{}".format(",".join([str(x) for x in ids])))
        else:
            data = self.request("v1.4/summoner/by-name/{}".format(",".join(names)))

        return [Summoner(s) for s in data.itervalues()]

    def get_featured_matches(self):
        data = self.request("featured", self.observer_base_url)
        return data["gameList"], data["clientRefreshInterval"]

    def get_match_history(self, summoner_id):
        if not summoner_id or not isinstance(summoner_id, int):
            raise ValueError("summoner_id must be a valid int")

        data = self.request("v2.2/matchhistory/{}".format(summoner_id))

        if data:
            for match in data["matches"]:
                yield Match(match)


class Match(object):
    def __init__(self, data):
        self.id = data["matchId"]
        self.mode = data["matchMode"]
        self.type = data["matchType"]
        self.creation_time = data["matchCreation"]
        self.duration = data["matchDuration"]
        self.queue_type = data["queueType"]
        self.version = data["matchVersion"]

        self.players = list(FeaturedParticipant.parse_participants(data["participants"], data["participantIdentities"]))

class FeaturedParticipant:
    def __init__(self, team_id, spells, champion_id, name, id=None):
        assert len(spells) == 2

        self.teamId = team_id
        self.spells = spells
        self.championId = champion_id
        self.name = name
        self.id = None

    @staticmethod
    def from_joined(data):
        return FeaturedParticipant(data["teamId"], [data["spell1Id"], data["spell2Id"]], data["championId"], data["summonerName"])

    @staticmethod
    def parse_participants(participants, participant_identities):
        for player, identity in zip(participants, participant_identities):
            yield FeaturedParticipant.from_split(player, identity)

    @staticmethod
    def from_split(player, identity):
        return FeaturedParticipant(player["teamId"], [player["spell1Id"], player["spell2Id"]], player["championId"], identity["player"]["summonerName"], id=identity["player"]["summonerId"])



