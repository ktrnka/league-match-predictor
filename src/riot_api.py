from __future__ import unicode_literals
import collections
import logging
import time
import urlparse

import requests
import requests.exceptions
from riot_data import Summoner, Match
import utilities

# error codes that happen cause a server is temporarily down/etc

HTTP_OK = 200
HTTP_TRANSIENT_ERRORS = {500, 503}

# the same errors will happen again if tried
HTTP_HARD_ERRORS = {400, 401, 404, 422}

# hit rate limit
RATE_LIMIT_ERROR = 429


def _merge_params(params, additional_params):
    if not additional_params:
        return params

    params = params.copy()
    params.update(additional_params)
    return params


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
        self.heartbeat_logger = logging.getLogger("RiotService.heartbeat")
        self.heartbeat_logger.addFilter(utilities.ThrottledFilter())

        self.num_requests = 0
        self.request_types = collections.Counter()

    @staticmethod
    def from_config(config_parser):
        return RiotService(config_parser.get("riot", "base"), config_parser.get("riot", "static_base"),
                           config_parser.get("riot", "observer_base"), config_parser.get("riot", "api_key"))

    def request(self, endpoint, base_url=None, tries_left=1, additional_params=None):
        params = _merge_params(self.params, additional_params)
        self.throttle()
        full_url = urlparse.urljoin(base_url or self.base_url, endpoint)
        response = requests.get(full_url, params=params)
        self.num_requests += 1

        if response.status_code != HTTP_OK:
            self.logger.error("Request %s error code %d", full_url, response.status_code)

        if response.status_code in HTTP_TRANSIENT_ERRORS and tries_left > 0:
            return self.request(endpoint, base_url, tries_left=tries_left - 1, additional_params=additional_params)

        for exponential_level in xrange(1, 4):
            if response.status_code != RATE_LIMIT_ERROR:
                break

            self.throttle(exponential_level)
            self.logger.warning("Waiting for %.1f seconds", self.scale_delay(exponential_level))
            response = requests.get(full_url, params=params)
            self.num_requests += 1

        response.raise_for_status()
        data = response.json()

        self.heartbeat()
        return data

    def get_match(self, match_id):
        self.request_types["match"] += 1
        return self.request("v2.2/match/{}".format(match_id))

    def get_champion_info(self, champion_id):
        if not self.champions:
            self.champions = self.get_champions()

        return self.champions[champion_id]

    def get_champion_name(self, champion_id):
        assert isinstance(champion_id, int)

        return self.get_champion_info(champion_id)["name"]

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

    def heartbeat(self):
        request_types = ", ".join("{}: {:,}".format(k, v) for k, v in self.request_types.most_common())
        self.heartbeat_logger.info("Made %d requests from the following high-level types: %s", self.num_requests, request_types)

    def throttle(self, delay_level=0):
        scaled_delay_seconds = self.scale_delay(delay_level)
        if self.most_recent_request and time.clock() - self.most_recent_request < scaled_delay_seconds:
            self.logger.debug("Sleeping for %.1f seconds", scaled_delay_seconds)
            time.sleep(scaled_delay_seconds)
        self.most_recent_request = time.clock()

    @staticmethod
    def get_team_name(team_id):
        if team_id == 100:
            return "Blue"
        elif team_id == 200:
            return "Red"
        else:
            return "Unknown, " + team_id

    def get_summoner_ranked_stats(self, summoner_id):
        try:
            self.request_types["stats/by-summoner"] += 1
            return self.request("v1.3/stats/by-summoner/{summonerId}/ranked".format(summonerId=summoner_id))
        except requests.exceptions.HTTPError as exc:
            if exc.response.status_code == 404:
                raise SummonerNotFoundError()
            else:
                raise exc

    def get_summoner_by_name(self, name):
        self.request_types["summoner/by-name"] += 1
        return Summoner(self.request("v1.4/summoner/by-name/{}".format(name)).values()[0])

    @staticmethod
    def _filter_ids(id_list):
        return [i for i in id_list if i]

    def get_summoners(self, ids=None, names=None):
        """Batch requesting info by id or name"""
        assert ids or names

        if ids:
            ids = self._filter_ids(ids)
            assert ids

            self.logger.debug("Requesting summoners {}".format(ids))
            data = self.request("v1.4/summoner/{}".format(",".join([str(x) for x in ids])))
            self.request_types["summoner"] += 1
        else:
            names = self._filter_ids(names)
            assert names

            self.logger.debug("Requesting summoners {}".format(names))
            data = self.request("v1.4/summoner/by-name/{}".format(",".join(names)))
            self.request_types["summoner/by-name"] += 1

        return [Summoner(s) for s in data.itervalues()]

    def get_featured_matches(self):
        data = self.request("featured", self.observer_base_url)
        self.request_types["featured"] += 1
        return data["gameList"], data["clientRefreshInterval"]

    def get_match_history(self, summoner_id):
        if not summoner_id or not isinstance(summoner_id, int):
            raise InvalidIdError("summoner_id must be a valid int")

        data = self.request("v2.2/matchhistory/{}".format(summoner_id), additional_params={"endIndex": 15})
        self.request_types["matchhistory"] += 1

        if data:
            for match in data["matches"]:
                yield Match(match)

class InvalidIdError(ValueError):
    pass

class SummonerNotFoundError(ValueError):
    pass