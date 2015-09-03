from __future__ import unicode_literals
import collections
import logging
import time
import urlparse
import requests
import requests.exceptions

import riot_data
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
    def __init__(self, base_url, static_base_url, observer_base_url, api_key, delay_seconds=1.2):
        self.base_url = base_url
        self.static_base_url = static_base_url
        self.observer_base_url = observer_base_url
        self.api_key = api_key
        self.params = {"api_key": self.api_key}
        self.champions = None

        self.most_recent_request = None
        self.delay_seconds = delay_seconds * 1.1
        self.logger = logging.getLogger("RiotService")
        self.heartbeat_logger = logging.getLogger("RiotService.heartbeat")
        self.heartbeat_logger.addFilter(utilities.ThrottledFilter(delay_seconds=5))

        self.num_requests = 0
        self.request_types = collections.Counter()
        self.request_timer = utilities.RequestTimer()

        self.requests_session = requests.Session()

        self.summoner_spells = None

    @staticmethod
    def from_config(config_parser):
        return RiotService(config_parser.get("riot", "base"),
                           config_parser.get("riot", "static_base"),
                           config_parser.get("riot", "observer_base"),
                           config_parser.get("riot", "api_key"),
                           float(config_parser.get("riot", "api_delay_seconds")))

    def request(self, endpoint, base_url=None, tries_left=1, additional_params=None, suppress_codes={}):
        params = _merge_params(self.params, additional_params)

        self.request_timer.start()
        self.throttle()
        full_url = urlparse.urljoin(base_url or self.base_url, endpoint)
        response = self.requests_session.get(full_url, params=params)
        self.num_requests += 1
        self.request_timer.stop()

        # don't give messages for 200, 429, and any special ones that are expected like 404 sometimes
        if response.status_code != HTTP_OK and response.status_code not in suppress_codes and response.status_code != RATE_LIMIT_ERROR:
            self.logger.error("Request %s error code %d", full_url, response.status_code)

        # transient errors like 500 can be tried again
        if response.status_code in HTTP_TRANSIENT_ERRORS and tries_left > 0:
            return self.request(endpoint, base_url, tries_left=tries_left - 1, additional_params=additional_params)

        # increase the delay exponentially for rate limit errors
        for exponential_level in xrange(1, 5):
            if response.status_code != RATE_LIMIT_ERROR:
                break

            self.request_timer.start()
            self.throttle(exponential_level)
            self.logger.warning("Waiting for %.1f seconds", self.scale_delay(exponential_level))
            response = self.requests_session.get(full_url, params=params)
            self.num_requests += 1
            self.request_timer.stop()

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

    def get_summoner_spell_name(self, summoner_spell_id):
        if not self.summoner_spells:
            self.summoner_spells = self.get_summoner_spells()

        try:
            return self.summoner_spells[summoner_spell_id]["name"]
        except KeyError:
            return "Unknown"

    def request_static(self, endpoint, additional_params=None):
        params = _merge_params(self.params, additional_params)

        self.throttle()
        response = requests.get(urlparse.urljoin(self.static_base_url, endpoint), params=params)
        response.raise_for_status()
        data = response.json()
        return data

    def get_champions(self):
        data = self.request_static("v1.2/champion", additional_params={"champData": "info,stats,tags"})
        data = data["data"]

        return {value["id"]: value for value in data.itervalues()}

    def scale_delay(self, delay_level):
        return self.delay_seconds * 10 ** delay_level

    def heartbeat(self):
        request_types = ", ".join("{}: {:,}".format(k, v) for k, v in self.request_types.most_common())
        self.heartbeat_logger.info("Made %d requests at %.1f req/s: %s", self.num_requests, self.request_timer.get_requests_per_second(), request_types)

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
            self.request_types["stats/by-summoner/_/ranked"] += 1
            return self.request("v1.3/stats/by-summoner/{summonerId}/ranked".format(summonerId=summoner_id), suppress_codes={404: True})
        except requests.exceptions.HTTPError as exc:
            if exc.response.status_code == 404:
                raise SummonerNotFoundError()
            else:
                raise exc

    def get_summoner_summary_stats(self, summoner_id):
        try:
            self.request_types["stats/by-summoner/_/summary"] += 1
            return self.request("v1.3/stats/by-summoner/{summonerId}/summary".format(summonerId=summoner_id))
        except requests.exceptions.HTTPError as exc:
            if exc.response.status_code == 404:
                raise SummonerNotFoundError()
            else:
                raise exc

    def get_summoner_by_name(self, name):
        self.request_types["summoner/by-name"] += 1
        return riot_data.Summoner(self.request("v1.4/summoner/by-name/{}".format(name)).values()[0])

    def get_master_plus_solo(self, queue=riot_data.Match.QUEUE_RANKED_SOLO):
        """
        Get summoners in master or challenger. If team queue is specified the Summoners returned
        will be the team ID and names.
        """
        for league in ["challenger", "master"]:
            data = self.request("v2.5/league/{}".format(league), additional_params={"type": queue})

            for entry in data["entries"]:
                yield riot_data.Summoner.from_fields(int(entry["playerOrTeamId"]), entry["playerOrTeamName"])

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

        return [riot_data.Summoner(s) for s in data.itervalues()]

    def get_leagues(self, player_ids):
        raise utilities.DevReminderError("set_league not implemented yet")


    def get_featured_matches(self):
        data = self.request("featured", self.observer_base_url)
        self.request_types["featured"] += 1
        return data["gameList"], data["clientRefreshInterval"]

    def get_match_history(self, summoner_id, recrawl_start_time=None):
        if not summoner_id or not isinstance(summoner_id, int):
            raise InvalidIdError("summoner_id must be a valid int")

        try:
            additional_args = None
            if recrawl_start_time:
                additional_args = {"beginTime": recrawl_start_time}
                self.logger.info("Setting recrawl start time to {}".format(recrawl_start_time))
            data = self.request("v2.2/matchlist/by-summoner/{}".format(summoner_id), additional_params=additional_args)
            self.request_types["matchlist"] += 1

            if data and data["totalGames"] > 0:
                for match in data["matches"]:
                    yield riot_data.MatchReference(match)
        except requests.HTTPError as exc:
            self.logger.exception("Ignoring error in match history")

    def get_summoner_spells(self):
        data = self.request_static("v1.2/summoner-spell")
        data = data["data"]

        return {value["id"]: value for value in data.itervalues()}



class InvalidIdError(ValueError):
    pass

class SummonerNotFoundError(ValueError):
    pass