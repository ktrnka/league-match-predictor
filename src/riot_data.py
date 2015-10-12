from __future__ import unicode_literals
import logging
import sys
import argparse
import datetime
import collections
import unittest
import functools32
import utilities

def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    sys.exit(main())

RED_TEAM = 200

BLUE_TEAM = 100

_STANDARD_ROLES = {"BOTTOM DUO_SUPPORT", "BOTTOM DUO_CARRY", "JUNGLE", "MIDDLE", "TOP"}


class Tier(object):
    _tiers = ["CHALLENGER", "MASTER", "DIAMOND", "PLATINUM", "GOLD", "SILVER", "UNRANKED", "BRONZE"]
    _indexes = {tier: i for i, tier in enumerate(_tiers)}

    @staticmethod
    def make_sortable_key(tier_label):
        return Tier._indexes.get(tier_label)

    @staticmethod
    def mean_level(tiers):
        return sum(Tier._indexes[tier] for tier in tiers) / len(tiers)


class Champion(object):
    known_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 48, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 67, 68, 69, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 89, 90, 91, 92, 96, 98, 99, 101, 102, 103, 104, 105, 106, 107, 110, 111, 112, 113, 114, 115, 117, 119, 120, 121, 122, 126, 127, 131, 133, 134, 143, 150, 154, 157, 161, 201, 222, 223, 236, 238, 245, 254, 266, 267, 268, 412, 421, 429, 432]


class Summoner:
    def __init__(self, data):
        self.id = data["id"]

        # unprocessed data may not have summoner names
        self.name = data.get("name", None)

    def export(self):
        return {"id": int(self.id), "name": self.name}

    def __str__(self):
        return "{}__{}".format(self.name, self.id)

    @staticmethod
    def from_fields(summoner_id, name):
        return Summoner({"id": summoner_id, "name": name})


class Participant(object):
    _SOLO_ROLES = {"SOLO", "NONE"}

    def __init__(self, team_id, spells, champion_id, name, id=None, tier=None, participant_id=None, role=None):
        assert len(spells) == 2

        self.team_id = team_id
        self.spells = spells
        self.champion_id = champion_id
        self.name = name

        self.id = id
        self.tier = tier
        self.participant_id = participant_id

        self.role = role

    def to_summoner(self):
        return Summoner.from_fields(self.id, self.name)

    @staticmethod
    def from_joined(data):
        return Participant(data["teamId"], [data["spell1Id"], data["spell2Id"]], data["championId"], data["summonerName"])

    def get_champion_spells_key(self):
        """Tuple of champion ID and the two summoner spells in numeric order"""
        spells = sorted(self.spells)
        return self.champion_id, spells[0], spells[1]

    @staticmethod
    def parse_participants(participants, participant_identities):
        if participant_identities:
            for player, identity in zip(participants, participant_identities):
                yield Participant.from_split(player, identity)
        else:
            for player in participants:
                yield Participant.from_joined(player)

    @staticmethod
    def from_split(player, identity):
        role = None
        if "timeline" in player:
            role = player["timeline"]["lane"]
            if player["timeline"]["role"] not in Participant._SOLO_ROLES:
                role = "{} {}".format(role, player["timeline"]["role"])
        return Participant(player["teamId"],
                           [player["spell1Id"], player["spell2Id"]],
                           player["championId"],
                           identity["player"]["summonerName"],
                           id=identity["player"]["summonerId"],
                           tier=player["highestAchievedSeasonTier"],
                           participant_id=player["participantId"],
                           role=role)


class Queue(object):
    id_to_name = {
        4: "RANKED_SOLO_5x5",
        42: "RANKED_TEAM_5x5"
    }

    interesting_queues = set(id_to_name.itervalues())

    @staticmethod
    def to_name(queue_id):
        assert isinstance(queue_id, int)

        return Queue.id_to_name.get(queue_id, unicode(queue_id))

    @staticmethod
    def is_interesting(queue_name):
        assert isinstance(queue_name, basestring)
        return queue_name in Queue.interesting_queues


class Season(object):
    @staticmethod
    def is_interesting(season_name):
        assert isinstance(season_name, basestring)
        return season_name == "SEASON2015"


class MatchBase(object):
    def __init__(self, match_id, timestamp):
        self.id = match_id
        self.timestamp = timestamp

    def get_creation_datetime(self):
        # The creation time from API is milliseconds since the epoch not seconds
        return datetime.datetime.fromtimestamp(self.timestamp / 1000.)


class Match(MatchBase):
    QUEUE_RANKED_SOLO = "RANKED_SOLO_5x5"
    QUEUE_RANKED_5 = "RANKED_TEAM_5x5"

    def __init__(self, data):
        super(Match, self).__init__(data["matchId"], data["matchCreation"])

        self.mode = data["matchMode"]
        self.type = data["matchType"]
        self.duration = data["matchDuration"]
        self.queue_type = data["queueType"]
        self.version = data["matchVersion"]
        self.players = list(Participant.parse_participants(data["participants"], data["participantIdentities"]))
        self.full_data = data

    def get_winning_team_id(self):
        for team in self.full_data["teams"]:
            if team["winner"]:
                return team["teamId"]

    def get_picks(self):
        """
        Get the champions picked for each side
        :return: Mapping of team IDs to of sets of champion IDs
        """
        picks = collections.defaultdict(set)
        for player in self.full_data["participants"]:
            picks[player["teamId"]].add(player["championId"])

        return picks

    def get_picks_role(self):
        """
        Get array of Participant for each side in order of participant ID.
        TODO: This should eventually be sorted by TOP, MID, etc.
        :return: dict of team ids mapped to list of champion ids
        """
        picks = collections.defaultdict(list)
        for player in self.players:
            picks[player.team_id].append(player)

        for team_id in picks.keys():
            picks[team_id] = sorted(picks[team_id], key=lambda p: p.id)

        return picks

    def get_roles(self):
        picks = collections.defaultdict(dict)
        for player in self.players:
            picks[player.team_id][player.role] = player.champion_id

        return picks

    def get_bans(self):
        """Get a set of bans"""
        bans = set()
        for team in self.full_data["teams"].itervalues():
            for ban in team["bans"]:
                bans.add(ban["championId"])

        return bans

    def get_team_tiers_numeric(self):
        team_tiers = collections.defaultdict(set)
        for player in self.full_data["participants"]:
            team_tiers[player["teamId"]].add(player["highestAchievedSeasonTier"])

        return {team_id: Tier.mean_level(tiers) for team_id, tiers in team_tiers.items()}

    def get_average_tier(self):
        """Get the most common tier among the participants. Doesn't do any fancy averaging. Returns a string."""
        tiers = [player["highestAchievedSeasonTier"] for player in self.full_data["participants"]]
        tier_counts = collections.Counter(tiers)

        return tier_counts.most_common(1)[0][0]

    def export(self):
        return self.full_data

    @staticmethod
    def from_featured(data):
        """Parse a partial Match object from a featured match"""
        assert isinstance(data, dict)

        wrapped_data = dict()
        wrapped_data["matchId"] = data["gameId"]
        wrapped_data["matchMode"] = data["gameMode"]
        wrapped_data["matchType"] = data["gameType"]
        wrapped_data["matchCreation"] = data["gameStartTime"]
        wrapped_data["matchDuration"] = data["gameLength"]  # note that this is partial only
        wrapped_data["queueType"] = Queue.to_name(data["gameQueueConfigId"])
        wrapped_data["matchVersion"] = -1

        # note that there's no identity information in featured mode
        wrapped_data["participants"] = data["participants"]
        wrapped_data["participantIdentities"] = None

        wrapped_data["original_data"] = data
        return Match(wrapped_data)


class MatchReference(MatchBase):
    """Reference to a match without much detailed information"""

    def __init__(self, data):
        super(MatchReference, self).__init__(data["matchId"], data["timestamp"])

        self.full_data = data
        self.platform_id = data["platformId"]
        self.queue = data["queue"]
        self.season = data["season"]

    def is_interesting(self):
        return Season.is_interesting(self.season) and Queue.is_interesting(self.queue) and self.platform_id == "NA1"

    def get_creation_datetime(self):
        return datetime.datetime.fromtimestamp(self.timestamp / 1000.)

    def export(self):
        return self.full_data


def _merge_stats(champion_datas):
    totals = collections.Counter()

    for champion_data in champion_datas:
        totals.update(champion_data)

    return ChampionStats(totals)


class PlayerStats(object):
    def __init__(self, data):
        self.summoner_id = data["summonerId"]
        self.modify_date = data["modifyDate"]
        self.champion_stats = {record["id"]: record["stats"] for record in data["champions"] if record["id"] != 0}
        self.totals = _merge_stats(self.champion_stats.itervalues())

        self.winning_streak_games = 0
        self.losing_streak_games = 0

    @staticmethod
    def make_blank():
        return PlayerStats({"summonerId": -1, "modifyDate": 0, "champions": []})

    @staticmethod
    def from_id(summoner_id):
        return PlayerStats({"summonerId": summoner_id, "modifyDate": 0, "champions": []})

    def get_champion(self, champion_id):
        return self.champion_stats[champion_id]

    def get_win_rate(self, champion_id, remove=False, won=False):
        try:
            stats = self.get_champion(champion_id)
            num_played = stats["totalSessionsPlayed"]
            num_won = stats["totalSessionsWon"]
            if remove:
                num_played -= 1
                if won:
                    num_won -= 1

            return num_won / float(num_played)
        except (KeyError, ZeroDivisionError):
            return 0.5

    def get_games_played(self, champion_id, remove=False):
        try:
            stats = self.get_champion(champion_id)
            played = stats["totalSessionsPlayed"]
            if remove:
                played -= 1

            return played
        except KeyError:
            return 0

    def get_first_blood_rate(self, champion_id):
        # TODO: This may be leaking something correlated with the outcome.
        try:
            stats = self.get_champion(champion_id)
        except KeyError:
            return 0.
        return stats["totalFirstBlood"] / float(stats["totalSessionsPlayed"])

    def get_champion_stats(self, champion_id):
        try:
            return ChampionStats(self.champion_stats[champion_id])
        except KeyError:
            return EMPTY_CHAMPION_STATS


class ChampionStats(object):
    def __init__(self, data):
        self.played = data["totalSessionsPlayed"]
        self.won = data["totalSessionsWon"]

        self.magic_damage = data["totalMagicDamageDealt"]
        self.physical_damage = data["totalPhysicalDamageDealt"]
        self.total_damage = data["totalDamageDealt"]

    def to_mongo(self):
        return {
            "totalSessionsPlayed": self.played,
            "totalSessionsWon": self.won,
            "totalMagicDamageDealt": self.magic_damage,
            "totalPhysicalDamageDealt": self.physical_damage,
            "totalDamageDealt": self.total_damage
        }

    @staticmethod
    def wrap(data):
        if isinstance(data, dict):
            return {str(k): ChampionStats.wrap(v) for k, v in data.iteritems()}
        elif isinstance(data, ChampionStats):
            return data.to_mongo()
        elif isinstance(data, collections.Sequence):
            return [ChampionStats.wrap(v) for v in data]
        else:
            raise ValueError("Trying to wrap unknown type: {}".format(type(data)))

    @staticmethod
    def unwrap(data):
        if isinstance(data, collections.Sequence):
            return [ChampionStats.unwrap(v) for v in data]
        elif isinstance(data, dict):
            if "totalSessionsPlayed" in data:
                return ChampionStats(data)
            else:
                return {int(k): ChampionStats.unwrap(v) for k, v in data.iteritems()}
        else:
            return data

    @staticmethod
    def from_wins_played(num_wins, num_played):
        data = collections.Counter()
        data["totalSessionsPlayed"] = num_played
        data["totalSessionsWon"] = num_wins
        return ChampionStats(data)

    def get_win_rate(self, remove_games=0, remove_wins=0, remove_stats=None):

        if remove_stats:
            assert isinstance(remove_stats, ChampionStats)
            remove_games += remove_stats.played
            remove_wins += remove_stats.won
        assert remove_wins <= remove_games

        # if any stat is inconsistent then the game(s) being removed can be ignored
        if self.played < remove_games or self.won < remove_wins or (self.played - remove_games) < (self.won - remove_wins):
            remove_games = 0
            remove_wins = 0

        if self.played - remove_games <= 0:
            return 0.5

        win_rate = (self.won - remove_wins) / float(self.played - remove_games)
        assert 0 <= win_rate <= 1
        return win_rate

    def get_played(self, remove_games=0, remove_stats=None):
        if self.played == 0:
            return 0

        if remove_stats:
            assert isinstance(remove_stats, ChampionStats)
            remove_games += remove_stats.played

        # if any stat is inconsistent then the game(s) being removed can be ignored
        if self.played < remove_games:
            remove_games = 0

        return self.played - remove_games


class LeagueEntry(object):
    """Represents someone's league entry"""
    __DIVISION_ADDS = {"I": 400, "II": 300, "III": 200, "IV": 100, "V": 0}
    __TIER_ADDS = {"BRONZE": 0, "SILVER": 500, "GOLD": 1000, "PLATINUM": 1500, "DIAMOND": 2000, "MASTER": 2500, "CHALLENGER": 2500}
    __SPECIAL_TIERS = {"MASTER", "CHALLENGER"}


    def __init__(self, queue, tier, division, points):
        self.tier = tier
        self.queue = queue
        self.division = division
        self.points = points

        if self.tier in self.__SPECIAL_TIERS and division != "I":
            raise ValueError("Only division I is allowed for tiers {}, found {}".format(" ".join(self.__SPECIAL_TIERS), self))

    def get_merged_points(self):
        if self.tier == "UNKNOWN":
            return -1

        points = self.points + self.__TIER_ADDS[self.tier]
        if self.tier not in self.__SPECIAL_TIERS:
            points += self.__DIVISION_ADDS[self.division]
        return points

    def to_mongo(self):
        return {
            "tier": self.tier,
            "queue": self.queue,
            "division": self.division,
            "points": self.points
        }

    def is_accurate(self):
        if self.division == "V" and self.points == 0:
            return False
        if self.tier in self.__SPECIAL_TIERS and self.points == 0:
            return False
        return True

    @staticmethod
    def average(leagues):
        # filter failed lookups
        leagues = [league for league in leagues if league]

        # filter division 5 0 LP
        leagues = [league for league in leagues if league.is_accurate()]

        if not leagues:
            return LeagueEntry("UNKNOWN", "UNKNOWN", "UNKNOWN", 0)

        average_points = sum(league.get_merged_points() for league in leagues) / float(len(leagues))
        league = LeagueEntry.from_points(average_points)
        return league

    @staticmethod
    def from_mongo(data):
        return LeagueEntry(data["queue"], data["tier"], data["division"], data["points"])

    @staticmethod
    def from_response(league_data, target_queue=Match.QUEUE_RANKED_SOLO):
        """Extract a dict of id -> LeagueEntry for the target queue"""
        entries = {}
        for player_team_id, leagues in league_data.iteritems():
            for league in leagues:
                if league["queue"] == target_queue:
                    entry = LeagueEntry.__find_entry(league["entries"], player_team_id)
                    try:
                        entries[player_team_id] = LeagueEntry(league["queue"], league["tier"], entry["division"], entry["leaguePoints"])
                    except KeyError as e:
                        logging.getLogger(__name__).error("Missing {} in data: {}".format(e.message, league))

        return entries

    @staticmethod
    def __find_entry(entries, player_team_id):
        for league_entry in entries:
            # the league entries have their keys as strings
            if league_entry["playerOrTeamId"] == str(player_team_id):
                return league_entry

    @staticmethod
    def from_points(average_points):
        tiers_scored = [(tier, average_points - tier_min_points) for tier, tier_min_points in LeagueEntry.__TIER_ADDS.iteritems()]

        # don't match against challenger
        tiers_scored = [pair for pair in tiers_scored if pair[1] >= 0 and pair[0] != "CHALLENGER"]
        tier, league_points = min(tiers_scored, key=lambda pair: pair[1])

        if tier in LeagueEntry.__SPECIAL_TIERS:
            division = "I"

            # hacky way to say it's a challenger game
            if league_points > 400:
                tier = "CHALLENGER"
        else:
            divs_scored = [(division, league_points - div_min_points) for division, div_min_points in LeagueEntry.__DIVISION_ADDS.iteritems()]
            divs_scored = [pair for pair in divs_scored if pair[1] >= 0]
            division, league_points = min(divs_scored, key=lambda pair: pair[1])


        league = LeagueEntry(None, tier, division, league_points)
        return league

    def __str__(self):
        return "{} {}, {} LP".format(self.tier, self.division, self.points)

    def get_tier_division(self):
        return "{} {}".format(self.tier, self.division)

    def __repr__(self):
        return "{} {}, {} LP [{}]".format(self.tier, self.division, self.points, self.get_merged_points())


EMPTY_CHAMPION_STATS = ChampionStats(collections.Counter())


class LeagueTests(unittest.TestCase):
    def test_interpret(self):
        min_points = LeagueEntry(None, "BRONZE", "V", 0)
        self.assertEqual(0, min_points.get_merged_points())

        normal_points = LeagueEntry(None, "GOLD", "II", 50)
        self.assertEqual(1350, normal_points.get_merged_points())

        master_points = LeagueEntry(None, "MASTER", "I", 50)
        self.assertEqual(2550, master_points.get_merged_points())

        self.assertRaises(ValueError, LeagueEntry, None, "MASTER", "IV", 50)

        challenger_points = LeagueEntry(None, "CHALLENGER", "I", 50)
        self.assertEqual(master_points.get_merged_points(), challenger_points.get_merged_points())

    def test_average(self):
        gold5 = LeagueEntry(None, "GOLD", "V", 0)
        gold2 = LeagueEntry(None, "GOLD", "II", 12)

        single_average = LeagueEntry.average([gold2])
        self.assertEqual(gold2.get_merged_points(), single_average.get_merged_points())

        # GOLD V 0 LP means nothing
        dual_average = LeagueEntry.average([gold2, gold5])
        self.assertEqual(gold2.get_merged_points(), dual_average.get_merged_points())

        # MASTER 1 0 LP means nothing
        master_min = LeagueEntry(None, "MASTER", "I", 0)
        dual_average = LeagueEntry.average([gold2, master_min])
        self.assertEqual(gold2.get_merged_points(), dual_average.get_merged_points())


class RoleStats(object):
    """Track win rates by champion and lane, help to normalize the lanes and roles"""
    def __init__(self):
        self.play_counts = collections.defaultdict(collections.Counter)
        self.win_counts = collections.defaultdict(collections.Counter)

        self.matchup_play_counts = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
        self.matchup_win_counts = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))

        self.num_games = 0
        self.role_matches = collections.Counter()
        self.role_matchups = collections.Counter()

    def add(self, champion_id, role, is_win):
        self.play_counts[champion_id][role] += 1
        if is_win:
            self.win_counts[champion_id][role] += 1

    def get_role_champions(self, match):
        """Mapping of coerced roles to team id to champions"""
        roles = collections.defaultdict(collections.defaultdict)
        for team_id, role_map in match.get_roles().iteritems():
            for role, champion_id in role_map.iteritems():
                # recompute every time until the stats are saturated then use the memoized version
                if self.num_games > 10000:
                    role = self.coerce_role(champion_id, role)
                else:
                    role = coerce_standard_lane(self.play_counts, champion_id, role)
                roles[role][team_id] = champion_id
        return roles

    def add_match(self, match):
        assert isinstance(match, Match)

        winner = match.get_winning_team_id()

        for team_id, role_map in match.get_roles().iteritems():
            for role, champion_id in role_map.iteritems():
                self.add(champion_id, role, team_id == winner)

        # add the lane matchup
        roles = self.get_role_champions(match)

        matchups_found = [len(roles.get(role, [])) == 2 for role in _STANDARD_ROLES]
        if all(matchups_found):
            self.role_matches[True] += 1
        else:
            self.role_matches[False] += 1

        self.role_matchups[True] += sum(matchups_found)
        self.role_matchups[False] += 5 - sum(matchups_found)

        teams = [BLUE_TEAM, RED_TEAM]
        for role, champion_map in roles.iteritems():
            if not all(team in champion_map for team in teams):
                continue

            champions = [champion_map[team] for team in teams]
            winners = [int(winner == team) for team in teams]

            assert sum(winners) == 1

            # increment blue side
            self.matchup_play_counts[champions[0]][role][champions[1]] += 1
            self.matchup_win_counts[champions[0]][role][champions[1]] += winners[0]

            # increment red side
            self.matchup_play_counts[champions[1]][role][champions[0]] += 1
            self.matchup_win_counts[champions[1]][role][champions[0]] += winners[1]

        self.num_games += 1

    def get_stats_by_role(self, match):
        assert isinstance(match, Match)

        roles = self.get_role_champions(match)
        return {self.get_role_blue_win_rate(role, roles) for role in _STANDARD_ROLES}

    @functools32.lru_cache(5000)
    def coerce_role(self, champion_id, role):
        return coerce_standard_lane(self.play_counts, champion_id, role)

    def coerce_roles(self):
        self.play_counts, self.win_counts = coerce_standard_lanes(self.play_counts, self.win_counts)

    def print_matchups(self, champion_names):
        for champion_a in sorted(self.matchup_play_counts.iterkeys()):
            for role, _ in utilities.most_common_percent(self.play_counts[champion_a], 0.9):
                role_played = self.play_counts[champion_a][role]
                role_won = self.win_counts[champion_a][role]

                independent_prob = role_won / float(role_played)
                independent_std = utilities.binomial_stddev(independent_prob, role_played)

                print "{} {} [{:.1f}% +/- {:.1f}%]".format(champion_names[champion_a], role, 100. * independent_prob, 100. * independent_std)

                for champion_b, num_played in utilities.most_common_percent(self.matchup_play_counts[champion_a][role], 0.9):
                    num_wins = self.matchup_win_counts[champion_a][role][champion_b]
                    p = num_wins / float(num_played)
                    std = utilities.binomial_stddev(p, num_played)

                    other_ind_played = self.play_counts[champion_b][role]
                    other_ind_won = self.win_counts[champion_b][role]

                    other_ind_prob = other_ind_won / float(other_ind_played)

                    avg_prob = (independent_prob + 1. - other_ind_prob) / 2

                    print "\tvs {:20s}: {:.1f}% +/- {:.1f}% in {:6,} games. z={:+3f}. IndP={:.1f}%".format(champion_names[champion_b],
                                                                         100. * p,
                                                                         100. * std,
                                                                         num_played,
                                                                         (p - independent_prob) / std,
                                                                         100. * avg_prob)

    def print_roles(self, champion_names):
        print "What percent of games match the standard 5 roles? {:.1f}%".format(100. * self.role_matches[True] / sum(self.role_matches.values()))
        print "What percent of roles have a matchup? {:.1f}%".format(100. * self.role_matchups[True] / sum(self.role_matchups.values()))
        print "Champion stats by role"
        for champion_id in sorted(self.play_counts.iterkeys()):
            print "{} [{}]".format(champion_names[champion_id], champion_id)

            for role, count in utilities.most_common_percent(self.play_counts[champion_id], 0.9):
                print "\t{:20s}: {:.1f}% win rate out of {:,} games played".format(role,
                    100. * self.win_counts[champion_id][role] / self.play_counts[champion_id][role], self.play_counts[champion_id][role])

    def get_role_blue_win_rate(self, role, roles, default_win_rate=50.5):
        assert isinstance(role, basestring)
        assert isinstance(roles, dict)

        # if we don't actually have both of them, return a default
        if len(roles[role]) != 2:
            # TODO: If it's just 1 then we could get a better estimate from just that one.
            return default_win_rate

        champion_blue = roles[role][BLUE_TEAM]
        champion_red = roles[role][RED_TEAM]

        matchup_won = self.matchup_win_counts[champion_blue][role][champion_red]
        matchup_played = self.matchup_play_counts[champion_blue][role][champion_red]

        blue_won = self.win_counts[champion_blue][role]
        blue_played = self.win_counts[champion_blue][role]

        red_won = self.win_counts[champion_red][role]
        red_played = self.play_counts[champion_red][role]

        # just add the counts
        independent_won = blue_won + red_played - red_won
        independent_played = blue_played + red_played

        # smooth even the backoff win rate
        independent_win_rate = utilities.smooth_win_rate(independent_won, independent_played, default_win_rate, crossover=50)

        return utilities.smooth_win_rate(matchup_won, matchup_played, independent_win_rate, crossover=50)


def coerce_standard_lane(played_role_counts, champion_id, role):
    if role in _STANDARD_ROLES:
        return role

    lane = role.split()[0]
    matching_roles = [r for r in _STANDARD_ROLES if r.startswith(lane)]
    return max(matching_roles, key=lambda r: played_role_counts[champion_id][r])


def coerce_standard_lanes(played_role_counts, victor_role_counts):
    victor_filtered = collections.defaultdict(collections.Counter)
    played_filtered = collections.defaultdict(collections.Counter)

    for champion_id in played_role_counts.iterkeys():
        for role, count in played_role_counts[champion_id].iteritems():
            converted_role = coerce_standard_lane(played_role_counts, champion_id, role)

            played_filtered[champion_id][converted_role] += count
            victor_filtered[champion_id][converted_role] += victor_role_counts[champion_id][role]

    return played_filtered, victor_filtered