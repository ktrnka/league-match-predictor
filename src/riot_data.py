from __future__ import unicode_literals
import sys
import argparse
import datetime
import collections


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    sys.exit(main())


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
    def __init__(self, team_id, spells, champion_id, name, id=None, tier=None, participant_id=None):
        assert len(spells) == 2

        self.team_id = team_id
        self.spells = spells
        self.champion_id = champion_id
        self.name = name

        self.id = id
        self.tier = tier
        self.participant_id = participant_id

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
        return Participant(player["teamId"],
                           [player["spell1Id"], player["spell2Id"]],
                           player["championId"],
                           identity["player"]["summonerName"],
                           id=identity["player"]["summonerId"],
                           tier=player["highestAchievedSeasonTier"],
                           participant_id=player["participantId"])

class Queue(object):
    id_to_name = {
        4: "RANKED_SOLO_5x5",
        42: "RANKED_TEAM_5x5"
    }

    @staticmethod
    def to_name(queue_id):
        assert isinstance(queue_id, int)

        return Queue.id_to_name.get(queue_id, unicode(queue_id))

class Match(object):
    QUEUE_RANKED_SOLO = "RANKED_SOLO_5x5"
    QUEUE_RANKED_5 = "RANKED_TEAM_5x5"

    def __init__(self, data):
        self.id = data["matchId"]
        self.mode = data["matchMode"]
        self.type = data["matchType"]
        self.creation_time = data["matchCreation"]
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

    def get_creation_datetime(self):
        # The creation time from API is milliseconds since the epoch not seconds
        # And it's pacific time not UTC
        return datetime.datetime.fromtimestamp(self.creation_time / 1000.)

    @staticmethod
    def from_featured(data):
        """Parse a partial Match object from a featured match"""
        assert isinstance(data, dict)

        # pprint.pprint(["Featured match data", data])

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

        # self.damage_taken = data["totalDamageTaken"]
        #
        # self.kills = data["totalChampionKills"]
        # self.deaths = data["totalDeathsPerSession"]
        # self.assists = data["totalAssists"]
        #
        # self.num_first_blood = data["totalFirstBlood"]
        #
        # self.double_kills = data["totalDoubleKills"]
        # self.triple_kills = data["totalTripleKills"]
        # self.quadra_kills = data["totalQuadraKills"]
        # self.penta_kills = data["totalPentaKills"]
        #
        # self.turret_kills = data["totalTurretsKilled"]

    @staticmethod
    def from_wins_played(num_wins, num_played):
        data = collections.Counter()
        data["totalSessionsPlayed"] = num_played
        data["totalSessionsWon"] = num_wins
        return ChampionStats(data)

    def get_kda(self, remove_stats=None):
        return 1
        # kills = self.kills
        # assists = self.assists
        # deaths = self.deaths
        # if remove_stats:
        #     kills -= remove_stats.kills
        #     assists -= remove_stats.assists
        #     deaths -= remove_stats.deaths
        #
        # return (kills + assists + 1) / float(deaths + 1)

    def get_damage_efficiency(self, remove_stats=None):
        return 1
        # damage_dealt = self.total_damage
        # damage_taken = self.damage_taken
        #
        # if remove_stats:
        #     damage_dealt -= remove_stats.total_damage
        #     damage_taken -= remove_stats.damage_taken
        #
        # return (damage_dealt + 1) / float(damage_taken + 1)

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

EMPTY_CHAMPION_STATS = ChampionStats(collections.Counter())