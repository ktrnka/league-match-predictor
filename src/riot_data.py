from __future__ import unicode_literals
import pprint
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
        return sum(Tier._indexes[tier] for tier in tiers) / float(len(tiers))



class Champion(object):
    known_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 48, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 67, 68, 69, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 89, 90, 91, 92, 96, 98, 99, 101, 102, 103, 104, 105, 106, 107, 110, 111, 112, 113, 114, 115, 117, 119, 120, 121, 122, 126, 127, 131, 133, 134, 143, 150, 154, 157, 161, 201, 222, 223, 236, 238, 245, 254, 266, 267, 268, 412, 421, 429, 432]


class Summoner:
    def __init__(self, data):
        self.id = data["id"]

        # unprocessed data may not have summoner names
        self.name = data.get("name", None)

    def export(self):
        return {"id": self.id, "name": self.name}

    def __str__(self):
        return "{}__{}".format(self.name, self.id)


class Participant:
    def __init__(self, team_id, spells, champion_id, name, id=None):
        assert len(spells) == 2

        self.team_id = team_id
        self.spells = spells
        self.champion_id = champion_id
        self.name = name
        self.id = None

    @staticmethod
    def from_joined(data):
        return Participant(data["teamId"], [data["spell1Id"], data["spell2Id"]], data["championId"], data["summonerName"])

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
        return Participant(player["teamId"], [player["spell1Id"], player["spell2Id"]], player["championId"],
                                   identity["player"]["summonerName"], id=identity["player"]["summonerId"])

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


"""
Match value documentation from Riot API

matchMode	string	Match mode (Legal values: CLASSIC, ODIN, ARAM, TUTORIAL, ONEFORALL, ASCENSION, FIRSTBLOOD, KINGPORO)

matchType	string	Match type (Legal values: CUSTOM_GAME, MATCHED_GAME, TUTORIAL_GAME)

queueType Match queue type (Legal values: CUSTOM, NORMAL_5x5_BLIND, RANKED_SOLO_5x5, RANKED_PREMADE_5x5, BOT_5x5, NORMAL_3x3, RANKED_PREMADE_3x3, NORMAL_5x5_DRAFT, ODIN_5x5_BLIND, ODIN_5x5_DRAFT, BOT_ODIN_5x5, BOT_5x5_INTRO, BOT_5x5_BEGINNER, BOT_5x5_INTERMEDIATE, RANKED_TEAM_3x3, RANKED_TEAM_5x5, BOT_TT_3x3, GROUP_FINDER_5x5, ARAM_5x5, ONEFORALL_5x5, FIRSTBLOOD_1x1, FIRSTBLOOD_2x2, SR_6x6, URF_5x5, ONEFORALL_MIRRORMODE_5x5, BOT_URF_5x5, NIGHTMARE_BOT_5x5_RANK1, NIGHTMARE_BOT_5x5_RANK2, NIGHTMARE_BOT_5x5_RANK5, ASCENSION_5x5, HEXAKILL, BILGEWATER_ARAM_5x5, KING_PORO_5x5, COUNTER_PICK, BILGEWATER_5x5)
"""