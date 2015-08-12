from __future__ import unicode_literals
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    sys.exit(main())


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
        wrapped_data["queueType"] = data["gameQueueConfigId"]
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