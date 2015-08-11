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

        self.teamId = team_id
        self.spells = spells
        self.championId = champion_id
        self.name = name
        self.id = None

    @staticmethod
    def from_joined(data):
        return Participant(data["teamId"], [data["spell1Id"], data["spell2Id"]], data["championId"], data["summonerName"])

    @staticmethod
    def parse_participants(participants, participant_identities):
        for player, identity in zip(participants, participant_identities):
            yield Participant.from_split(player, identity)

    @staticmethod
    def from_split(player, identity):
        return Participant(player["teamId"], [player["spell1Id"], player["spell2Id"]], player["championId"],
                                   identity["player"]["summonerName"], id=identity["player"]["summonerId"])


class Match(object):
    def __init__(self, data):
        self.id = data["matchId"]
        self.mode = data["matchMode"]
        self.type = data["matchType"]
        self.creation_time = data["matchCreation"]
        self.duration = data["matchDuration"]
        self.queue_type = data["queueType"]
        self.version = data["matchVersion"]

        self.players = list(Participant.parse_participants(data["participants"], data["participantIdentities"]))