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