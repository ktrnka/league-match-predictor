from __future__ import unicode_literals
import sys
import argparse

from src.riot_api_cache import make_unset


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()

    print "{" + ", ".join('"{}": {}'.format(field, value) for field, value in make_unset().items()) + "}"




if __name__ == "__main__":
    sys.exit(main())