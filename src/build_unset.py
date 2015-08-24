from __future__ import unicode_literals
import sys
import argparse

import riot_api_cache


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()

    print "{" + ", ".join('"{}": {}'.format(field, 1) for field, _ in riot_api_cache.make_unset().items()) + "}"




if __name__ == "__main__":
    sys.exit(main())