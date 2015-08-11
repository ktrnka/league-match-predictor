from __future__ import unicode_literals
import logging
import sys
import argparse
import time


class ThrottledFilter(logging.Filter):
    def __init__(self, name="", delay_seconds=5):
        super(ThrottledFilter, self).__init__(name=name)

        self.last_message = None
        self.delay = delay_seconds

    def filter(self, record):
        return super(ThrottledFilter, self).filter(record) or self._filter(record)

    def _filter(self, record):
        if not self.last_message or time.clock() - self.last_message > self.delay:
            self.last_message = time.clock()
            return True

        return False


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    sys.exit(main())