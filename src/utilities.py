from __future__ import unicode_literals
import logging
import sys
import argparse
import time


class ThrottledFilter(logging.Filter):
    """
    Filter to print a message every N seconds.
    """
    def __init__(self, name="", delay_seconds=2):
        super(ThrottledFilter, self).__init__(name=name)

        self.last_message = None
        self.delay = delay_seconds

    def filter(self, record):
        return super(ThrottledFilter, self).filter(record) and self._filter(record)

    def _filter(self, record):
        if not self.last_message or (time.clock() - self.last_message) > self.delay:
            self.last_message = time.clock()
            return True

        return False


class DevReminderError(BaseException):
    """Error to remind me to implement something"""
    def __init__(self, message):
        super(DevReminderError, self).__init__(message)


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    sys.exit(main())