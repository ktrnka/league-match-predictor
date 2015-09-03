from __future__ import unicode_literals
import logging
from operator import itemgetter
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
        if not self.last_message or (time.time() - self.last_message) > self.delay:
            self.last_message = time.time()
            return True

        return False


class RequestTimer(object):
    """Simple tracker to help test the time of network requests"""
    def __init__(self):
        self.elapsed_time = 0
        self.num_requests = 0

        self.__start_time = None

    def start(self):
        self.__start_time = time.time()

    def stop(self):
        self.elapsed_time += time.time() - self.__start_time
        self.num_requests += 1

    def get_requests_per_second(self):
        return self.num_requests / float(self.elapsed_time)

    def get_seconds_per_request(self):
        return self.elapsed_time / float(self.num_requests)


class DevReminderError(BaseException):
    """Error to remind me to implement something"""
    def __init__(self, message):
        super(DevReminderError, self).__init__(message)

def summarize_counts(counter):
    total = sum(counter.itervalues())
    return ", ".join("{}: {:.1f}% ({:,})".format(k, 100. * v / total, v) for k, v in sorted(counter.iteritems(), key=itemgetter(1), reverse=True))


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    sys.exit(main())