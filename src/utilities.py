from __future__ import unicode_literals
import logging
import sys
import argparse
import time
import collections

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


class EstCompletionTimer(object):
    """Simple tracker to estimate how long it'll take to finish a long-running task"""
    def __init__(self):
        self.start_time = None
        self.units_processed = 0
        self.outcomes = collections.Counter()
        self.start()

    def start(self):
        self.start_time = time.time()
        self.units_processed = 0
        self.outcomes.clear()

        return self

    def update(self, outcome_label=None, units_processed=1):
        self.units_processed += units_processed

        if outcome_label:
            self.outcomes[outcome_label] += units_processed

        return self

    def get_expected_total_seconds(self, total_units):
        elapsed = time.time() - self.start_time
        units_completed = self.units_processed / float(total_units)

        return elapsed / units_completed

    def get_expected_remaining_seconds(self, total_units):
        elapsed = time.time() - self.start_time
        return self.get_expected_total_seconds(total_units) - elapsed

    def log_info(self, total_units):
        remaining = int(self.get_expected_remaining_seconds(total_units))

        hours, seconds = divmod(remaining, 60 * 60)
        minutes, seconds = divmod(seconds, 60)
        rem_string = number_string(minutes, "minute", "minutes")
        if hours:
            rem_string = ", ".join((number_string(hours, "hour", "hours"), rem_string))

        return "{} remaining. {:,} / {:,} completed. {}".format(rem_string, self.units_processed, total_units, summarize_counts(self.outcomes))


class DevReminderError(BaseException):
    """Error to remind me to implement something"""
    def __init__(self, message):
        super(DevReminderError, self).__init__(message)


def number_string(number, singular_unit, plural_unit):
    return "{} {}".format(number, singular_unit if number == 1 else plural_unit)


def summarize_counts(counter):
    total = sum(counter.itervalues())
    return ", ".join("{}: {:.1f}% ({:,})".format(k, 100. * v / total, v) for k, v in counter.most_common())


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    sys.exit(main())


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]