from __future__ import unicode_literals
import ConfigParser
import collections
import logging
import sys
import argparse
from src import scrape_featured
from src.riot_api import RiotService
from src.riot_api_cache import ApiCache
from src.riot_data import Tier


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def explore_side(riot_cache):
    victor_counts = collections.Counter()
    victor_by_tier = collections.defaultdict(collections.Counter)
    victor_by_queue_tier = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))

    for match in riot_cache.get_matches():
        victor_counts[match.get_winning_team_id()] += 1
        victor_by_tier[match.get_average_tier()][match.get_winning_team_id()] += 1
        victor_by_queue_tier[match.queue_type][match.get_average_tier()][match.get_winning_team_id()] += 1

    total_matches = sum(victor_counts.itervalues())
    for team_id, win_count in victor_counts.iteritems():
        print "{} team: {:.1f}% of {:,} matches won".format(RiotService.get_team_name(team_id), 100. * win_count / total_matches, total_matches)

    tiers = sorted(victor_by_tier.iterkeys(), key=Tier.make_sortable_key)
    for tier in tiers:
        print "{} tier".format(tier)

        total_matches = sum(victor_by_tier[tier].itervalues())
        for team_id, win_count in victor_by_tier[tier].iteritems():
            print "\t{} team: {:.1f}% of {:,} matches won".format(RiotService.get_team_name(team_id), 100. * win_count / total_matches, total_matches)

    for queue, tier_stats in victor_by_queue_tier.iteritems():
        print "Queue {}".format(queue)

        tiers = sorted(tier_stats.iterkeys(), key=Tier.make_sortable_key)
        for tier in tiers:
            print "\t{} tier".format(tier)

            total_matches = sum(tier_stats[tier].itervalues())
            for team_id, win_count in tier_stats[tier].iteritems():
                print "\t\t{} team: {:.1f}% of {:,} matches won".format(RiotService.get_team_name(team_id), 100. * win_count / total_matches, total_matches)

def explore_champions(riot_cache, riot_connection):
    victor_counts = collections.Counter()
    played_counts = collections.Counter()

    for match in riot_cache.get_matches():
        winner = match.get_winning_team_id()

        for team_id, champion_set in match.get_picks().iteritems():
            played_counts.update(champion_set)

            if team_id == winner:
                victor_counts.update(champion_set)

    print "Champion IDs found: {}".format(sorted(played_counts.keys()))
    champion_names = {i: riot_connection.get_champion_info(i)["name"] for i in played_counts.iterkeys()}

    for champion_id in sorted(played_counts.iterkeys()):
        print "{} [{}]: {:.1f}% win rate out of {:,} games played".format(champion_names[champion_id], champion_id, 100. * victor_counts[champion_id] / played_counts[champion_id], played_counts[champion_id])

def main():
    args = scrape_featured.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.captureWarnings(True)
    logger = logging.getLogger(__name__)

    # reduce connection spam
    logging.getLogger("requests.packages.urllib3.connectionpool").setLevel(logging.WARNING)

    config = ConfigParser.RawConfigParser()
    config.read([args.config])

    riot_cache = ApiCache(config)
    riot_connection = RiotService.from_config(config)

    # explore_side(riot_cache)
    explore_champions(riot_cache, riot_connection)



if __name__ == "__main__":
    sys.exit(main())