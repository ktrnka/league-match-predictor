from __future__ import unicode_literals
import ConfigParser
import collections
import logging
import sys
import argparse
import time
import scrape_riot_api
import riot_api
import riot_api_cache
import riot_data


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def explore_side(riot_cache):
    victor_counts = collections.Counter()
    victor_by_tier = collections.defaultdict(collections.Counter)
    victor_by_queue_tier = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    victor_by_patch = collections.defaultdict(collections.Counter)

    queue_counts = collections.Counter()

    for match in riot_cache.get_matches():
        victor_counts[match.get_winning_team_id()] += 1
        victor_by_tier[match.get_average_tier()][match.get_winning_team_id()] += 1
        victor_by_patch[match.version][match.get_winning_team_id()] += 1
        victor_by_queue_tier[match.queue_type][match.get_average_tier()][match.get_winning_team_id()] += 1
        queue_counts[match.queue_type] += 1

    print "Queue counts", queue_counts.most_common()

    total_matches = sum(victor_counts.itervalues())
    for team_id, win_count in victor_counts.iteritems():
        print "{} team: {:.1f}% of {:,} matches won".format(riot_api.RiotService.get_team_name(team_id), 100. * win_count / total_matches, total_matches)

    tiers = sorted(victor_by_tier.iterkeys(), key=riot_data.Tier.make_sortable_key)
    for tier in tiers:
        print "{} tier".format(tier)

        total_matches = sum(victor_by_tier[tier].itervalues())
        for team_id, win_count in victor_by_tier[tier].iteritems():
            print "\t{} team: {:.1f}% of {:,} matches won".format(riot_api.RiotService.get_team_name(team_id), 100. * win_count / total_matches, total_matches)

    for queue, tier_stats in victor_by_queue_tier.iteritems():
        print "Queue {}".format(queue)

        tiers = sorted(tier_stats.iterkeys(), key=riot_data.Tier.make_sortable_key)
        for tier in tiers:
            print "\t{} tier".format(tier)

            total_matches = sum(tier_stats[tier].itervalues())
            for team_id, win_count in tier_stats[tier].iteritems():
                print "\t\t{} team: {:.1f}% of {:,} matches won".format(riot_api.RiotService.get_team_name(team_id), 100. * win_count / total_matches, total_matches)

    patches = sorted(victor_by_patch.iterkeys())
    for patch in patches:
        total_matches = sum(victor_by_patch[patch].itervalues())

        if total_matches < 1000:
            continue

        print "Patch {}".format(patch)

        for team_id, win_count in victor_by_patch[patch].iteritems():
            print "\t{} team: {:.1f}% of {:,} matches won".format(riot_api.RiotService.get_team_name(team_id), 100. * win_count / total_matches, total_matches)



def explore_champions(riot_cache, riot_connection):
    victor_counts = collections.Counter()
    played_counts = collections.Counter()

    # compute champion win rates from match history
    for match in riot_cache.get_matches():
        winner = match.get_winning_team_id()

        for team_id, champion_set in match.get_picks().iteritems():
            played_counts.update(champion_set)

            if team_id == winner:
                victor_counts.update(champion_set)

    agg_stats, agg_champion_stats = riot_cache.aggregate_champion_stats()

    print "Champion IDs found: {}".format(sorted(played_counts.keys()))
    champion_names = {i: riot_connection.get_champion_info(i)["name"] for i in played_counts.iterkeys()}

    print "Champion stats"
    for champion_id in sorted(played_counts.iterkeys()):
        print "{} [{}]".format(champion_names[champion_id], champion_id)

        print "\tMatch history: {:.1f}% win rate out of {:,} games played".format(100. * victor_counts[champion_id] / played_counts[champion_id], played_counts[champion_id])
        print "\tRanked stats:  {:.1f}% win rate out of {:,} games played".format(100. * agg_champion_stats[champion_id].get_win_rate(), agg_champion_stats[champion_id].get_played())

    print "Total win rate from ranked stats:  {:.1f}% in {:,} games".format(100. * agg_stats.get_win_rate(), agg_stats.get_played())
    print "Total win rate from ranked stats2: {:.1f}%".format(100. * sum(v.won for v in agg_champion_stats.values()) / sum(v.played for v in agg_champion_stats.values()))
    print "Total win rate from match history: {:.1f}% in {:,} games".format(100. * sum(victor_counts.values()) / sum(played_counts.values()), sum(played_counts.values()))


def main():
    args = scrape_riot_api.parse_args()

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

    riot_cache = riot_api_cache.ApiCache(config)
    riot_connection = riot_api.RiotService.from_config(config)

    explore_side(riot_cache)
    explore_champions(riot_cache, riot_connection)



if __name__ == "__main__":
    sys.exit(main())