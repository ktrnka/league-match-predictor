from __future__ import unicode_literals
import ConfigParser
import collections
import logging
import sys
import argparse

import scrape_riot_api
import riot_api
import riot_api_cache
import riot_data
import utilities


"""
Exploratory analysis of the database.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def get_current_average_league(riot_cache, match):
    assert isinstance(match, riot_data.Match)
    assert isinstance(riot_cache, riot_api_cache.MemoizeCache)

    leagues = [riot_cache.get_league(p.id) for p in match.players]
    return riot_data.LeagueEntry.average(leagues)


def explore_side(riot_cache):
    victor_counts = collections.Counter()
    victor_by_tier = collections.defaultdict(collections.Counter)
    victor_by_queue_tier = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    victor_by_patch = collections.defaultdict(collections.Counter)

    queue_counts = collections.Counter()

    timer = utilities.EstCompletionTimer().start(riot_cache.get_num_matches())
    logger = logging.getLogger("explore_side_heartbeat")
    logger.addFilter(utilities.ThrottledFilter(delay_seconds=10))

    for match in riot_cache.get_matches():
        victor_counts[match.get_winning_team_id()] += 1
        victor_by_patch[match.version][match.get_winning_team_id()] += 1

        average_highest_achieved_tier = match.get_average_tier()
        average_league = get_current_average_league(riot_cache, match)
        for average_league in [average_highest_achieved_tier, average_league.tier + "*", average_league.get_tier_division()]:
            victor_by_tier[average_league][match.get_winning_team_id()] += 1
            victor_by_queue_tier[match.queue_type][average_league][match.get_winning_team_id()] += 1

        queue_counts[match.queue_type] += 1
        timer.update()
        logger.info(timer.log_info())

    print "Queue counts", queue_counts.most_common()

    total_matches = sum(victor_counts.itervalues())
    for team_id, win_count in victor_counts.iteritems():
        print "{} team: {:.1f}% of {:,} matches won".format(riot_api.RiotService.get_team_name(team_id),
                                                            100. * win_count / total_matches, total_matches)

    tiers = sorted(victor_by_tier.iterkeys(), key=riot_data.Tier.make_sortable_key)
    for tier in tiers:
        print "{} tier".format(tier)

        total_matches = sum(victor_by_tier[tier].itervalues())
        for team_id, win_count in victor_by_tier[tier].iteritems():
            print "\t{} team: {:.1f}% of {:,} matches won".format(riot_api.RiotService.get_team_name(team_id),
                                                                  100. * win_count / total_matches, total_matches)

    for queue, tier_stats in victor_by_queue_tier.iteritems():
        print "Queue {}".format(queue)

        tiers = sorted(tier_stats.iterkeys(), key=riot_data.Tier.make_sortable_key)
        for tier in tiers:
            print "\t{} tier".format(tier)

            total_matches = sum(tier_stats[tier].itervalues())
            for team_id, win_count in tier_stats[tier].iteritems():
                print "\t\t{} team: {:.1f}% of {:,} matches won".format(riot_api.RiotService.get_team_name(team_id),
                                                                        100. * win_count / total_matches, total_matches)

    patches = sorted(victor_by_patch.iterkeys())
    for patch in patches:
        total_matches = sum(victor_by_patch[patch].itervalues())

        if total_matches < 1000:
            continue

        print "Patch {}".format(patch)

        for team_id, win_count in victor_by_patch[patch].iteritems():
            print "\t{} team: {:.1f}% of {:,} matches won".format(riot_api.RiotService.get_team_name(team_id),
                                                                  100. * win_count / total_matches, total_matches)


def explore_champions(riot_cache, riot_connection):
    victor_counts = collections.Counter()
    played_counts = collections.Counter()

    role_stats = riot_data.RoleStats()

    # compute champion win rates from match history
    for match in riot_cache.get_matches():
        winner = match.get_winning_team_id()

        for team_id, champion_set in match.get_picks().iteritems():
            played_counts.update(champion_set)

            if team_id == winner:
                victor_counts.update(champion_set)

        role_stats.add_match(match)

    agg_stats, agg_champion_stats = riot_cache.aggregate_champion_stats()

    print "Champion IDs found: {}".format(sorted(played_counts.keys()))
    champion_names = {i: riot_connection.get_champion_info(i)["name"] for i in played_counts.iterkeys()}

    print "Champion stats"
    for champion_id in sorted(played_counts.iterkeys()):
        print "{} [{}]".format(champion_names[champion_id], champion_id)

        print "\tMatch history: {:.1f}% win rate out of {:,} games played".format(
            100. * victor_counts[champion_id] / played_counts[champion_id], played_counts[champion_id])
        print "\tRanked stats:  {:.1f}% win rate out of {:,} games played".format(
            100. * agg_champion_stats[champion_id].get_win_rate(), int(agg_champion_stats[champion_id].get_played()))

    role_stats.coerce_roles()

    role_stats.print_roles(champion_names)

    print "Total win rate from ranked stats:  {:.1f}% in {:,} games".format(100. * agg_stats.get_win_rate(),
                                                                            agg_stats.get_played())
    print "Total win rate from ranked stats2: {:.1f}%".format(
        100. * sum(v.won for v in agg_champion_stats.values()) / sum(v.played for v in agg_champion_stats.values()))
    print "Total win rate from match history: {:.1f}% in {:,} games".format(
        100. * sum(victor_counts.values()) / sum(played_counts.values()), sum(played_counts.values()))

    role_stats.print_matchups(champion_names)


def explore_versions(riot_cache):
    version_counts = collections.Counter()
    for match in riot_cache.get_matches():
        version_counts[match.version] += 1

    # compute counts for the major version (biweekly)
    for version in version_counts.keys():
        pieces = version.split(".")
        major = ".".join(pieces[0:2])
        version_counts[major] += version_counts[version]

    print "Distribution of matches by game version"
    for key in sorted(version_counts.iterkeys()):
        print "{}: {:,}".format(key, version_counts[key])


def explore_current_league(riot_cache):
    assert isinstance(riot_cache, riot_api_cache.MemoizeCache)
    league_counts = collections.Counter()
    for player in riot_cache.get_players():
        league = riot_cache.get_league(player.id)
        if league:
            league_counts[league.tier] += 1
            league_counts["{} {}".format(league.tier, league.division)] += 1

    print "Distribution of players by league"
    for key in sorted(league_counts.iterkeys()):
        print "{}: {:,}".format(key, league_counts[key])


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

    riot_connection = riot_api.RiotService.from_config(config)
    riot_cache = riot_api_cache.MemoizeCache(config, riot_connection)

    explore_champions(riot_cache, riot_connection)
    explore_current_league(riot_cache)
    explore_versions(riot_cache)
    explore_side(riot_cache)


if __name__ == "__main__":
    sys.exit(main())