from __future__ import unicode_literals
import ConfigParser
import collections
import logging
import sys
import argparse
import functools32
import scrape_riot_api
import riot_api
import riot_api_cache
import riot_data
import utilities

_STANDARD_ROLES = {"BOTTOM DUO_SUPPORT", "BOTTOM DUO_CARRY", "JUNGLE", "MIDDLE", "TOP"}

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


def coerce_standard_lanes(played_role_counts, victor_role_counts):
    victor_filtered = collections.defaultdict(collections.Counter)
    played_filtered = collections.defaultdict(collections.Counter)

    for champion_id in played_role_counts.iterkeys():
        for role, count in played_role_counts[champion_id].iteritems():
            converted_role = coerce_standard_lane(played_role_counts, champion_id, role)

            played_filtered[champion_id][converted_role] += count
            victor_filtered[champion_id][converted_role] += victor_role_counts[champion_id][role]

    return played_filtered, victor_filtered

class LaneStats(object):
    """Track win rates by champion and lane, help to normalize the lanes and roles"""
    def __init__(self):
        self.play_counts = collections.defaultdict(collections.Counter)
        self.win_counts = collections.defaultdict(collections.Counter)

        self.matchup_play_counts = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
        self.matchup_win_counts = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))

        self.num_games = 0

    def add(self, champion_id, role, is_win):
        self.play_counts[champion_id][role] += 1
        if is_win:
            self.win_counts[champion_id][role] += 1

    def add_match(self, match):
        assert isinstance(match, riot_data.Match)

        winner = match.get_winning_team_id()

        for team_id, role_map in match.get_roles().iteritems():
            for role, champion_id in role_map.iteritems():
                self.add(champion_id, role, team_id == winner)

        # add the lane matchup
        roles = collections.defaultdict(collections.defaultdict)
        for team_id, role_map in match.get_roles().iteritems():
            for role, champion_id in role_map.iteritems():
                # recompute every time until the stats are saturated then use the memoized version
                if self.num_games > 10000:
                    role = self.coerce_role(champion_id, role)
                else:
                    role = coerce_standard_lane(self.play_counts, champion_id, role)
                roles[role][team_id] = champion_id

        teams = [100, 200]
        for role, champion_map in roles.iteritems():
            if not all(team in champion_map for team in teams):
                continue

            champions = [champion_map[team] for team in teams]
            winners = [int(winner == team) for team in teams]

            assert sum(winners) == 1

            # increment blue side
            self.matchup_play_counts[champions[0]][role][champions[1]] += 1
            self.matchup_win_counts[champions[0]][role][champions[1]] += winners[0]

            # increment red side
            self.matchup_play_counts[champions[1]][role][champions[0]] += 1
            self.matchup_win_counts[champions[1]][role][champions[0]] += winners[1]

        self.num_games += 1

    def get_stats_by_role(self, match):
        assert isinstance(match, riot_data.Match)

    @functools32.lru_cache(5000)
    def coerce_role(self, champion_id, role):
        return coerce_standard_lane(self.play_counts, champion_id, role)

    def coerce_roles(self):
        self.play_counts, self.win_counts = coerce_standard_lanes(self.play_counts, self.win_counts)

    def print_matchups(self, champion_names):
        for champion_a in sorted(self.matchup_play_counts.iterkeys()):
            for role, _ in utilities.most_common_percent(self.play_counts[champion_a], 0.9):
                print "{} {}".format(champion_names[champion_a], role)

                for champion_b, num_played in utilities.most_common_percent(self.matchup_play_counts[champion_a][role], 0.9):
                    num_wins = self.matchup_win_counts[champion_a][role][champion_b]
                    p = num_wins / float(num_played)

                    print "\tvs {} {}: {:.1f}% +/- {:.1f}% in {:,} games".format(champion_names[champion_b],
                                                                         role,
                                                                         100. * num_wins / num_played,
                                                                         100. * utilities.binomial_stddev(p, num_played),
                                                                         num_played)



def coerce_standard_lane(played_role_counts, champion_id, role):
    if role in _STANDARD_ROLES:
        return role

    lane = role.split()[0]
    matching_roles = [r for r in _STANDARD_ROLES if r.startswith(lane)]
    return max(matching_roles, key=lambda r: played_role_counts[champion_id][r])


def explore_champions(riot_cache, riot_connection):
    victor_counts = collections.Counter()
    played_counts = collections.Counter()

    role_stats = LaneStats()

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

    print "Champion stats by role"
    for champion_id in sorted(role_stats.play_counts.iterkeys()):
        print "{} [{}]".format(champion_names[champion_id], champion_id)

        for role, count in utilities.most_common_percent(role_stats.play_counts[champion_id], 0.9):
            print "\t{:20s}: {:.1f}% win rate out of {:,} games played".format(role,
                100. * role_stats.win_counts[champion_id][role] / role_stats.play_counts[champion_id][role], role_stats.play_counts[champion_id][role])

    print "Total win rate from ranked stats:  {:.1f}% in {:,} games".format(100. * agg_stats.get_win_rate(),
                                                                            agg_stats.get_played())
    print "Total win rate from ranked stats2: {:.1f}%".format(
        100. * sum(v.won for v in agg_champion_stats.values()) / sum(v.played for v in agg_champion_stats.values()))
    print "Total win rate from match history: {:.1f}% in {:,} games".format(
        100. * sum(victor_counts.values()) / sum(played_counts.values()), sum(played_counts.values()))

    role_stats.print_matchups(champion_names)

    # second pass: force each team into their roles then do diffs
    played_counts = collections.defaultdict(collections.Counter)
    victor_counts = collections.defaultdict(collections.Counter)
    for match in riot_cache.get_matches():
        winner = match.get_winning_team_id()

        roles = collections.defaultdict(collections.defaultdict)
        for team_id, role_map in match.get_roles().iteritems():
            for role, champion_id in role_map.iteritems():
                role = role_stats.coerce_role(champion_id, role)
                roles[team_id][role] = champion_id

        teams = sorted(roles.iterkeys())
        for role in roles[teams[0]]:
            if role not in roles[teams[1]]:
                continue

            champions = roles[teams[0]][role], roles[teams[1]][role]

            played_counts[champions[0]][(champions[1], role)] += 1
            played_counts[champions[1]][(champions[0], role)] += 1

            if winner == teams[0]:
                victor_counts[champions[0]][(champions[1], role)] += 1
            else:
                victor_counts[champions[1]][(champions[0], role)] += 1

    for champion_a in sorted(played_counts.iterkeys()):
        print "{} [{}]".format(champion_names[champion_a], champion_a)
        for (champion_b, role), num_played in utilities.most_common_percent(played_counts[champion_a], 0.9):
            num_wins = victor_counts[champion_a][(champion_b, role)]
            p = num_wins / float(num_played)

            print "\tvs {} {}: {:.1f}% +/- {:.1f}% in {:,} games".format(champion_names[champion_b],
                                                                 role,
                                                                 100. * num_wins / num_played,
                                                                 100. * utilities.binomial_stddev(p, num_played),
                                                                 num_played)



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