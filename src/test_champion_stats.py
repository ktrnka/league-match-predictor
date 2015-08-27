import unittest
import riot_data


class MyTestCase(unittest.TestCase):
    def test_empty(self):
        stats = riot_data.ChampionStats.from_wins_played(0, 0)

        self.assertEqual(0, stats.get_played())
        self.assertEqual(0.5, stats.get_win_rate())

        self.assertEqual(0, stats.get_played(remove_games=1))
        self.assertEqual(0, stats.get_played(remove_games=5))

        self.assertEqual(0.5, stats.get_win_rate(remove_games=1, remove_wins=0))
        self.assertEqual(0.5, stats.get_win_rate(remove_games=1, remove_wins=1))

    def test_nonempty(self):
        stats = riot_data.ChampionStats.from_wins_played(6, 10)

        self.assertEqual(10, stats.get_played())
        self.assertEqual(0.6, stats.get_win_rate())

        self.assertEqual(9, stats.get_played(remove_games=1))
        self.assertEqual(5, stats.get_played(remove_games=5))

        self.assertEqual(6. / 9, stats.get_win_rate(remove_games=1, remove_wins=0))
        self.assertEqual(5. / 9, stats.get_win_rate(remove_games=1, remove_wins=1))

        # ignore inconsistent
        self.assertEqual(0.6, stats.get_win_rate(remove_games=7, remove_wins=7))
        self.assertEqual(0.6, stats.get_win_rate(remove_games=11, remove_wins=1))



if __name__ == '__main__':
    unittest.main()
