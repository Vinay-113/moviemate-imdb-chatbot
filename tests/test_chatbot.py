from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from movie_mate import MemoryStore, MovieChatbot, load_movies


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOVIES = load_movies(PROJECT_ROOT / "data" / "imdb_top_1000.csv")


class MovieMateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bot = MovieChatbot(MOVIES)

    def test_apollo_13_year_is_repaired(self) -> None:
        apollo = next(movie for movie in MOVIES if movie.title == "Apollo 13")
        self.assertEqual(apollo.year, 1995)

    def test_scifi_after_2010_filters_results(self) -> None:
        response = self.bot.respond("Suggest sci-fi movies released after 2010")
        self.assertGreater(len(response["cards"]), 0)
        for card in response["cards"]:
            self.assertIn("Sci-Fi", card["genres"])
            self.assertGreater(card["year"], 2010)

    def test_actor_query_starts_a_new_search(self) -> None:
        initial = self.bot.respond("Suggest sci-fi movies released after 2010")
        response = self.bot.respond("Movies starring Leonardo DiCaprio", initial["session_id"])
        self.assertEqual(response["filters"]["genres"], [])
        self.assertGreater(len(response["cards"]), 0)
        for card in response["cards"]:
            self.assertIn("Leonardo DiCaprio", card["stars"])

    def test_follow_up_refines_previous_results(self) -> None:
        initial = self.bot.respond("Suggest action movies")
        response = self.bot.respond("Only those released after 2015", initial["session_id"])
        self.assertEqual(response["filters"]["genres"], ["Action"])
        self.assertEqual(response["filters"]["year_min"], 2016)
        self.assertGreater(len(response["cards"]), 0)
        for card in response["cards"]:
            self.assertIn("Action", card["genres"])
            self.assertGreater(card["year"], 2015)

    def test_detail_lookup_answers_director_question(self) -> None:
        response = self.bot.respond("Who directed The Godfather?")
        self.assertIn("Francis Ford Coppola", response["reply"])
        self.assertEqual(response["cards"][0]["title"], "The Godfather")

    def test_plot_lookup_returns_single_title_detail(self) -> None:
        response = self.bot.respond("plot of Inception")
        self.assertIn("Overview:", response["reply"])
        self.assertEqual([card["title"] for card in response["cards"]], ["Inception"])

    def test_memory_store_updates_and_affects_response_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryStore(Path(directory) / "memory.json")
            bot = MovieChatbot(MOVIES, memory_store=memory)
            first = bot.respond("Suggest sci-fi movies", profile_id="user-1")
            second = bot.respond("Recommend something for me", profile_id="user-1")
            self.assertEqual(first["mode"], "fallback")
            self.assertIn("Sci-Fi", second["memory_summary"])


if __name__ == "__main__":
    unittest.main()
