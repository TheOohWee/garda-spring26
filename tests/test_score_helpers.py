from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from score import management_confidence_score, regex_flag, risk_mentions_count, sentences_matching_keywords


class ScoreHelperTests(unittest.TestCase):
    def test_sentences_matching_keywords_finds_relevant_sentences(self) -> None:
        text = "Demand remains strong. We continue hiring engineers. Margins were stable."
        matches = sentences_matching_keywords(text, ["demand", "hiring"])
        self.assertEqual(len(matches), 2)

    def test_guidance_regex_flags_directional_language(self) -> None:
        self.assertTrue(regex_flag("We raised guidance for the full year.", [r"\brais(?:e|ed|ing)\s+guidance\b"]))
        self.assertFalse(regex_flag("We reiterated guidance.", [r"\blower(?:ed|ing)?\s+guidance\b"]))

    def test_confidence_and_risk_helpers_behave(self) -> None:
        confident = "We are confident and on track despite some headwinds and risk."
        hedged = "We may face volatility and uncertain demand."
        self.assertGreater(management_confidence_score(confident), management_confidence_score(hedged))
        self.assertGreaterEqual(risk_mentions_count(confident), 2)


if __name__ == "__main__":
    unittest.main()
