from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clean import extract_clean_text, extract_speakers
from scrape import parse_transcript_metadata


SAMPLE_HTML = """
<html>
  <head>
    <title>Apple (AAPL) Q2 2024 Earnings Call Transcript</title>
    <script type="application/ld+json">
      {
        "@type": "NewsArticle",
        "headline": "Apple (AAPL) Q2 2024 Earnings Call Transcript",
        "datePublished": "2024-04-26T16:30:00Z"
      }
    </script>
  </head>
  <body>
    <article>
      <h1>Apple (AAPL) Q2 2024 Earnings Call Transcript</h1>
      <p>Prepared Remarks:</p>
      <p>Tim Cook -- Chief Executive Officer</p>
      <p>We saw strong demand for iPhone and services.</p>
      <p>Question-and-Answer Session</p>
      <p>Operator</p>
      <p>The next question comes from the line of Analyst.</p>
      <p>Find out why 1,000,000 investors trust Motley Fool.</p>
    </article>
  </body>
</html>
"""


class ScrapeCleanTests(unittest.TestCase):
    def test_parse_transcript_metadata_extracts_core_fields(self) -> None:
        metadata, text = parse_transcript_metadata(SAMPLE_HTML, "https://www.fool.com/earnings/call/example")
        self.assertEqual(metadata["ticker"], "AAPL")
        self.assertEqual(metadata["quarter"], "Q2")
        self.assertEqual(metadata["fiscal_year"], "2024")
        self.assertEqual(metadata["call_date"], "2024-04-26")
        self.assertEqual(metadata["company"], "Apple")
        self.assertIn("Tim Cook", text)

    def test_extract_clean_text_removes_boilerplate(self) -> None:
        cleaned = extract_clean_text(SAMPLE_HTML)
        self.assertIn("Tim Cook -- Chief Executive Officer", cleaned)
        self.assertNotIn("Find out why 1,000,000 investors", cleaned)
        self.assertNotIn("Prepared Remarks:", cleaned)

    def test_extract_speakers_builds_sidecar_rows(self) -> None:
        cleaned = "Tim Cook -- Chief Executive Officer\nWe saw strong demand in iPhone.\n\nOperator -- Moderator\nNext question please."
        speakers = extract_speakers(cleaned)
        self.assertEqual(speakers[0]["speaker"], "Tim Cook")
        self.assertEqual(speakers[0]["role"], "Chief Executive Officer")
        self.assertIn("strong demand", speakers[0]["text"])


if __name__ == "__main__":
    unittest.main()
