from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import load_app_config
from garda1.cli import build_parser


class ConfigAndCliTests(unittest.TestCase):
    def test_load_app_config_reads_toml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "garda1.toml"
            config_path.write_text(
                """
[scrape]
delay_seconds = 1.5
max_retries = 7

[clean]
write_speakers = false

[score]
finbert_model_name = "custom/finbert"
""".strip(),
                encoding="utf-8",
            )
            config = load_app_config(config_path)
            self.assertEqual(config.scrape_delay_seconds, 1.5)
            self.assertEqual(config.scrape_max_retries, 7)
            self.assertFalse(config.write_speakers)
            self.assertEqual(config.finbert_model_name, "custom/finbert")

    def test_cli_parser_accepts_run_all(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run-all", "--seed-archive", "--write-speakers"])
        self.assertEqual(args.command, "run-all")
        self.assertTrue(args.seed_archive)
        self.assertTrue(args.write_speakers)


if __name__ == "__main__":
    unittest.main()
