from __future__ import annotations

import argparse
from pathlib import Path

from build_csv import main as build_csv_main
from clean import main as clean_main
from report import main as report_main
from score import main as score_main
from scrape import main as scrape_main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the full garda1 pipeline.")
    parser.add_argument("--config", type=Path, default=None, help="Path to a TOML config file.")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip scraping.")
    parser.add_argument("--skip-clean", action="store_true", help="Skip cleaning.")
    parser.add_argument("--skip-build-csv", action="store_true", help="Skip metadata CSV build.")
    parser.add_argument("--skip-score", action="store_true", help="Skip transcript scoring.")
    parser.add_argument("--skip-report", action="store_true", help="Skip report generation.")
    parser.add_argument("--write-speakers", action="store_true", help="Emit speaker JSON sidecars during cleaning.")
    parser.add_argument("--seed-archive", action="store_true", help="Force archive discovery before scraping.")
    parser.add_argument("--force-clean", action="store_true", help="Reclean existing transcripts.")
    parser.add_argument("--force-score", action="store_true", help="Rescore cached transcripts.")
    parser.add_argument("--us-only", action="store_true", help="Filter metadata to US companies only.")
    parser.add_argument("--delay-seconds", type=float, default=3.0, help="Scraper delay between requests.")
    parser.add_argument("--verbose", action="store_true", help="Pass verbose logging through to each stage.")
    args = parser.parse_args(argv)

    if not args.skip_scrape:
        command = ["--delay-seconds", str(args.delay_seconds)]
        if args.config:
            command.extend(["--config", str(args.config)])
        if args.seed_archive:
            command.append("--seed-archive")
        if args.verbose:
            command.append("--verbose")
        scrape_main(command)

    if not args.skip_clean:
        command: list[str] = []
        if args.config:
            command.extend(["--config", str(args.config)])
        if args.write_speakers:
            command.append("--write-speakers")
        if args.force_clean:
            command.append("--force")
        if args.verbose:
            command.append("--verbose")
        clean_main(command)

    if not args.skip_build_csv:
        command = []
        if args.us_only:
            command.append("--us-only")
        if args.verbose:
            command.append("--verbose")
        build_csv_main(command)

    if not args.skip_score:
        command = []
        if args.config:
            command.extend(["--config", str(args.config)])
        if args.force_score:
            command.append("--force")
        if args.verbose:
            command.append("--verbose")
        score_main(command)

    if not args.skip_report:
        command = []
        if args.verbose:
            command.append("--verbose")
        report_main(command)


if __name__ == "__main__":
    main()
