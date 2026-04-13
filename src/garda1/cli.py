from __future__ import annotations

import argparse
from pathlib import Path

from build_csv import main as build_csv_main
from clean import main as clean_main
from prune_legacy import main as prune_legacy_main
from report import main as report_main
from run_pipeline import main as run_pipeline_main
from score import main as score_main
from scrape import main as scrape_main
from update_companies import main as update_companies_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="garda1 earnings-call analysis application")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape_parser = subparsers.add_parser("scrape", help="Discover and scrape Motley Fool transcripts.")
    scrape_parser.add_argument("--config", type=Path, default=None)
    scrape_parser.add_argument("--seed-archive", action="store_true")
    scrape_parser.add_argument("--delay-seconds", type=float, default=None)
    scrape_parser.add_argument("--max-retries", type=int, default=None)
    scrape_parser.add_argument("--verbose", action="store_true")

    clean_parser = subparsers.add_parser("clean", help="Clean raw transcript HTML into text files.")
    clean_parser.add_argument("--config", type=Path, default=None)
    clean_parser.add_argument("--write-speakers", action="store_true")
    clean_parser.add_argument("--force", action="store_true")
    clean_parser.add_argument("--verbose", action="store_true")

    csv_parser = subparsers.add_parser("build-csv", help="Build the transcript metadata CSV.")
    csv_parser.add_argument("--us-only", action="store_true")
    csv_parser.add_argument("--verbose", action="store_true")

    score_parser = subparsers.add_parser("score", help="Score transcripts with FinBERT and lexicons.")
    score_parser.add_argument("--config", type=Path, default=None)
    score_parser.add_argument("--force", action="store_true")
    score_parser.add_argument("--verbose", action="store_true")

    report_parser = subparsers.add_parser("report", help="Generate downstream CSV outputs.")
    report_parser.add_argument("--verbose", action="store_true")

    prune_parser = subparsers.add_parser("prune-legacy", help="Move old flat transcript leftovers into backup storage.")
    prune_parser.add_argument("--verbose", action="store_true")

    update_companies_parser = subparsers.add_parser("update-companies", help="Refresh data/companies.csv from scraped metadata.")
    update_companies_parser.add_argument("--verbose", action="store_true")

    run_parser = subparsers.add_parser("run-all", help="Run the full pipeline end to end.")
    run_parser.add_argument("--config", type=Path, default=None)
    run_parser.add_argument("--skip-scrape", action="store_true")
    run_parser.add_argument("--skip-clean", action="store_true")
    run_parser.add_argument("--skip-build-csv", action="store_true")
    run_parser.add_argument("--skip-score", action="store_true")
    run_parser.add_argument("--skip-report", action="store_true")
    run_parser.add_argument("--write-speakers", action="store_true")
    run_parser.add_argument("--seed-archive", action="store_true")
    run_parser.add_argument("--force-clean", action="store_true")
    run_parser.add_argument("--force-score", action="store_true")
    run_parser.add_argument("--delay-seconds", type=float, default=3.0)
    run_parser.add_argument("--verbose", action="store_true")

    return parser


def _namespace_to_argv(namespace: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    for key, value in vars(namespace).items():
        if key == "command" or value in (None, False):
            continue
        option = f"--{key.replace('_', '-')}"
        if value is True:
            argv.append(option)
        else:
            argv.extend([option, str(value)])
    return argv


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    forwarded = _namespace_to_argv(args)

    if args.command == "scrape":
        scrape_main(forwarded)
    elif args.command == "clean":
        clean_main(forwarded)
    elif args.command == "build-csv":
        build_csv_main(forwarded)
    elif args.command == "score":
        score_main(forwarded)
    elif args.command == "report":
        report_main(forwarded)
    elif args.command == "prune-legacy":
        prune_legacy_main(forwarded)
    elif args.command == "update-companies":
        update_companies_main(forwarded)
    elif args.command == "run-all":
        run_pipeline_main(forwarded)


if __name__ == "__main__":
    main()
