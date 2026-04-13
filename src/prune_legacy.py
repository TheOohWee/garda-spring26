from __future__ import annotations

import shutil
from pathlib import Path

from common import DATA_DIR, SPEAKERS_DIR, TRANSCRIPTS_DIR, configure_logging, ensure_directories, parse_args


def move_legacy_files(source_dir: Path, backup_dir: Path) -> int:
    moved = 0
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.glob("*"):
        if not path.is_file():
            continue
        target = backup_dir / path.name
        if target.exists():
            target.unlink()
        shutil.move(str(path), str(target))
        moved += 1
    return moved


def main(argv: list[str] | None = None) -> None:
    parser = parse_args("Move old flat transcript/speaker files into a backup folder.")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    ensure_directories()

    backup_root = DATA_DIR / "legacy_flat_backup"
    moved_transcripts = move_legacy_files(TRANSCRIPTS_DIR, backup_root / "transcripts")
    moved_speakers = move_legacy_files(SPEAKERS_DIR, backup_root / "speakers")
    print(
        f"Moved {moved_transcripts} legacy transcript files and {moved_speakers} legacy speaker files to {backup_root}"
    )


if __name__ == "__main__":
    main()
