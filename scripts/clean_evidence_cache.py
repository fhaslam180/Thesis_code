"""Delete evidence cache snapshots that fail the _is_readable_text quality gate.

Safer than a blind grep | xargs rm — only removes files whose content is
confirmed unreadable (binary garbage, sparse PDF output, empty files).

Usage:
    python3 scripts/clean_evidence_cache.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from bor_risk.tools import _is_readable_text  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Remove unreadable evidence cache snapshots.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be deleted without deleting them.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/evidence_cache",
        help="Path to the evidence cache directory (default: data/evidence_cache).",
    )
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"[INFO] Cache directory not found: {cache_dir} — nothing to clean.")
        return

    deleted = 0
    checked = 0
    for f in sorted(cache_dir.rglob("*.txt")):
        checked += 1
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"[WARN] Could not read {f}: {e}")
            continue
        if not _is_readable_text(text):
            if args.dry_run:
                print(f"[DRY-RUN] would delete: {f}")
            else:
                f.unlink()
                print(f"Deleted: {f}")
            deleted += 1

    action = "Would delete" if args.dry_run else "Deleted"
    print(f"\n{action} {deleted}/{checked} unreadable snapshots.")


if __name__ == "__main__":
    main()
