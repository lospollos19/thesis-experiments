"""Fold testmon's write-ahead log into the database file before archiving it.

`.testmondata` is SQLite in WAL mode. Straight after a run the main file holds ~4 KB
and `.testmondata-wal` holds the content — measured at 288 KB for this suite. Archiving
the main file alone therefore ships an empty database, and testmon responds by treating
every test as unknown and re-running the whole suite: no error, no warning, a full
re-run reported as if it were a selection.

Carrying all three files works, but a checkpoint removes the failure class instead of
handling it. `wal_checkpoint(TRUNCATE)` folds the log back in and leaves a single file
to publish. Checksums are still worth taking: this guards against the WAL, not against
a truncated download.

    python tools/checkpoint_testmon_db.py .testmondata
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def checkpoint(path: Path) -> tuple[int, int, int]:
    """Return SQLite's ``(busy, wal_pages, checkpointed_pages)``.

    ``busy`` non-zero means another connection held the database and the log was not
    fully folded in — the file would then be incomplete, so callers must treat it as
    a failure rather than archive it anyway.
    """
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    finally:
        conn.close()
    return tuple(int(v) for v in row)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, nargs="?", default=Path(".testmondata"))
    args = parser.parse_args(argv)

    if not args.database.is_file():
        print(f"no database at {args.database}")
        return 1

    before = {p.name: p.stat().st_size for p in sorted(args.database.parent.glob(".testmondata*"))}
    busy, wal_pages, done = checkpoint(args.database)
    after = {p.name: p.stat().st_size for p in sorted(args.database.parent.glob(".testmondata*"))}

    print(f"before: {before}")
    print(f"checkpoint: busy={busy} wal_pages={wal_pages} checkpointed={done}")
    print(f"after:  {after}")

    if busy:
        print("checkpoint was blocked; the database file is not self-contained")
        return 1
    leftovers = [n for n in after if n.endswith(("-wal", "-shm")) and after[n] > 0]
    if leftovers:
        print(f"write-ahead log still present and non-empty: {leftovers}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
