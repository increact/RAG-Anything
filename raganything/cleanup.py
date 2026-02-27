"""
Output directory cleanup — removes stale processing artefacts.

Files and sub-directories inside OUTPUT_DIR are deleted once they are older
than OUTPUT_FILE_TTL_DAYS (default: 7 days).  A background asyncio task
runs the cleanup every OUTPUT_CLEANUP_INTERVAL_HOURS (default: 24 hours),
with an initial run shortly after startup.

Directory layout produced by the parsers
-----------------------------------------
MinerU (most common):
    output/
        tmpXXXXXX/               ← top-level dir named after the temp file stem
            auto/
                tmpXXXXXX.md
                tmpXXXXXX_content_list.json
                tmpXXXXXX_middle.json
                images/
                    fig_0.png
                    ...

Docling / simple output:
    output/
        tmpXXXXXX.md             ← single flat .md file

Result store (added by result_store.py):
    output/
        <doc_id>_result.json     ← persisted result for GET /api/v1/result/{doc_id}

Cleanup strategy
----------------
Walk the *immediate* children of output_dir only (no deep recursion).
Each child is tested by its last-modified time (mtime on the entry itself;
for directories we use the most recently modified file inside them to avoid
false-positives while a job is still writing).  If the entry is older than
the configured TTL it is removed — files with os.remove, directories with
shutil.rmtree.
"""

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class CleanupStats:
    deleted_files: int = 0
    deleted_dirs: int = 0
    freed_bytes: int = 0
    errors: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        mb = self.freed_bytes / 1024 / 1024
        return (
            f"deleted {self.deleted_files} files, {self.deleted_dirs} dirs, "
            f"freed {mb:.1f} MB"
            + (f", {len(self.errors)} error(s)" if self.errors else "")
        )


def _dir_newest_mtime(path: Path) -> float:
    """Return the newest mtime of any file inside *path* (non-recursive)."""
    newest = path.stat().st_mtime
    try:
        for child in path.rglob("*"):
            if child.is_file():
                newest = max(newest, child.stat().st_mtime)
    except OSError:
        pass
    return newest


def cleanup_output_directory(output_dir: str, ttl_days: int) -> CleanupStats:
    """
    Synchronously remove stale entries from *output_dir*.

    An entry is considered stale when its effective mtime is older than
    ``now - ttl_days``.  For directories the effective mtime is the newest
    mtime among all contained files (to avoid deleting directories while a
    worker is still writing into them).

    Returns a :class:`CleanupStats` summary.
    """
    stats = CleanupStats()
    output_path = Path(output_dir)

    if not output_path.exists():
        logger.debug(f"Output dir does not exist yet, skipping cleanup: {output_dir}")
        return stats

    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    cutoff_ts = cutoff.timestamp()

    logger.info(
        f"Running output cleanup: dir={output_dir}, ttl={ttl_days}d, "
        f"cutoff={cutoff.strftime('%Y-%m-%d %H:%M UTC')}"
    )

    for entry in output_path.iterdir():
        try:
            if entry.is_file():
                mtime = entry.stat().st_mtime
                size = entry.stat().st_size
                if mtime < cutoff_ts:
                    entry.unlink()
                    stats.deleted_files += 1
                    stats.freed_bytes += size
                    logger.debug(f"Deleted file: {entry.name}")

            elif entry.is_dir():
                # Use newest mtime inside the directory so we don't remove
                # a directory while the worker is still writing to it.
                effective_mtime = _dir_newest_mtime(entry)
                if effective_mtime < cutoff_ts:
                    dir_size = sum(
                        f.stat().st_size
                        for f in entry.rglob("*")
                        if f.is_file()
                    )
                    shutil.rmtree(entry)
                    stats.deleted_dirs += 1
                    stats.freed_bytes += dir_size
                    logger.debug(f"Deleted directory: {entry.name}/")

        except OSError as e:
            msg = f"Could not remove {entry}: {e}"
            stats.errors.append(msg)
            logger.warning(msg)

    logger.info(f"Output cleanup complete: {stats}")
    return stats


async def run_cleanup_loop(
    output_dir: str,
    ttl_days: int,
    interval_hours: float,
    initial_delay_seconds: float = 60.0,
) -> None:
    """
    Asyncio background loop that periodically calls :func:`cleanup_output_directory`.

    Args:
        output_dir:            Path to the output directory.
        ttl_days:              Files older than this are removed.
        interval_hours:        How often to run (in hours).
        initial_delay_seconds: Wait this long after startup before the first
                               run (avoids competing with service initialisation).
    """
    logger.info(
        f"Output cleanup loop started — ttl={ttl_days}d, "
        f"interval={interval_hours}h, first run in {initial_delay_seconds}s"
    )

    await asyncio.sleep(initial_delay_seconds)

    while True:
        try:
            # Run the blocking I/O in a thread-pool so the event loop stays free
            await asyncio.to_thread(cleanup_output_directory, output_dir, ttl_days)
        except asyncio.CancelledError:
            logger.info("Output cleanup loop cancelled — shutting down")
            return
        except Exception as e:
            logger.error(f"Unexpected error in cleanup loop: {e}", exc_info=True)

        try:
            await asyncio.sleep(interval_hours * 3600)
        except asyncio.CancelledError:
            logger.info("Output cleanup loop cancelled — shutting down")
            return
