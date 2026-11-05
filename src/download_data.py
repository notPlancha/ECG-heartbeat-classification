"""
download_data.py
================
Parallel, high-throughput downloader for the MIT-BIH Arrhythmia Database
and MIT-BIH Noise Stress Test Database using PhysioNet's official WFDB API.

Features:
- Multi-threaded parallel downloads (ThreadPoolExecutor) for maximum network throughput
- Skips already-downloaded files to enable instant resumption
- Real-time progress bar via tqdm
- Bypasses platform-specific curl/wget incompatibilities on Windows

Usage:
    uv run python src/download_data.py
    uv run python src/download_data.py --include-noise
"""

import argparse
import concurrent.futures
import sys
from pathlib import Path

from pyhere import here
from tqdm import tqdm
import wfdb

DEFAULT_DATA_DIR = Path(here("dataset", "mit-bih-arrhythmia-database-1.0.0"))
DEFAULT_NOISE_DIR = Path(here("dataset", "mit-bih-noise-stress-test-database-1.0.0"))


def download_single_record(db_dir: str, dl_dir: Path, record_name: str) -> str:
    """Download .dat, .hea, and .atr files for a single record if not already present."""
    expected_files = [f"{record_name}.dat", f"{record_name}.hea", f"{record_name}.atr"]
    files_to_dl = [
        f for f in expected_files
        if not ((dl_dir / f).exists() and (dl_dir / f).stat().st_size > 0)
    ]
    if files_to_dl:
        wfdb.dl_files(db_dir, str(dl_dir), files_to_dl)
    return record_name


def download_database_parallel(db_dir: str, dest_dir: Path, max_workers: int = 6) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    records = wfdb.get_record_list(db_dir)
    print(f"[{db_dir.upper()}] Identified {len(records)} records on PhysioNet.")
    
    # Ensure RECORDS file exists
    if not (dest_dir / "RECORDS").exists():
        try:
            wfdb.dl_files(db_dir, str(dest_dir), ["RECORDS"])
        except Exception:
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_single_record, db_dir, dest_dir, r): r
            for r in records
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(records),
            desc=f"Fetching {db_dir}",
            unit="rec",
        ):
            future.result()

    # Quick validation
    valid_count = sum(1 for r in records if (dest_dir / f"{r}.hea").exists() and (dest_dir / f"{r}.dat").exists())
    print(f"[{db_dir.upper()}] Download verified: {valid_count}/{len(records)} records available in {dest_dir.name}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel PhysioNet record downloader via official WFDB API."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Target folder for MIT-BIH Arrhythmia Database.",
    )
    parser.add_argument(
        "--include-noise",
        action="store_true",
        help="Also download the MIT-BIH Noise Stress Test Database (nstdb).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of concurrent worker threads (default: 6).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("PHYSIOMET DATABASE SYNCHRONIZATION")
    print("=" * 80)
    download_database_parallel("mitdb", args.output_dir, max_workers=args.workers)
    if args.include_noise:
        download_database_parallel("nstdb", DEFAULT_NOISE_DIR, max_workers=args.workers)
    print("=" * 80)
    print("[Done] All requested databases are ready.")
    print("=" * 80)


if __name__ == "__main__":
    main()
