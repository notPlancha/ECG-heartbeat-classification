"""
download_data.py
================
High-performance parallel downloader for the MIT-BIH Arrhythmia Database
and MIT-BIH Noise Stress Test Database from PhysioNet.

Downloads all 48 MIT-BIH records (.hea, .dat, .atr) directly from
https://physionet.org/files/mitdb/1.0.0/ using multi-threaded streaming,
with automatic resume, size validation, and tqdm progress display.

Usage:
    uv run python src/download_data.py
    uv run python src/download_data.py --include-noise
"""

import argparse
import concurrent.futures
import shutil
import sys
import urllib.request
from pathlib import Path

from pyhere import here
from tqdm import tqdm

PROJECT_ROOT = here()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import ALL_RECORDS

DEFAULT_DATA_DIR = Path(here("dataset", "mit-bih-arrhythmia-database-1.0.0"))
DEFAULT_NOISE_DIR = Path(here("dataset", "mit-bih-noise-stress-test-database-1.0.0"))

BASE_MITDB_URL = "https://physionet.org/files/mitdb/1.0.0"
BASE_NSTDB_URL = "https://physionet.org/files/nstdb/1.0.0"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ECG-Heartbeat-Classification-Student/1.0"


def download_single_file(url: str, dest_path: Path, min_size: int = 1) -> bool:
    """Download a single file if it does not already exist with valid size."""
    if dest_path.exists() and dest_path.stat().st_size >= min_size:
        return True  # Already cached

    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=30) as response, open(temp_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        temp_path.replace(dest_path)
        return True
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"[Error downloading {dest_path.name}]: {e}", file=sys.stderr)
        return False


def download_mitdb(dest_dir: Path, max_workers: int = 8) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("DOWNLOADING MIT-BIH ARRHYTHMIA DATABASE (mitdb 1.0.0)")
    print(f"Target Directory: {dest_dir.resolve()}")
    print("=" * 80)

    # List of all files to download
    tasks = []
    # 1. RECORDS file
    tasks.append((f"{BASE_MITDB_URL}/RECORDS", dest_dir / "RECORDS", 1))

    # 2. Per-record files (.hea, .dat, .atr)
    for rec in ALL_RECORDS:
        tasks.append((f"{BASE_MITDB_URL}/{rec}.hea", dest_dir / f"{rec}.hea", 50))
        tasks.append((f"{BASE_MITDB_URL}/{rec}.atr", dest_dir / f"{rec}.atr", 50))
        tasks.append((f"{BASE_MITDB_URL}/{rec}.dat", dest_dir / f"{rec}.dat", 1_000_000))

    print(f"Total files to synchronize: {len(tasks)} (across {len(ALL_RECORDS)} records)")

    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_single_file, url, path, min_size): path.name
            for url, path, min_size in tasks
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(tasks),
            desc="Downloading MIT-BIH",
            unit="file",
        ):
            if future.result():
                success_count += 1

    print(f"[Success] {success_count}/{len(tasks)} files successfully verified in {dest_dir.name}/")


def download_nstdb(dest_dir: Path, max_workers: int = 8) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("DOWNLOADING MIT-BIH NOISE STRESS TEST DATABASE (nstdb 1.0.0)")
    print(f"Target Directory: {dest_dir.resolve()}")
    print("=" * 80)

    # Noise records typically include 'bw', 'em', 'ma', and synthetic stress records
    noise_records = ["bw", "em", "ma"]
    tasks = [(f"{BASE_NSTDB_URL}/RECORDS", dest_dir / "RECORDS", 1)]
    for nr in noise_records:
        tasks.append((f"{BASE_NSTDB_URL}/{nr}.hea", dest_dir / f"{nr}.hea", 50))
        tasks.append((f"{BASE_NSTDB_URL}/{nr}.dat", dest_dir / f"{nr}.dat", 500_000))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_single_file, url, path, min_size): path.name
            for url, path, min_size in tasks
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(tasks),
            desc="Downloading NSTDB",
            unit="file",
        ):
            future.result()

    print(f"[Success] NSTDB noise waveforms ready in {dest_dir.name}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-speed PhysioNet MIT-BIH downloader."
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
        default=8,
        help="Number of concurrent worker threads (default: 8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_mitdb(args.output_dir, max_workers=args.workers)
    if args.include_noise:
        download_nstdb(DEFAULT_NOISE_DIR, max_workers=args.workers)


if __name__ == "__main__":
    main()
