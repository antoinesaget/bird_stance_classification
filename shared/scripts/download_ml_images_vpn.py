#!/usr/bin/env python3
"""Download Macaulay Library images from a CSV export with Mullvad VPN rotation.

This script expects an ML export CSV with an "ML Catalog Number" column.
It downloads image assets via:
https://cdn.download.ams.birds.cornell.edu/api/v2/asset/<ML_ID>/<SIZE>

VPN Rotation Feature:
When --use-mullvad-vpn is enabled, the script will automatically rotate
VPN servers via Mullvad CLI when cooldown is triggered, instead of just
waiting. This helps bypass rate limiting by changing your IP address.

Requirements:
- Mullvad VPN subscription
- Mullvad CLI installed (mullvad command available)
- Active Mullvad connection
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


ASSET_URL_TEMPLATE = "https://cdn.download.ams.birds.cornell.edu/api/v2/asset/{ml_id}/{size}"
DEFAULT_ID_COLUMN = "ML Catalog Number"
DEFAULT_RETRYABLE_STATUS_CODES = {403, 408, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class DownloadTask:
    row_index: int
    ml_id: str
    url: str
    output_file: Path


@dataclass(frozen=True)
class DownloadResult:
    task: DownloadTask
    ok: bool
    status_code: int | None
    bytes_written: int
    attempts: int
    error: str | None
    skipped: bool


@dataclass
class MullvadLocation:
    """Represents a Mullvad relay location."""
    country: str
    country_code: str
    city: str | None = None
    city_code: str | None = None
    host: str | None = None


class MullvadVPNManager:
    """Manages Mullvad VPN connections and IP rotation."""

    def __init__(
        self,
        enabled: bool = False,
        preferred_countries: Optional[list[str]] = None,
        rotation_delay: float = 5.0,
        max_rotations: int = 100,
    ) -> None:
        self.enabled = enabled
        self.preferred_countries = preferred_countries or []
        self.rotation_delay = rotation_delay
        self.max_rotations = max_rotations
        self._lock = threading.Lock()
        self._available_locations: list[MullvadLocation] = []
        self._used_locations: list[MullvadLocation] = field(default_factory=list)
        self._rotation_count = 0
        self._current_location: MullvadLocation | None = None
        self._initialized = False

    def _run_mullvad_command(self, args: list[str]) -> tuple[bool, str, str]:
        """Execute a mullvad CLI command."""
        try:
            result = subprocess.run(
                ["mullvad"] + args,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except FileNotFoundError:
            return False, "", "mullvad command not found. Is Mullvad CLI installed?"
        except Exception as e:
            return False, "", str(e)

    def _parse_relay_list(self, output: str) -> list[MullvadLocation]:
        """Parse mullvad relay list output into location objects."""
        locations = []
        current_country = None
        current_country_code = None

        for line in output.strip().split("\n"):
            # Match country lines (e.g., "United States (us)")
            country_match = re.match(r"^([^(]+)\s+\(([a-z]{2})\)\s*$", line)
            if country_match:
                current_country = country_match.group(1).strip()
                current_country_code = country_match.group(2)
                locations.append(MullvadLocation(
                    country=current_country,
                    country_code=current_country_code
                ))
                continue

            # Match city lines with indentation (e.g., "  New York (nyc)")
            city_match = re.match(r"^\s+([^(]+)\s+\(([a-z0-9]+)\)\s*$", line)
            if city_match and current_country:
                city = city_match.group(1).strip()
                city_code = city_match.group(2)
                locations.append(MullvadLocation(
                    country=current_country,
                    country_code=current_country_code,
                    city=city,
                    city_code=city_code
                ))

        return locations

    def _get_available_locations(self) -> list[MullvadLocation]:
        """Fetch and cache available Mullvad locations."""
        if self._available_locations:
            return self._available_locations

        success, stdout, stderr = self._run_mullvad_command(["relay", "list"])
        if not success:
            raise RuntimeError(f"Failed to get relay list: {stderr}")

        self._available_locations = self._parse_relay_list(stdout)
        return self._available_locations

    def _get_current_location(self) -> MullvadLocation | None:
        """Get current VPN location from mullvad status."""
        success, stdout, stderr = self._run_mullvad_command(["status", "-v"])
        if not success:
            return None

        # Parse status output to find current location
        # Example: "Connected to us-nyc-wg-001 in New York, USA"
        connected_match = re.search(
            r"Connected to\s+(\w+-\w+-\w+)\s+in\s+([^,]+),\s+(\w+)",
            stdout
        )
        if connected_match:
            return MullvadLocation(
                country=connected_match.group(3),
                city=connected_match.group(2)
            )
        return None

    def _set_location(self, location: MullvadLocation) -> bool:
        """Set VPN location via mullvad CLI."""
        args = ["relay", "set", "location"]
        args.append(location.country_code)
        if location.city_code:
            args.append(location.city_code)

        success, _, stderr = self._run_mullvad_command(args)
        if not success:
            return False

        # Trigger reconnect
        success, _, stderr = self._run_mullvad_command(["connect"])
        return success

    def initialize(self) -> bool:
        """Check if Mullvad is available and working."""
        if not self.enabled:
            return False

        with self._lock:
            if self._initialized:
                return True

            # Check if mullvad is installed
            success, stdout, stderr = self._run_mullvad_command(["status"])
            if not success:
                return False

            # Check if connected
            if "Connected" not in stdout:
                return False

            # Load available locations
            try:
                self._get_available_locations()
                self._current_location = self._get_current_location()
                self._initialized = True
                return True
            except Exception:
                return False

    def rotate_ip(self) -> tuple[bool, str]:
        """Rotate to a new VPN server."""
        with self._lock:
            if not self._initialized:
                return False, "VPN manager not initialized"

            if self._rotation_count >= self.max_rotations:
                return False, f"Max rotations ({self.max_rotations}) reached"

            locations = self._available_locations
            if not locations:
                return False, "No available locations"

            # Filter by preferred countries if specified
            if self.preferred_countries:
                filtered = [
                    loc for loc in locations
                    if loc.country_code in self.preferred_countries
                    or loc.country in self.preferred_countries
                ]
                if filtered:
                    locations = filtered

            # Pick a random location different from current
            available = [loc for loc in locations if loc != self._current_location]
            if not available:
                available = locations

            new_location = random.choice(available)

            # Set new location
            if not self._set_location(new_location):
                return False, f"Failed to set location to {new_location.country}"

            # Wait for connection to establish
            time.sleep(self.rotation_delay)

            # Verify connection
            success, stdout, _ = self._run_mullvad_command(["status"])
            if not success or "Connected" not in stdout:
                return False, "Failed to verify connection after rotation"

            self._current_location = new_location
            self._rotation_count += 1
            self._used_locations.append(new_location)

            location_str = new_location.country
            if new_location.city:
                location_str += f" ({new_location.city})"

            return True, location_str

    def get_status(self) -> dict:
        """Get current VPN status."""
        with self._lock:
            status = {
                "enabled": self.enabled,
                "initialized": self._initialized,
                "rotation_count": self._rotation_count,
                "max_rotations": self.max_rotations,
                "current_location": None,
                "available_locations": len(self._available_locations),
            }

            if self._current_location:
                status["current_location"] = {
                    "country": self._current_location.country,
                    "city": self._current_location.city,
                }

            return status


class GlobalRateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            self._interval = 0.0
        else:
            self._interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self._interval <= 0:
            return
        sleep_for = 0.0
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                sleep_for = self._next_allowed - now
            anchor = max(now, self._next_allowed)
            self._next_allowed = anchor + self._interval
        if sleep_for > 0:
            time.sleep(sleep_for)


class ErrorTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._consecutive_403 = 0

    def record(self, result: DownloadResult) -> int:
        with self._lock:
            if result.ok:
                self._consecutive_403 = 0
            elif result.status_code == 403 or result.error == "http_403":
                self._consecutive_403 += 1
            else:
                self._consecutive_403 = 0
            return self._consecutive_403

    def reset(self) -> None:
        with self._lock:
            self._consecutive_403 = 0

    def current(self) -> int:
        with self._lock:
            return self._consecutive_403


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download images from an ML export CSV by catalog number with optional VPN rotation."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to input CSV (expects ML Catalog Number column).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where image files will be written.",
    )
    parser.add_argument(
        "--id-column",
        default=DEFAULT_ID_COLUMN,
        help=f"CSV column containing ML asset IDs (default: {DEFAULT_ID_COLUMN}).",
    )
    parser.add_argument(
        "--size",
        default="1200",
        help="Asset size parameter for the ML API URL (default: 1200).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of data rows to skip from start (default: 0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of rows to process (default: all).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel download workers (default: 4).",
    )
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=2.0,
        help="Global request budget across all workers (default: 2.0).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Per-request timeout in seconds (default: 20).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Retry attempts after the first try (default: 4).",
    )
    parser.add_argument(
        "--backoff-seconds",
        type=float,
        default=1.0,
        help="Base delay (seconds) for exponential backoff (default: 1.0).",
    )
    parser.add_argument(
        "--max-backoff-seconds",
        type=float,
        default=20.0,
        help="Cap for backoff delay in seconds (default: 20.0).",
    )
    parser.add_argument(
        "--retry-status-codes",
        default="403,408,425,429,500,502,503,504",
        help=(
            "Comma-separated HTTP status codes that should be retried "
            "(default: 403,408,425,429,500,502,503,504)."
        ),
    )
    parser.add_argument(
        "--consecutive-403-threshold",
        type=int,
        default=40,
        help="Trigger cooldown/VPN rotation after this many consecutive 403 results (default: 40).",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=120.0,
        help="Cooldown duration after a 403 storm is detected when VPN rotation is disabled (default: 120).",
    )
    parser.add_argument(
        "--max-cooldowns",
        type=int,
        default=5,
        help="Abort run after this many cooldown cycles (default: 5).",
    )
    parser.add_argument(
        "--cooldown-log-interval",
        type=float,
        default=10.0,
        help="Seconds between cooldown remaining logs (default: 10).",
    )
    parser.add_argument(
        "--log-403-every",
        type=int,
        default=10,
        help="Log each Nth consecutive 403 event (default: 10).",
    )
    parser.add_argument(
        "--retry-failures-only",
        action="store_true",
        help=(
            "Only process unresolved rows from failures.csv "
            "(failed rows minus rows present in manifest.csv)."
        ),
    )
    parser.add_argument(
        "--skip-status-codes",
        default="404",
        help=(
            "Comma-separated HTTP status codes to exclude when "
            "--retry-failures-only is used (default: 404)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if output file already exists.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Success manifest CSV path (default: <output-dir>/manifest.csv).",
    )
    parser.add_argument(
        "--failures",
        default=None,
        help="Failure CSV path (default: <output-dir>/failures.csv).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N completed tasks (default: 25).",
    )
    # VPN-related arguments
    parser.add_argument(
        "--use-mullvad-vpn",
        action="store_true",
        help="Enable Mullvad VPN IP rotation on cooldown (default: disabled).",
    )
    parser.add_argument(
        "--vpn-countries",
        default="de,nl,se,ch,fr,at,es,it,gb,fi,ie,pl,dk,no,cz,be,pt,ro,gr,hu",
        help=(
            "Comma-separated list of preferred country codes for VPN rotation "
            "(default: European countries). Use 'us' for US-only, or 'us,ca,gb,de' for mix."
        ),
    )
    parser.add_argument(
        "--vpn-rotation-delay",
        type=float,
        default=5.0,
        help="Seconds to wait after VPN rotation for connection to stabilize (default: 5).",
    )
    parser.add_argument(
        "--vpn-max-rotations",
        type=int,
        default=100,
        help="Maximum number of VPN rotations before aborting (default: 100).",
    )
    parser.add_argument(
        "--vpn-min-rotation-interval",
        type=float,
        default=30.0,
        help="Minimum seconds between VPN rotations to prevent rapid switching (default: 30).",
    )
    return parser.parse_args()


def compute_backoff(
    attempt: int,
    base_seconds: float,
    max_backoff_seconds: float,
) -> float:
    exp_delay = base_seconds * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0.0, base_seconds * 0.25)
    return min(max_backoff_seconds, exp_delay + jitter)


def log(level: str, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)


def sleep_with_countdown(total_seconds: float, log_interval_seconds: float) -> None:
    if total_seconds <= 0:
        return
    end_time = time.monotonic() + total_seconds
    interval = log_interval_seconds if log_interval_seconds > 0 else total_seconds
    while True:
        remaining = end_time - time.monotonic()
        if remaining <= 0:
            break
        log("INFO", f"Cooldown active, {remaining:.1f}s remaining.")
        time.sleep(min(interval, remaining))
    log("INFO", "Cooldown complete, resuming downloads.")


def load_row_index_set(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    indices: set[int] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        index_column = None
        if "row_index" in fieldnames:
            index_column = "row_index"
        elif "index" in fieldnames:
            index_column = "index"
        if index_column is None:
            return set()
        for row in reader:
            raw_value = (row.get(index_column) or "").strip()
            if not raw_value:
                continue
            try:
                indices.add(int(raw_value))
            except ValueError:
                continue
    return indices


def load_failure_status_map(csv_path: Path) -> dict[int, int | None]:
    if not csv_path.exists():
        return {}

    status_by_row: dict[int, int | None] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        index_column = None
        if "row_index" in fieldnames:
            index_column = "row_index"
        elif "index" in fieldnames:
            index_column = "index"

        if index_column is None:
            return {}

        for row in reader:
            raw_index = (row.get(index_column) or "").strip()
            if not raw_index:
                continue
            try:
                row_index = int(raw_index)
            except ValueError:
                continue

            raw_status = (row.get("status_code") or "").strip()
            try:
                status_by_row[row_index] = int(raw_status) if raw_status else None
            except ValueError:
                status_by_row[row_index] = None

    return status_by_row


def load_tasks(
    csv_path: Path,
    id_column: str,
    size: str,
    output_dir: Path,
    offset: int,
    limit: int | None,
    row_filter: set[int] | None = None,
) -> list[DownloadTask]:
    tasks: list[DownloadTask] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if id_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Column '{id_column}' not found in CSV. Available columns: {reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader, start=1):
            if row_idx <= offset:
                continue
            if row_filter is not None and row_idx not in row_filter:
                continue
            ml_id = (row.get(id_column) or "").strip()
            if not ml_id:
                continue

            url = ASSET_URL_TEMPLATE.format(ml_id=ml_id, size=size)
            out_name = f"{row_idx:05d}_{ml_id}.jpg"
            tasks.append(
                DownloadTask(
                    row_index=row_idx,
                    ml_id=ml_id,
                    url=url,
                    output_file=output_dir / out_name,
                )
            )
            if limit is not None and len(tasks) >= limit:
                break
    return tasks


def download_one(
    task: DownloadTask,
    timeout: float,
    retries: int,
    backoff_seconds: float,
    max_backoff_seconds: float,
    retryable_status_codes: set[int],
    rate_limiter: GlobalRateLimiter,
    overwrite: bool,
) -> DownloadResult:
    if task.output_file.exists() and task.output_file.stat().st_size > 0 and not overwrite:
        return DownloadResult(
            task=task,
            ok=True,
            status_code=None,
            bytes_written=task.output_file.stat().st_size,
            attempts=0,
            error=None,
            skipped=True,
        )

    task.output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = task.output_file.with_suffix(task.output_file.suffix + ".part")
    last_error = None
    last_status_code: int | None = None
    attempts_total = retries + 1

    for attempt in range(1, attempts_total + 1):
        try:
            rate_limiter.wait()
            req = urllib.request.Request(
                task.url,
                headers={"User-Agent": "bird-leg-ml-downloader/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = response.getcode()
                if status < 200 or status >= 300:
                    raise urllib.error.HTTPError(
                        task.url,
                        status,
                        f"HTTP {status}",
                        hdrs=response.headers,
                        fp=None,
                    )

                bytes_written = 0
                with tmp_path.open("wb") as out:
                    while True:
                        chunk = response.read(1024 * 64)
                        if not chunk:
                            break
                        out.write(chunk)
                        bytes_written += len(chunk)

                if bytes_written <= 0:
                    raise RuntimeError("empty_response")

                tmp_path.replace(task.output_file)
                return DownloadResult(
                    task=task,
                    ok=True,
                    status_code=status,
                    bytes_written=bytes_written,
                    attempts=attempt,
                    error=None,
                    skipped=False,
                )
        except urllib.error.HTTPError as exc:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            last_error = f"http_{exc.code}"
            last_status_code = exc.code
            retry_after = None
            if exc.headers is not None:
                retry_after = exc.headers.get("Retry-After")
            if 400 <= exc.code < 500 and exc.code not in retryable_status_codes:
                return DownloadResult(
                    task=task,
                    ok=False,
                    status_code=exc.code,
                    bytes_written=0,
                    attempts=attempt,
                    error=last_error,
                    skipped=False,
                )
            if retry_after:
                try:
                    retry_after_seconds = float(retry_after.strip())
                    if retry_after_seconds > 0:
                        time.sleep(min(max_backoff_seconds, retry_after_seconds))
                except ValueError:
                    pass
        except Exception as exc:  # noqa: BLE001
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            last_error = type(exc).__name__

        if attempt < attempts_total:
            time.sleep(compute_backoff(attempt, backoff_seconds, max_backoff_seconds))

    return DownloadResult(
        task=task,
        ok=False,
        status_code=last_status_code,
        bytes_written=0,
        attempts=attempts_total,
        error=last_error or "unknown_error",
        skipped=False,
    )


def ensure_csv_header(path: Path, header: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + "\n", encoding="utf-8")


def handle_cooldown_or_rotation(
    error_tracker: ErrorTracker,
    vpn_manager: MullvadVPNManager,
    args: argparse.Namespace,
    cooldowns_used: int,
    last_rotation_time: float,
) -> tuple[bool, bool, int, float]:  # (handled, aborted, new_cooldowns, new_last_rotation_time)
    """
    Handle cooldown logic with optional VPN rotation.
    Returns: (was_handled, was_aborted, updated_cooldowns_count, updated_last_rotation_time)
    """
    new_cooldowns = cooldowns_used + 1

    # Check if VPN rotation is available and ready
    if vpn_manager.enabled and vpn_manager.initialize():
        # Check minimum rotation interval
        now = time.monotonic()
        time_since_last_rotation = now - last_rotation_time

        if time_since_last_rotation < args.vpn_min_rotation_interval:
            # Too soon, just do a short cooldown
            remaining = args.vpn_min_rotation_interval - time_since_last_rotation
            log(
                "INFO",
                f"VPN rotation throttled, waiting {remaining:.1f}s before next rotation."
            )
            sleep_with_countdown(remaining, args.cooldown_log_interval)
            error_tracker.reset()
            return True, False, new_cooldowns, last_rotation_time

        # Perform VPN rotation
        log(
            "WARN",
            f"403 storm detected ({error_tracker.current()} consecutive). "
            f"Rotating VPN (rotation {new_cooldowns}/{args.max_cooldowns})..."
        )

        success, message = vpn_manager.rotate_ip()
        if success:
            log("INFO", f"VPN rotated successfully to: {message}")
            error_tracker.reset()
            return True, False, new_cooldowns, now
        else:
            log("ERROR", f"VPN rotation failed: {message}")
            # Fall back to cooldown
            log("INFO", f"Falling back to cooldown for {args.cooldown_seconds}s...")
            sleep_with_countdown(args.cooldown_seconds, args.cooldown_log_interval)
            error_tracker.reset()
            return True, False, new_cooldowns, last_rotation_time
    else:
        # Standard cooldown without VPN
        log(
            "WARN",
            f"403 storm detected ({error_tracker.current()} consecutive). "
            f"Starting cooldown {new_cooldowns}/{args.max_cooldowns} for {args.cooldown_seconds}s."
        )
        sleep_with_countdown(args.cooldown_seconds, args.cooldown_log_interval)
        error_tracker.reset()
        return True, False, new_cooldowns, last_rotation_time


def main() -> int:
    args = parse_args()
    started_at = time.monotonic()

    csv_path = Path(args.csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else output_dir / "manifest.csv"
    )
    failures_path = (
        Path(args.failures).expanduser().resolve()
        if args.failures
        else output_dir / "failures.csv"
    )

    # Validate arguments
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0 when set")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.retries < 0:
        raise ValueError("--retries must be >= 0")
    if args.requests_per_second < 0:
        raise ValueError("--requests-per-second must be >= 0")
    if args.consecutive_403_threshold <= 0:
        raise ValueError("--consecutive-403-threshold must be > 0")
    if args.cooldown_seconds < 0:
        raise ValueError("--cooldown-seconds must be >= 0")
    if args.max_cooldowns < 0:
        raise ValueError("--max-cooldowns must be >= 0")
    if args.cooldown_log_interval <= 0:
        raise ValueError("--cooldown-log-interval must be > 0")
    if args.log_403_every <= 0:
        raise ValueError("--log-403-every must be > 0")

    # Parse retry status codes
    retryable_status_codes: set[int] = set()
    for item in args.retry_status_codes.split(","):
        item = item.strip()
        if not item:
            continue
        retryable_status_codes.add(int(item))
    if not retryable_status_codes:
        retryable_status_codes = set(DEFAULT_RETRYABLE_STATUS_CODES)

    # Parse skip status codes
    skip_status_codes: set[int] = set()
    for item in args.skip_status_codes.split(","):
        item = item.strip()
        if not item:
            continue
        skip_status_codes.add(int(item))

    # Parse VPN countries
    vpn_countries = []
    if args.vpn_countries:
        vpn_countries = [c.strip().lower() for c in args.vpn_countries.split(",") if c.strip()]

    # Initialize VPN manager
    vpn_manager = MullvadVPNManager(
        enabled=args.use_mullvad_vpn,
        preferred_countries=vpn_countries,
        rotation_delay=args.vpn_rotation_delay,
        max_rotations=args.vpn_max_rotations,
    )

    if args.use_mullvad_vpn:
        log("INFO", "Mullvad VPN rotation enabled")
        if not vpn_manager.initialize():
            log(
                "WARN",
                "Failed to initialize Mullvad VPN. "
                "Make sure mullvad CLI is installed and you're connected. "
                "Falling back to standard cooldown."
            )
        else:
            status = vpn_manager.get_status()
            log("INFO", f"VPN initialized. Available locations: {status['available_locations']}")
            if status['current_location']:
                loc = status['current_location']
                log("INFO", f"Current location: {loc['country']} ({loc.get('city', 'N/A')})")

    # Load row filter for retry mode
    row_filter: set[int] | None = None
    if args.retry_failures_only:
        if not failures_path.exists():
            log(
                "ERROR",
                f"--retry-failures-only requested but failures file does not exist: {failures_path}",
            )
            return 1
        failed_rows = load_row_index_set(failures_path)
        resolved_rows = load_row_index_set(manifest_path)
        unresolved_rows = failed_rows - resolved_rows
        failure_status_map = load_failure_status_map(failures_path)

        status_excluded = 0
        if skip_status_codes:
            kept_rows: set[int] = set()
            for row_index in unresolved_rows:
                status = failure_status_map.get(row_index)
                if status is not None and status in skip_status_codes:
                    status_excluded += 1
                    continue
                kept_rows.add(row_index)
            row_filter = kept_rows
        else:
            row_filter = unresolved_rows

        log(
            "INFO",
            "Retry mode enabled: "
            f"failed_rows={len(failed_rows)} resolved_rows={len(resolved_rows)} "
            f"unresolved_rows={len(unresolved_rows)} "
            f"status_excluded={status_excluded} retry_rows={len(row_filter)} "
            f"skip_status_codes={sorted(skip_status_codes)}",
        )
        if not row_filter:
            log("INFO", "No unresolved failed rows found. Nothing to do.")
            return 0

    # Load download tasks
    tasks = load_tasks(
        csv_path=csv_path,
        id_column=args.id_column,
        size=args.size,
        output_dir=output_dir,
        offset=args.offset,
        limit=args.limit,
        row_filter=row_filter,
    )
    if not tasks:
        log("ERROR", "No tasks found from CSV after applying offset/limit/filter.")
        return 1

    # Ensure CSV headers exist
    ensure_csv_header(
        manifest_path,
        "row_index,ml_catalog_number,url,file,bytes,attempts,skipped",
    )
    ensure_csv_header(
        failures_path,
        "row_index,ml_catalog_number,url,error,attempts,status_code",
    )

    # Initialize statistics
    lock = threading.Lock()
    completed = 0
    ok = 0
    failed = 0
    skipped = 0
    total_bytes = 0

    log("INFO", f"CSV: {csv_path}")
    log("INFO", f"Output directory: {output_dir}")
    log("INFO", f"Tasks queued: {len(tasks)}")
    log(
        "INFO",
        "Settings: "
        f"workers={args.workers}, rps={args.requests_per_second}, timeout={args.timeout}s, "
        f"retries={args.retries}, size={args.size}, retry_failures_only={args.retry_failures_only}, "
        f"vpn_enabled={args.use_mullvad_vpn}"
    )

    # Initialize rate limiter and error tracker
    rate_limiter = GlobalRateLimiter(args.requests_per_second)
    error_tracker = ErrorTracker()
    cooldowns_used = 0
    aborted_due_to_403_storm = False
    last_rotation_time = 0.0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        task_iter = iter(tasks)
        in_flight = set()

        def submit_next() -> None:
            try:
                task = next(task_iter)
            except StopIteration:
                return
            in_flight.add(
                pool.submit(
                    download_one,
                    task,
                    args.timeout,
                    args.retries,
                    args.backoff_seconds,
                    args.max_backoff_seconds,
                    retryable_status_codes,
                    rate_limiter,
                    args.overwrite,
                )
            )

        # Initial task submission
        for _ in range(min(args.workers, len(tasks))):
            submit_next()

        # Main download loop
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                in_flight.remove(future)
                result = future.result()
                previous_403_streak = error_tracker.current()
                consecutive_403 = error_tracker.record(result)

                # Log 403 streak reset
                if previous_403_streak > 0 and consecutive_403 == 0:
                    log("INFO", f"403 streak reset after reaching {previous_403_streak}.")

                # Log 403 events
                if not result.ok and (result.status_code == 403 or result.error == "http_403"):
                    remaining_before_cooldown = max(
                        0, args.consecutive_403_threshold - consecutive_403
                    )
                    if (
                        consecutive_403 == 1
                        or consecutive_403 % args.log_403_every == 0
                        or remaining_before_cooldown == 0
                    ):
                        log(
                            "WARN",
                            "HTTP 403 received "
                            f"(row={result.task.row_index}, ml_id={result.task.ml_id}, "
                            f"attempts={result.attempts}); "
                            f"streak={consecutive_403}/{args.consecutive_403_threshold}, "
                            f"remaining_before_cooldown={remaining_before_cooldown}",
                        )

                # Update statistics
                with lock:
                    completed += 1
                    if result.ok:
                        ok += 1
                        if result.skipped:
                            skipped += 1
                        else:
                            total_bytes += result.bytes_written
                            with manifest_path.open("a", encoding="utf-8") as mf:
                                mf.write(
                                    f"{result.task.row_index},{result.task.ml_id},{result.task.url},"
                                    f"{result.task.output_file},{result.bytes_written},{result.attempts},"
                                    f"{int(result.skipped)}\n"
                                )
                    else:
                        failed += 1
                        with failures_path.open("a", encoding="utf-8") as ff:
                            ff.write(
                                f"{result.task.row_index},{result.task.ml_id},{result.task.url},"
                                f"{result.error},{result.attempts},{result.status_code}\n"
                            )

                    # Log progress
                    if completed % args.progress_every == 0 or completed == len(tasks):
                        log(
                            "INFO",
                            f"Progress: {completed}/{len(tasks)} | ok={ok} failed={failed} "
                            f"skipped={skipped} | consecutive_403={consecutive_403} "
                            f"| cooldowns={cooldowns_used}/{args.max_cooldowns}"
                        )

                # Check for 403 storm and handle cooldown/rotation
                if consecutive_403 >= args.consecutive_403_threshold:
                    handled, aborted, cooldowns_used, last_rotation_time = handle_cooldown_or_rotation(
                        error_tracker,
                        vpn_manager,
                        args,
                        cooldowns_used,
                        last_rotation_time,
                    )

                    if cooldowns_used >= args.max_cooldowns:
                        log(
                            "ERROR",
                            "Reached max cooldown cycles. Aborting early to avoid burning "
                            "through remaining queue while blocked.",
                        )
                        aborted_due_to_403_storm = True
                        break

                # Submit next task
                submit_next()

            if aborted_due_to_403_storm:
                break

    # Final summary
    if aborted_due_to_403_storm:
        log(
            "WARN",
            "Aborted early due to persistent 403 responses. "
            "Rerun later to resume from remaining files."
        )

    # Log VPN stats if used
    if args.use_mullvad_vpn:
        vpn_status = vpn_manager.get_status()
        log("INFO", f"VPN rotations performed: {vpn_status['rotation_count']}")

    elapsed = time.monotonic() - started_at
    log("INFO", "Done.")
    log("INFO", f"Successful: {ok}")
    log("INFO", f"Failed: {failed}")
    log("INFO", f"Skipped(existing): {skipped}")
    log("INFO", f"Downloaded bytes this run: {total_bytes}")
    log("INFO", f"Elapsed: {elapsed:.1f}s")
    log("INFO", f"Manifest: {manifest_path}")
    log("INFO", f"Failures: {failures_path}")

    if aborted_due_to_403_storm:
        return 3
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
