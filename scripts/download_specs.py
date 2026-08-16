"""
Download 3GPP specifications from the official 3GPP FTP archive.

Each spec is downloaded as the latest ZIP from:
  https://www.3gpp.org/ftp/Specs/archive/<series>/<spec_number>/

The ZIP is extracted and the .docx / .doc file saved under:
  data/raw/<generation>/<domain>/<spec_number>.<ext>

Usage:
    # Download everything
    python scripts/download_specs.py

    # Only 5G RAN specs
    python scripts/download_specs.py --generation 5G --domain RAN

    # Preview what would be downloaded (no files written)
    python scripts/download_specs.py --dry-run

    # Re-download even if file already exists
    python scripts/download_specs.py --force

    # Limit to specific specs
    python scripts/download_specs.py --specs 38.300 38.401 36.300
"""
import argparse
import io
import logging
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

# Add repo root to path so we can import src.*
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.spec_catalog import CATALOG, SpecEntry, get_specs_by_filter

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
FTP_BASE = "https://www.3gpp.org/ftp/Specs/archive/"
REQUEST_TIMEOUT = 30          # seconds per HTTP call
RETRY_DELAY = 5               # seconds between retries
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "3gpp-rag-assistant/2.0 (github.com/ericnc09/3gpp-rag-assistant)"
    return s


def _get_with_retry(session: requests.Session, url: str) -> Optional[requests.Response]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r
            logger.warning(f"HTTP {r.status_code} for {url} (attempt {attempt}/{MAX_RETRIES})")
        except requests.RequestException as exc:
            logger.warning(f"Request error for {url}: {exc} (attempt {attempt}/{MAX_RETRIES})")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
    return None


# ---------------------------------------------------------------------------
# ZIP / file helpers
# ---------------------------------------------------------------------------

def _parse_zip_links(html: str, base_url: str) -> list[str]:
    """Extract .zip file URLs from a 3GPP FTP directory listing page."""
    pattern = re.compile(r'href="([^"]+\.zip)"', re.IGNORECASE)
    links = pattern.findall(html)
    # Resolve relative links and return absolute URLs
    return [urljoin(base_url, link) for link in links]


def _latest_zip_url(zip_urls: list[str]) -> Optional[str]:
    """Pick the most recent ZIP by sorting filenames descending."""
    if not zip_urls:
        return None
    # Sort by basename — 3GPP names use version strings like 38300-h80.zip
    return sorted(zip_urls, key=lambda u: Path(u).name.lower())[-1]


def _extract_doc_from_zip(zip_bytes: bytes) -> Optional[tuple[str, bytes]]:
    """Return (filename, content) of the first .docx or .doc in a ZIP."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                lower = name.lower()
                if lower.endswith(".docx") or lower.endswith(".doc"):
                    return Path(name).name, zf.read(name)
    except zipfile.BadZipFile as exc:
        logger.error(f"Bad ZIP: {exc}")
    return None


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------

def download_spec(
    entry: SpecEntry,
    session: requests.Session,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """Download a single spec and save it to data/raw/.

    Returns True on success (or skip), False on failure.
    """
    spec_num = entry["spec_number"]
    generation = entry["generation"]
    domain = entry["domain"]

    # Destination directory: data/raw/<generation>/<domain>/
    dest_dir = RAW_DATA_DIR / generation / domain
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (any file starting with spec_nodot)
    spec_nodot = spec_num.replace(".", "")
    existing = list(dest_dir.glob(f"{spec_nodot}*"))
    if existing and not force:
        logger.info(f"  ✓ {spec_num} already downloaded ({existing[0].name}) — skipping")
        return True

    ftp_url = f"{FTP_BASE}{entry['series']}/{spec_num}/"
    logger.info(f"  → {spec_num}: {ftp_url}")

    if dry_run:
        print(f"    [dry-run] would download from {ftp_url}")
        return True

    # 1. Fetch directory listing
    dir_resp = _get_with_retry(session, ftp_url)
    if not dir_resp:
        logger.error(f"  ✗ {spec_num}: could not fetch directory listing")
        return False

    zip_urls = _parse_zip_links(dir_resp.text, ftp_url)
    if not zip_urls:
        logger.error(f"  ✗ {spec_num}: no ZIP files found in directory listing")
        return False

    latest_url = _latest_zip_url(zip_urls)
    logger.info(f"    Latest ZIP: {Path(latest_url).name}")

    # 2. Download ZIP
    zip_resp = _get_with_retry(session, latest_url)
    if not zip_resp:
        logger.error(f"  ✗ {spec_num}: could not download ZIP")
        return False

    # 3. Extract .doc / .docx
    result = _extract_doc_from_zip(zip_resp.content)
    if not result:
        logger.error(f"  ✗ {spec_num}: no .doc/.docx found in ZIP")
        return False

    orig_name, content = result
    ext = Path(orig_name).suffix.lower()
    dest_file = dest_dir / f"{spec_nodot}{ext}"

    dest_file.write_bytes(content)
    size_kb = len(content) // 1024
    logger.info(f"    ✓ Saved {dest_file} ({size_kb} KB)")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download 3GPP spec documents from the 3GPP FTP archive"
    )
    p.add_argument(
        "--generation",
        choices=["5G", "LTE"],
        help="Only download specs for this generation",
    )
    p.add_argument(
        "--domain",
        choices=["RAN", "CORE"],
        help="Only download specs for this domain",
    )
    p.add_argument(
        "--specs",
        nargs="+",
        metavar="SPEC",
        help="Only download these specific spec numbers (e.g. 38.300 38.401)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without writing any files",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Build the list of specs to download
    if args.specs:
        specs = [s for s in CATALOG if s["spec_number"] in args.specs]
        missing = set(args.specs) - {s["spec_number"] for s in specs}
        if missing:
            logger.warning(f"Unknown spec numbers (not in catalog): {', '.join(sorted(missing))}")
    else:
        specs = get_specs_by_filter(
            domain=args.domain,
            generation=args.generation,
        )

    if not specs:
        print("No specs matched the given filters.")
        sys.exit(0)

    print(f"\n{'='*60}")
    print(f"3GPP Spec Downloader")
    print(f"{'='*60}")
    if args.dry_run:
        print("DRY RUN — no files will be written\n")
    print(f"Specs to download: {len(specs)}")
    for s in specs:
        print(f"  {s['spec_number']:10s}  {s['generation']} {s['domain']:4s}  {s['title']}")
    print()

    session = _session()
    success = 0
    failed: list[str] = []

    for i, entry in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] TS {entry['spec_number']} — {entry['title']}")
        ok = download_spec(entry, session, dry_run=args.dry_run, force=args.force)
        if ok:
            success += 1
        else:
            failed.append(entry["spec_number"])
        # Polite delay between requests
        if i < len(specs) and not args.dry_run:
            time.sleep(1)

    print(f"\n{'='*60}")
    print(f"Download complete: {success}/{len(specs)} succeeded")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        print("Re-run with --force to retry failed specs.")
    print(f"Files saved to: {RAW_DATA_DIR.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
