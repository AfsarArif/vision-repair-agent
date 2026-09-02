#!/usr/bin/env python3
"""Download public DeepPCB images, workmanship PDFs, and Wikipedia extracts.

Binaries land in gitignored paths. See docs/DATASETS.md for licenses.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
PDF_DIR = ROOT / "docs" / "corpus" / "pdfs"
WIKI_DIR = ROOT / "docs" / "corpus" / "wikipedia"

USER_AGENT = "vision-repair-agent/0.1 (research demo; local use)"

NASA_PDFS = {
    "nasa-std-8739.6b.pdf": "https://standards.nasa.gov/sites/default/files/standards/NASA/B/0/nasa-std-87396b.pdf",
    "nasa-std-8739.1b.pdf": "https://standards.nasa.gov/sites/default/files/standards/NASA/B/2/nasa-std-87391B-Change-2.pdf",
}

ECSS_PDFS = {
    "ecss-q-st-70-61c.pdf": "https://ecss.nl/wp-content/uploads/2022/04/ECSS-Q-ST-70-61C(8April2022).pdf",
}

ARXIV_PDFS = {
    "arxiv-1902.06197-deeppcb.pdf": "https://arxiv.org/pdf/1902.06197",
    "arxiv-1901.08204-pku-pcb.pdf": "https://arxiv.org/pdf/1901.08204",
}

WIKI_TITLES = [
    "Printed_circuit_board",
    "Printed_circuit_board_manufacturing",
    "Automated_optical_inspection",
    "Soldering",
    "Rework_(electronics)",
    "Conformal_coating",
    "Solder_mask",
]


def _urlopen(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=60)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip (exists): {dest}")
        return
    print(f"download {url} -> {dest}")
    with _urlopen(url) as response:
        dest.write_bytes(response.read())


def clone_deeppcb() -> None:
    dest = DATA_RAW / "deeppcb"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / ".git").exists() or (dest / "PCBData").exists():
        print(f"skip clone (exists): {dest}")
        return
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/tangsanli5201/DeepPCB.git",
            str(dest),
        ],
        check=True,
    )


def download_wikipedia() -> None:
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    for title in WIKI_TITLES:
        dest = WIKI_DIR / f"{title.lower()}.md"
        if dest.exists():
            print(f"skip (exists): {dest}")
            continue
        api = (
            "https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
            f"&explaintext=1&redirects=1&format=json&titles={title}"
        )
        print(f"wikipedia {title}")
        with _urlopen(api) as response:
            payload = json.loads(response.read().decode("utf-8"))
        pages = payload.get("query", {}).get("pages", {})
        extract = ""
        canonical = title
        for page in pages.values():
            extract = page.get("extract") or ""
            canonical = page.get("title") or title
        url = f"https://en.wikipedia.org/wiki/{title}"
        dest.write_text(
            f"# {canonical}\n\n"
            f"source_id: wikipedia-{title.lower()}\n"
            f"license: CC-BY-SA-4.0\n"
            f"source_url: {url}\n\n"
            f"{extract}\n",
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deeppcb", action="store_true", help="git clone DeepPCB into data/raw/deeppcb")
    parser.add_argument("--corpus", action="store_true", help="NASA, ECSS, and arXiv PDFs")
    parser.add_argument("--wikipedia", action="store_true", help="Wikipedia plaintext extracts (CC BY-SA)")
    parser.add_argument("--all", action="store_true", help="all of the above")
    args = parser.parse_args()

    if not (args.deeppcb or args.corpus or args.wikipedia or args.all):
        parser.print_help()
        return 1

    try:
        if args.all or args.deeppcb:
            clone_deeppcb()
        if args.all or args.corpus:
            for name, url in {**NASA_PDFS, **ECSS_PDFS, **ARXIV_PDFS}.items():
                download_file(url, PDF_DIR / name)
        if args.all or args.wikipedia:
            download_wikipedia()
    except subprocess.CalledProcessError as exc:
        print(f"command failed: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
