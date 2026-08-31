"""Fetch the AMI Meeting Corpus manual annotations -- real meetings, real decisions.

`training/generate_transcripts.py` writes meetings we invented. Held-out templates
keep that evaluation honest about phrasing, but they cannot invent the thing that
makes real meetings hard: people decide things across several turns, in ordinary
conversational language, surrounded by chatter that looks identical.

AMI is 142 real recorded meetings whose abstractive summaries were written by
human annotators under explicit ABSTRACT / DECISIONS / ACTIONS / PROBLEMS
headings, with `extractive/*.summlink.xml` linking each summary sentence back to
the dialogue acts that support it. That link is what turns AMI into a labelled
decision- and action-item-detection set, and it is how the published work on this
task builds it too.

    Carletta et al., "The AMI Meeting Corpus: A Pre-Announcement", MLMI 2005.
    https://groups.inf.ed.ac.uk/ami/corpus/
    Licensed CC BY 4.0.

Only the manual annotations are fetched -- 23MB. The full corpus with audio and
video is roughly 100GB and is not needed: this task is text.

Cached under datasets/real/ and NOT committed.

    python training/fetch_real_data.py           # download and cache
    python training/fetch_real_data.py --check   # verify the cache, no network
"""
import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "datasets" / "real"
ARCHIVE = CACHE_DIR / "ami_public_manual_1.6.2.zip"

URL = ("https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/"
       "ami_public_manual_1.6.2.zip")
EXPECTED_SHA256 = "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d"
EXPECTED_BYTES = 22887865
# Meetings that ship an abstractive summary; the rest cannot be labelled.
EXPECTED_MEETINGS = 142


def summarise(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    meetings = {n.split("/")[1].split(".")[0]
                for n in names if n.startswith("abstractive/") and n.endswith(".xml")}
    return len(names), len(meetings)


def download():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    request = urllib.request.Request(URL, headers={"User-Agent": "portfolio-fetch"})
    with urllib.request.urlopen(request, timeout=300) as response:
        data = response.read()
    ARCHIVE.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    print(f"cached  {ARCHIVE} ({len(data) / 1e6:.1f} MB)")
    print(f"sha256  {digest}")
    if digest != EXPECTED_SHA256:
        print(f"FAIL: checksum mismatch, expected {EXPECTED_SHA256}")
        return 1
    entries, meetings = summarise(ARCHIVE)
    print(f"entries {entries}, meetings with abstractive summaries {meetings}")
    if meetings != EXPECTED_MEETINGS:
        print(f"FAIL: expected {EXPECTED_MEETINGS} summarised meetings, got {meetings}")
        return 1
    return 0


def check():
    if not ARCHIVE.exists():
        print(f"MISSING: {ARCHIVE}\nrun: python training/fetch_real_data.py")
        return 1
    data = ARCHIVE.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    print(f"cached  {ARCHIVE} ({len(data) / 1e6:.1f} MB)")
    print(f"sha256  {digest}")
    if digest != EXPECTED_SHA256:
        print(f"FAIL: checksum mismatch, expected {EXPECTED_SHA256}")
        return 1
    if len(data) != EXPECTED_BYTES:
        print(f"FAIL: expected {EXPECTED_BYTES} bytes, got {len(data)}")
        return 1
    entries, meetings = summarise(ARCHIVE)
    print(f"entries {entries}, meetings with abstractive summaries {meetings}")
    if meetings != EXPECTED_MEETINGS:
        print(f"FAIL: expected {EXPECTED_MEETINGS} summarised meetings, got {meetings}")
        return 1
    print("OK: AMI annotations present, checksum and contents match")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify the cached copy without downloading")
    args = parser.parse_args()
    return check() if args.check else download()


if __name__ == "__main__":
    sys.exit(main())
