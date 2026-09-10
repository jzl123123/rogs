#!/usr/bin/env python3
"""Fetch pinned, unmodified public source archives; never execute downloaded code."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "thirdparty"
LOCK = VENDOR / "sources.lock.json"


def download(url, output):
    for attempt in range(3):
        result = subprocess.run(["curl", "--fail", "--location", "--http1.1", "--retry", "2",
                                 "--connect-timeout", "15", "--max-time", "900",
                                 "--silent", "--show-error", url, "--output", str(output)])
        if result.returncode == 0:
            return
    result.check_returncode()


def extract(archive, destination, sparse_directories=None):
    with zipfile.ZipFile(archive) as zf:
        for entry in zf.infolist():
            rel = PurePosixPath(entry.filename)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe archive path: " + entry.filename)
            if len(rel.parts) < 2:
                continue
            # Sparse vendor layouts retain all root files plus explicitly selected directories.
            if sparse_directories is not None and len(rel.parts) > 2 and rel.parts[1] not in sparse_directories:
                continue
            if sparse_directories is not None and entry.is_dir() and rel.parts[1] not in sparse_directories:
                continue
            target = destination.joinpath(*rel.parts[1:])
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(entry))
                mode = (entry.external_attr >> 16) & 0o777
                if mode:
                    target.chmod(mode)


def tree_hash(directory):
    result = hashlib.sha256()
    for file in sorted(Path(directory).rglob("*")):
        if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
            result.update(file.relative_to(directory).as_posix().encode())
            result.update(b"\0")
            result.update(file.read_bytes())
            result.update(b"\0")
    return result.hexdigest()


def fetch_one(name, repository, ref, sparse_directories=None):
    destination = VENDOR / name
    if destination.exists():
        raise FileExistsError("Will not overwrite existing vendor source: " + str(destination))
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "source.zip"
        download("https://codeload.github.com/{}/zip/{}".format(repository, ref), archive)
        with zipfile.ZipFile(archive) as zf:
            commit = zf.comment.decode().strip()
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError("Archive does not identify an upstream commit")
        staged = Path(temporary) / "source"
        extract(archive, staged, sparse_directories)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), str(destination))
        return {"repository": repository, "commit": commit,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "tree_sha256": tree_hash(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    entries = json.loads(LOCK.read_text())["sources"]
    if args.only and set(args.only) - set(entries):
        parser.error("Unknown dependency: " + ", ".join(sorted(set(args.only) - set(entries))))
    failures = []
    for name, entry in entries.items():
        if args.only and name not in args.only:
            continue
        target = VENDOR / name
        if not target.exists() and not args.verify:
            fetched = fetch_one(name, entry["repository"], entry["commit"], entry.get("sparse_directories"))
            if fetched["commit"] != entry["commit"]:
                raise ValueError("Commit mismatch for " + name)
        if not target.exists() or tree_hash(target) != entry["tree_sha256"]:
            failures.append(name)
        else:
            print("OK", name, entry["commit"])
    if failures:
        raise SystemExit("Missing or modified source: " + ", ".join(failures))


if __name__ == "__main__":
    main()
