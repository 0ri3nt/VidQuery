"""Stream local-only VidQuery artifacts to a versioned GitHub Release.

The archive contains files outside Git, excluding credentials and portable
build caches. Use ``plan`` before ``upload`` and ``verify`` after publication.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import truststore
except ImportError:
    pass
else:
    truststore.inject_into_ssl()

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
STATE = ROOT / "tmp" / "archive-release-state.json"
FILE_MANIFEST = ROOT / "archive" / "backup-files.jsonl.gz"
SUMMARY = ROOT / "archive" / "RELEASE-2026-11-26.json"
OWNER_REPO = "0ri3nt/VidQuery"
TAG = "archive-dev-aryaman-2026-11-26"
MAX_TAR = 100_000_000
MAX_PART = 90_000_000
LOCAL_PATHS = [
    "data/app/models", "data/app/features", "data/app/generated",
    "data/app/backups", "data/app/demo_library", "data/app/cache",
    "data/app/uploads", "data/app/vidquery.sqlite3",
    "data/app/benchmark/yolo_audit_2026-10-26/weights",
    "data/ava/gnn_outputs", "data/ava/audio_outputs",
    "evaluation", "archive/superseded_results", "docs", "data/demo",
    "yolov8m.pt", "yolov8n.pt", "yolov8s.pt", "AGENTS.md",
]
SKIP_DIRS = {
    ".git", ".venv", ".venv-local", ".mypy_cache", ".ruff_cache",
    ".pytest_cache", "__pycache__", "node_modules", "vidquery.egg-info",
}
SKIP_FILES = {".env", "archive-release-state.json", "backup-files.jsonl.gz", SUMMARY.name}


def credential() -> str:
    result = subprocess.run(
        ["git", "credential", "fill"],
        input=b"protocol=https\nhost=github.com\n\n",
        capture_output=True,
        check=True,
    )
    for line in result.stdout.decode().splitlines():
        if line.startswith("password="):
            return line.partition("=")[2]
    raise RuntimeError("GitHub credential unavailable")


def api(method: str, path: str, token: str | None, data: dict | None = None) -> dict:
    payload = None if data is None else json.dumps(data).encode()
    request = urllib.request.Request(
        "https://api.github.com" + path,
        data=payload,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "VidQuery-archive",
            **({"Authorization": "Bearer " + token} if token else {}),
            **({"Content-Type": "application/json"} if payload else {}),
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
        return json.loads(body) if body else {}


def release_assets(release_id: int, token: str | None) -> dict[str, dict]:
    assets = {}
    page = 1
    while True:
        batch = api("GET", f"/repos/{OWNER_REPO}/releases/{release_id}/assets?per_page=100&page={page}", token)
        for asset in batch:
            assets[asset["name"]] = asset
        if len(batch) < 100:
            return assets
        page += 1


def tracked() -> set[str]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return {os.fsdecode(item).replace("\\", "/") for item in raw.split(b"\0") if item}


def inventory() -> list[dict]:
    indexed = tracked()
    found = []
    for source in LOCAL_PATHS:
        initial = ROOT / source
        if not initial.exists():
            continue
        candidates = [initial] if initial.is_file() else (
            path for path in initial.rglob("*") if path.is_file()
            and not any(part in SKIP_DIRS for part in path.relative_to(initial).parts)
        )
        for path in candidates:
            name = path.name
            rel = path.relative_to(ROOT).as_posix()
            if name in SKIP_FILES or rel in indexed or path.is_symlink():
                continue
            stat = path.stat()
            found.append({"path": rel, "source": str(path), "size": stat.st_size,
                          "mtime_ns": stat.st_mtime_ns})
    for path in sorted(PARENT.iterdir()):
        if path == ROOT or path.name in SKIP_DIRS:
            continue
        candidates = [path] if path.is_file() else (
            child for child in path.rglob("*") if child.is_file()
            and not any(part in SKIP_DIRS for part in child.relative_to(PARENT).parts)
        )
        for child in candidates:
            if child.is_symlink():
                continue
            rel = "__capstone_extras__/" + child.relative_to(PARENT).as_posix()
            stat = child.stat()
            found.append({"path": rel, "source": str(child), "size": stat.st_size,
                          "mtime_ns": stat.st_mtime_ns})
    expanded = []
    for item in sorted(found, key=lambda x: x["path"]):
        parts = max(1, (item["size"] + MAX_PART - 1) // MAX_PART)
        for part in range(parts):
            offset = part * MAX_PART
            expanded.append({**item, "original_path": item["path"],
                             "path": item["path"] if parts == 1 else
                             f"__segments__/{item['path']}.part{part:04d}",
                             "original_size": item["size"],
                             "size": min(MAX_PART, item["size"] - offset),
                             "offset": offset, "part": part, "parts": parts})
    return expanded


def tarinfo(item: dict) -> tarfile.TarInfo:
    info = tarfile.TarInfo(item["path"])
    info.size = item["size"]
    info.mode = 0o644
    info.mtime = 0
    return info


def entry_size(item: dict) -> int:
    header = tarinfo(item).tobuf(format=tarfile.PAX_FORMAT, encoding="utf-8")
    return len(header) + ((item["size"] + 511) // 512) * 512


def tar_size(items: list[dict]) -> int:
    used = sum(entry_size(item) for item in items) + 1024
    return ((used + 10239) // 10240) * 10240


def volumes(items: list[dict]) -> list[dict]:
    result = []
    group = []
    used = 0
    for item in items:
        amount = entry_size(item)
        if group and used + amount + 11264 > MAX_TAR:
            result.append(group)
            group, used = [], 0
        if amount + 11264 > MAX_TAR:
            raise RuntimeError("Single file too large for Release: " + item["path"])
        group.append(item)
        used += amount
    if group:
        result.append(group)
    return [{"name": f"vidquery-local-{n:03d}.tar", "items": group,
             "size": tar_size(group)} for n, group in enumerate(result, 1)]


class HashReader:
    def __init__(self, path: str, offset: int, length: int):
        self.file = open(path, "rb")
        self.file.seek(offset)
        self.remaining = length
        self.hash = hashlib.sha256()

    def read(self, amount: int) -> bytes:
        data = self.file.read(min(amount, self.remaining))
        self.hash.update(data)
        self.remaining -= len(data)
        return data

    def close(self) -> None:
        self.file.close()


class SocketWriter:
    def __init__(self, output, name: str):
        self.output = output
        self.name = name
        self.hash = hashlib.sha256()
        self.count = 0
        self.next_report = 256 * 1024 * 1024

    def write(self, data: bytes) -> int:
        self.output.write(data)
        self.hash.update(data)
        self.count += len(data)
        if self.count >= self.next_report:
            print(f"{self.name}: {self.count / 1e9:.2f} GB sent", flush=True)
            self.next_report += 256 * 1024 * 1024
        return len(data)

    def flush(self) -> None:
        pass


def upload_tar(volume: dict, release_id: int, token: str) -> dict:
    query = urllib.parse.urlencode({"name": volume["name"]})
    url = f"https://uploads.github.com/repos/{OWNER_REPO}/releases/{release_id}/assets?{query}"
    process = subprocess.Popen(
        ["curl.exe", "--http1.1", "--silent", "--show-error", "--request", "POST",
         "--header", "Authorization: Bearer " + token,
         "--header", "Accept: application/vnd.github+json",
         "--header", "Content-Type: application/x-tar",
         "--header", "Content-Length: " + str(volume["size"]),
         "--data-binary", "@-", url],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    writer = SocketWriter(process.stdin, volume["name"])
    records = []
    with tarfile.open(fileobj=writer, mode="w|", format=tarfile.PAX_FORMAT,
                      bufsize=1024 * 1024) as archive:
        for item in volume["items"]:
            before = os.stat(item["source"])
            if (before.st_size, before.st_mtime_ns) != (item["original_size"], item["mtime_ns"]):
                raise RuntimeError("Source changed before upload: " + item["path"])
            reader = HashReader(item["source"], item["offset"], item["size"])
            try:
                archive.addfile(tarinfo(item), reader)
            finally:
                reader.close()
            after = os.stat(item["source"])
            if (after.st_size, after.st_mtime_ns) != (item["original_size"], item["mtime_ns"]):
                raise RuntimeError("Source changed during upload: " + item["path"])
            records.append({"path": item["path"], "size": item["size"],
                            "sha256": reader.hash.hexdigest(), "asset": volume["name"],
                            "original_path": item["original_path"], "part": item["part"],
                            "parts": item["parts"], "offset": item["offset"]})
    if writer.count != volume["size"]:
        raise RuntimeError(f"Tar size mismatch: {writer.count} != {volume['size']}")
    process.stdin.close()
    assert process.stdout is not None and process.stderr is not None
    body = process.stdout.read()
    error = process.stderr.read()
    status = process.wait(timeout=3600)
    if status or not body:
        raise RuntimeError(f"GitHub upload failed ({status}): {error[:500]!r} {body[:500]!r}")
    asset = json.loads(body)
    if asset.get("id") is None:
        raise RuntimeError(f"GitHub rejected asset: {body[:500]!r}")
    if asset["size"] != writer.count or asset["state"] != "uploaded":
        raise RuntimeError("GitHub asset size/state mismatch")
    return {"name": volume["name"], "size": writer.count,
            "sha256": writer.hash.hexdigest(), "asset_id": asset["id"],
            "file_count": len(records), "records": records}


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".new")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["plan", "upload", "check", "verify", "restore"])
    parser.add_argument("--destination", type=Path,
                        help="Existing clone directory for the restore command")
    parser.add_argument("--sample-assets", type=int, default=6,
                        help="Number of volumes to download and parse during the check command")
    args = parser.parse_args()
    if args.mode in {"check", "verify", "restore"}:
        if args.mode == "restore" and not args.destination:
            parser.error("restore requires --destination")
        verify(args.destination if args.mode == "restore" else None,
               args.sample_assets if args.mode == "check" else None)
        return
    items = inventory()
    groups = volumes(items)
    print(f"{len(items)} files, {sum(i['size'] for i in items) / 2**30:.2f} GiB, "
          f"{len(groups)} Release assets", flush=True)
    if args.mode == "plan":
        for group in groups:
            print(group["name"], group["size"], len(group["items"]))
        return
    token = credential()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if not state:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        release = api("POST", f"/repos/{OWNER_REPO}/releases", token, {
            "tag_name": TAG, "target_commitish": head, "name": "VidQuery full local archive 2026-11-26",
            "body": "Functioning system and trained model archive for dev_aryaman. Raw datasets are excluded. See archive/RESTORE.md on that branch.",
            "draft": True, "prerelease": True,
        })
        state = {"release_id": release["id"], "source_commit": head, "volumes": []}
        save_json(STATE, state)
    completed = {entry["name"] for entry in state["volumes"]}
    for group in groups:
        if group["name"] in completed:
            continue
        print("Uploading", group["name"], flush=True)
        for attempt in range(1, 6):
            try:
                result = upload_tar(group, state["release_id"], token)
                break
            except (OSError, RuntimeError) as error:
                print(f"Attempt {attempt} failed for {group['name']}: {error}", flush=True)
                asset = release_assets(state["release_id"], token).get(group["name"])
                if asset is not None:
                    api("DELETE", f"/repos/{OWNER_REPO}/releases/assets/{asset['id']}", token)
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 15))
        state["volumes"].append(result)
        save_json(STATE, state)
    records = [record for volume in state["volumes"] for record in volume["records"]]
    with gzip.open(FILE_MANIFEST, "wt", encoding="utf-8", compresslevel=9) as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
    summary = {"repository": OWNER_REPO, "branch": "dev_aryaman", "tag": TAG,
               "source_commit": state["source_commit"], "release_id": state["release_id"],
               "files": len(records), "bytes": sum(r["size"] for r in records),
               "excluded": [".env (credentials)", "Git metadata and tracked files (on branch)",
                            "raw AVA/VidOR datasets and benchmark media/databases",
                            "virtual environments, Python build and lint caches"],
               "manifest_sha256": hashlib.sha256(FILE_MANIFEST.read_bytes()).hexdigest(),
               "volumes": [{k: v[k] for k in ("name", "size", "sha256", "asset_id", "file_count")}
                           for v in state["volumes"]]}
    save_json(SUMMARY, summary)
    print("Wrote", SUMMARY, flush=True)


class HashStream:
    def __init__(self, source):
        self.source = source
        self.hash = hashlib.sha256()
        self.count = 0

    def read(self, amount: int) -> bytes:
        data = self.source.read(amount)
        self.hash.update(data)
        self.count += len(data)
        return data


def verify(destination: Path | None = None, sample_assets: int | None = None) -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    token = None  # Public Release downloads require no GitHub account.
    assets = release_assets(summary["release_id"], token)
    for volume in summary["volumes"]:
        asset = assets[volume["name"]]
        if (asset["size"] != volume["size"] or asset["state"] != "uploaded" or
                asset.get("digest") != "sha256:" + volume["sha256"]):
            raise RuntimeError("GitHub asset metadata/hash mismatch: " + volume["name"])
    print(f"GitHub SHA-256 matches all {len(summary['volumes'])} volumes", flush=True)
    expected = {}
    with gzip.open(FILE_MANIFEST, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            expected[(record["asset"], record["path"])] = record
    seen = set()
    if destination is not None:
        destination = destination.resolve()
        destination.mkdir(parents=True, exist_ok=True)
    selected = summary["volumes"]
    if sample_assets is not None:
        if sample_assets < 1:
            raise ValueError("sample_assets must be positive")
        count = min(sample_assets, len(selected))
        indices = {round(i * (len(selected) - 1) / max(count - 1, 1))
                   for i in range(count)}
        selected = [selected[i] for i in sorted(indices)]
    for volume in selected:
        asset = assets[volume["name"]]
        if asset["size"] != volume["size"] or asset["state"] != "uploaded":
            raise RuntimeError("Remote asset metadata mismatch: " + volume["name"])
        request = urllib.request.Request(asset["browser_download_url"],
                                         headers={"User-Agent": "VidQuery-archive"})
        with urllib.request.urlopen(request, timeout=3600) as response:
            stream = HashStream(response)
            with tarfile.open(fileobj=stream, mode="r|", format=tarfile.PAX_FORMAT) as archive:
                for member in archive:
                    if not member.isfile():
                        raise RuntimeError("Unexpected tar entry: " + member.name)
                    parts = Path(member.name).parts
                    if not parts or any(part in {"..", "."} for part in parts):
                        raise RuntimeError("Unsafe tar path: " + member.name)
                    key = (volume["name"], member.name)
                    if key not in expected or key in seen:
                        raise RuntimeError("Unexpected/duplicate archived file: " + member.name)
                    record = expected[key]
                    if member.size != record["size"]:
                        raise RuntimeError("Restored file size mismatch: " + member.name)
                    original_parts = Path(record["original_path"]).parts
                    if any(part in {"..", "."} for part in original_parts):
                        raise RuntimeError("Unsafe restored path: " + record["original_path"])
                    target = None
                    if destination is not None:
                        if original_parts[0] == "__capstone_extras__":
                            target = destination.parent.joinpath(*original_parts[1:])
                        else:
                            target = destination.joinpath(*original_parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if record["part"] and (not target.exists() or
                                               target.stat().st_size != record["offset"]):
                            raise RuntimeError("Missing prior file part: " + str(target))
                    digest = hashlib.sha256()
                    with archive.extractfile(member) as source:
                        if target is None:
                            while chunk := source.read(1024 * 1024):
                                digest.update(chunk)
                        else:
                            with target.open("wb" if record["part"] == 0 else "ab") as output:
                                while chunk := source.read(1024 * 1024):
                                    output.write(chunk)
                                    digest.update(chunk)
                    if digest.hexdigest() != record["sha256"]:
                        raise RuntimeError("Restored file hash mismatch: " + member.name)
                    seen.add(key)
            while stream.read(1024 * 1024):
                pass
        if stream.count != volume["size"] or stream.hash.hexdigest() != volume["sha256"]:
            raise RuntimeError("Remote bytes mismatch: " + volume["name"])
        print("Verified", volume["name"], volume["file_count"], "files", flush=True)
    if sample_assets is None and len(seen) != len(expected):
        raise RuntimeError(f"Missing files: {len(expected) - len(seen)}")
    print(f"Parsed and verified {len(seen)} archived file parts in {len(selected)} volumes", flush=True)


if __name__ == "__main__":
    main()
