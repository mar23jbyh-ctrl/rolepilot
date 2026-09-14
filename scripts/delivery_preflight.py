"""Read-only installed-dependency and publishable-file checks; no cloud calls.

This is a high-confidence credential/path scan, not a semantic privacy audit.
Ignored local data and .env are intentionally never read. Git objects are read,
never rewritten. Only findings' paths/rules/line numbers are exported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import subprocess
import sys

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_MODELS = {
    "assets/ocr/tessdata/chi_sim.traineddata": "4fef2d1306c8e87616d4d3e4c6c67faf5d44be3342290cf8f2f0f6e3aa7e735b",
    "assets/ocr/tessdata/eng.traineddata": "8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba",
}
RULES = {
    "provider_key": re.compile(r"\b(?:sk-|tvly-)[A-Za-z0-9_-]{20,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "literal_bearer": re.compile(r"\bBearer\s+[A-Za-z0-9_.-]{32,}"),
}


def dependency_check(root=ROOT):
    queue = []
    for name in ("requirements.txt", "requirements-dev.txt"):
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                queue.append((Requirement(text), "direct:" + name))
    visited, checks = set(), []
    environment = default_environment()
    while queue:
        requirement, parent = queue.pop(0)
        key = (canonicalize_name(requirement.name), tuple(sorted(requirement.extras)))
        try:
            installed = metadata.version(requirement.name)
            passed = not requirement.specifier or requirement.specifier.contains(installed, prereleases=True)
        except metadata.PackageNotFoundError:
            installed, passed = None, False
        checks.append({"package": requirement.name, "required": str(requirement.specifier),
                       "installed": installed, "parent": parent, "passed": bool(passed)})
        if key in visited or installed is None:
            continue
        visited.add(key)
        for dependency in metadata.requires(requirement.name) or []:
            child = Requirement(dependency)
            extras = {"", *requirement.extras}
            if child.marker and not any(child.marker.evaluate({**environment, "extra": extra}) for extra in extras):
                continue
            queue.append((child, requirement.name))
    return {"method": "Installed metadata closure; active platform markers and requested extras. Does not import services or replace a fresh-machine installation test.",
            "checks": checks, "unique_package_count": len(visited),
            "issue_count": sum(not row["passed"] for row in checks)}


def scan_text(text, path):
    findings = []
    for rule, pattern in RULES.items():
        for match in pattern.finditer(text):
            value = match.group()
            # Explicit synthetic fixtures only, not arbitrary strings containing 'test'.
            if re.search(r"(?:sk-|tvly-)(?:synthetic|fixture|example|test)[_-]", value, re.I):
                continue
            findings.append({"path": path, "rule": rule, "line": text.count("\n", 0, match.start()) + 1})
    return findings


def private_path(path):
    name = Path(path).name.lower()
    return (name == ".env" or (name.startswith(".env.") and name not in {".env.example", ".env.sample"})
            or name.endswith((".db", ".db-wal", ".db-shm", ".sqlite", ".sqlite3", ".pem", ".p12")))


def git(*arguments, input=None):
    result = subprocess.run(["git", *arguments], cwd=ROOT, input=input, capture_output=True, check=True)
    return result.stdout


def publication_scan():
    candidates = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode("utf-8").split("\0")
    working, files, hashes = [], 0, {}
    for relative in sorted(set(filter(None, candidates))):
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            continue
        files += 1
        if private_path(relative):
            working.append({"path": relative, "rule": "private_artifact_path", "line": None})
            continue
        payload = path.read_bytes()
        if b"\0" not in payload:
            working.extend(scan_text(payload.decode("utf-8", errors="replace"), relative))
        hashes[relative] = hashlib.sha256(payload).hexdigest()
        if relative in OFFICIAL_MODELS and hashes[relative] != OFFICIAL_MODELS[relative]:
            working.append({"path": relative, "rule": "official_model_checksum_mismatch", "line": None})
    objects = git("rev-list", "--objects", "--all").decode("utf-8").splitlines()
    paths = dict(row.split(" ", 1) if " " in row else (row, "<unmapped>") for row in objects)
    descriptions = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                       input=("\n".join(paths) + "\n").encode()).decode().splitlines() if paths else []
    historical, scanned, skipped, verified_models = [], 0, [], []
    for description in descriptions:
        object_id, kind, size = description.split()
        if kind != "blob":
            continue
        relative = paths[object_id]
        if private_path(relative):
            historical.append({"path": relative, "rule": "private_artifact_path", "line": None, "blob": object_id})
        if int(size) > 5 * 1024 * 1024:
            # Only exact pinned official binaries receive a larger bounded read.
            # A model filename alone must never bypass publication review.
            if relative in OFFICIAL_MODELS and int(size) <= 32 * 1024 * 1024:
                payload = git("cat-file", "blob", object_id)
                scanned += 1
                if len(payload) == int(size) and hashlib.sha256(payload).hexdigest() == OFFICIAL_MODELS[relative]:
                    verified_models.append({"path": relative, "blob": object_id,
                                            "sha256": OFFICIAL_MODELS[relative]})
                    continue
                historical.append({"path": relative, "rule": "official_model_checksum_mismatch",
                                   "line": None, "blob": object_id})
            skipped.append({"blob": object_id, "reason": "over_5MiB"})
            continue
        payload = git("cat-file", "blob", object_id)
        scanned += 1
        if b"\0" not in payload:
            historical.extend({**finding, "blob": object_id} for finding in scan_text(payload.decode("utf-8", errors="replace"), relative))
    return {"scope": "Git-candidate current files and every unique reachable historical blob from all refs. Unreachable objects, ignored private runtime data and semantic personal information are not audited.",
            "current_files": files, "current_findings": working, "history_blobs_read": scanned,
            "history_findings": historical, "skipped_blobs": skipped,
            "verified_official_model_blobs": verified_models, "current_file_hashes": hashes,
            "commit_count": int(git("rev-list", "--all", "--count").strip() or b"0")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional NEW folder inside docs/release/evidence")
    args = parser.parse_args(argv)
    output = (ROOT / args.output).resolve() if args.output else None
    if output and (output.exists() or not output.is_relative_to((ROOT / "docs/release/evidence").resolve())):
        parser.error("Output must be a NEW folder inside docs/release/evidence")
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
              "dependencies": dependency_check(), "publication": publication_scan()}
    if output:
        output.mkdir(parents=True)
        (output / "preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"dependency_issues": result["dependencies"]["issue_count"],
               "packages": result["dependencies"]["unique_package_count"],
               "current_findings": result["publication"]["current_findings"],
               "history_findings": result["publication"]["history_findings"],
               "history_blobs_read": result["publication"]["history_blobs_read"],
               "commit_count": result["publication"]["commit_count"],
               "skipped_blobs": len(result["publication"]["skipped_blobs"]),
               "verified_official_models": len(result["publication"]["verified_official_model_blobs"])}
    print(json.dumps(summary, ensure_ascii=False))
    return int(bool(summary["dependency_issues"] or summary["current_findings"] or summary["history_findings"] or summary["skipped_blobs"]))


if __name__ == "__main__":
    raise SystemExit(main())
