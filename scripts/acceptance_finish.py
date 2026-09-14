"""Hash public source and check documentation links without cloud calls.

An optional offline result is supplied explicitly; no historical logs or runtime
databases are required. Reports are created exclusively in the ignored evidence
directory. Input contents and environment secrets are never copied to a report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("app", "api", "frontend/src", "frontend/tests", "tests", ".github")
SOURCE_SUFFIXES = {".py", ".ts", ".tsx", ".cjs", ".json", ".css", ".yml", ".yaml", ".txt"}
ROOT_FILES = (
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", ".env.example", ".gitignore",
    "main.py", "requirements.txt", "requirements.lock.txt", "requirements-dev.txt",
    "frontend/index.html", "frontend/package.json", "frontend/package-lock.json",
    "frontend/tsconfig.json", "frontend/vite.config.ts",
    "scripts/README.md", "scripts/acceptance_cleanup.py", "scripts/acceptance_finish.py",
    "scripts/delivery_preflight.py", "scripts/delivery_smoke.py", "scripts/guard_benchmark.py",
    "scripts/install_ocr_languages.py", "scripts/ocr_benchmark.py", "scripts/release_run.py",
    "scripts/release_score_replay.py", "scripts/release_server.py",
)


def public_source_files():
    paths = {ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()}
    for folder in SOURCE_DIRS:
        for base, directories, names in os.walk(ROOT / folder):
            directories[:] = [name for name in directories if name not in {"__pycache__", ".pytest_cache"}]
            for name in names:
                path = Path(base) / name
                if path.suffix in SOURCE_SUFFIXES and not path.is_symlink():
                    paths.add(path)
    paths.update(path for path in (ROOT / "docs").glob("*.md") if path.is_file())
    return sorted(paths)


def build_report(offline=None):
    paths = public_source_files()
    hashes, broken = {}, []
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix != ".md":
            continue
        for match in re.finditer(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8-sig")):
            url = match.group(1).strip().strip("<>")
            if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:|^#", url):
                continue
            target = unquote(url.split("#", 1)[0])
            if target and not (path.parent / target).exists():
                broken.append({"document": relative, "target": url})
    result = {
        "schema_version": "source-validation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashes,
        "broken_local_links": broken,
        "scope": "Source hashes and local documentation links; no model quality or deployment certification",
    }
    if offline is not None:
        result["supplied_offline_result"] = {
            "all_passed": offline["all_passed"], "check_count": len(offline["checks"]),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/release/evidence/my-source-validation.json",
                        help="New JSON report inside docs/release/evidence; existing files are never replaced")
    parser.add_argument("--offline", help="Optional JSON result with boolean all_passed and a checks array")
    args = parser.parse_args(argv)
    output = (ROOT / args.output).resolve()
    evidence = (ROOT / "docs/release/evidence").resolve()
    if not output.is_relative_to(evidence) or output.suffix != ".json" or output.exists():
        parser.error("output must be a NEW JSON file inside docs/release/evidence")
    offline = None
    if args.offline:
        offline = json.loads((ROOT / args.offline).read_text(encoding="utf-8"))
        if (not isinstance(offline, dict) or type(offline.get("all_passed")) is not bool
                or not isinstance(offline.get("checks"), list)):
            parser.error("offline result requires boolean all_passed and a checks array")
    result = build_report(offline)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"source_files": len(result["source_sha256"]),
                      "broken_local_links": result["broken_local_links"]}, ensure_ascii=False))
    return int(bool(result["broken_local_links"]) or (offline is not None and not offline["all_passed"]))


if __name__ == "__main__":
    raise SystemExit(main())
