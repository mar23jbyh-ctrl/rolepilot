"""Install pinned official chi_sim locally; preserve the existing English model.

No admin rights, pip packages, .env edits or replacement of system tessdata.
Run with the project venv after installing the Tesseract executable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COMMIT = "e12c65a915945e4c28e237a9b52bc4a8f39a0cec"
URL = f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/{COMMIT}/chi_sim.traineddata"
EXPECTED_SHA256 = "4fef2d1306c8e87616d4d3e4c6c67faf5d44be3342290cf8f2f0f6e3aa7e735b"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(destination: Path, command: str, source: Path | None = None) -> dict:
    executable = shutil.which(command)
    if not executable:
        raise RuntimeError("Tesseract executable missing; install the engine first")
    if source is None:
        result = subprocess.run([executable, "--list-langs"], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=5, check=True)
        match = re.search(r'"([^"\n]+)"', result.stdout)
        if not match:
            raise RuntimeError("Cannot locate system tessdata; specify --source-tessdata")
        source = Path(match.group(1))
    destination = destination.resolve()
    if destination == source.resolve():
        raise RuntimeError("Use a separate project language directory, not system tessdata")
    if not (source / "eng.traineddata").is_file() and not (destination / "eng.traineddata").is_file():
        raise RuntimeError("Existing eng.traineddata missing; install the English model first")
    destination.mkdir(parents=True, exist_ok=True)
    model = destination / "chi_sim.traineddata"
    if model.exists() and digest(model) != EXPECTED_SHA256:
        raise RuntimeError("Existing chi_sim differs; do not overwrite. Select a new directory")
    if not model.exists():
        with tempfile.TemporaryDirectory(prefix="interview-ocr-language-") as temp:
            downloaded = Path(temp) / "chi_sim.traineddata"
            with urllib.request.urlopen(URL, timeout=60) as response, downloaded.open("xb") as target:
                total = 0
                while chunk := response.read(65536):
                    total += len(chunk)
                    if total > 32 * 1024 * 1024:
                        raise RuntimeError("Official model download exceeds size limit")
                    target.write(chunk)
            if digest(downloaded) != EXPECTED_SHA256:
                raise RuntimeError("Official model SHA256 mismatch; nothing installed")
            # Exclusive creation also avoids replacing a concurrent install.
            with downloaded.open("rb") as src, model.open("xb") as target:
                shutil.copyfileobj(src, target)
    for lang in ("eng", "osd"):
        original, local = source / f"{lang}.traineddata", destination / f"{lang}.traineddata"
        if original.is_file() and not local.exists():
            with original.open("rb") as src, local.open("xb") as target:
                shutil.copyfileobj(src, target)
    check = subprocess.run([executable, "--tessdata-dir", str(destination), "--list-langs"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5, check=True)
    languages = sorted(line.strip() for line in check.stdout.splitlines()
                       if re.fullmatch(r"[A-Za-z0-9_]+", line.strip()))
    if not {"chi_sim", "eng"}.issubset(languages):
        raise RuntimeError("Installed directory does not expose chi_sim and eng")
    record = {"source_url": URL, "commit": COMMIT, "chi_sim_sha256": digest(model),
              "chi_sim_bytes": model.stat().st_size, "eng_sha256": digest(destination / "eng.traineddata"),
              "languages": languages, "system_tessdata_modified": False}
    (destination.parent / "installation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main():
    from app.config import settings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=ROOT / "data/ocr/tessdata")
    parser.add_argument("--source-tessdata", type=Path)
    args = parser.parse_args()
    print(json.dumps(install(args.dest, settings.tesseract_cmd, args.source_tessdata), indent=2))


if __name__ == "__main__":
    main()
