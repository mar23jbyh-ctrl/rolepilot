"""Deterministic OCR-only benchmark. No LLM, search API or private resume.

Prepare once, then compare runs on identical input hashes.
Ground truth is used only by metrics, never passed to the parser/upload endpoint.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tempfile
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRUTH = ROOT / "tests/fixtures/ocr/cases.json"
DEFAULT_OUT = ROOT / "docs/release/evidence/my-ocr-benchmark"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text))


def edit_distance(reference: str, prediction: str) -> int:
    row = list(range(len(prediction) + 1))
    for i, expected in enumerate(reference, 1):
        next_row = [i]
        for j, actual in enumerate(prediction, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (expected != actual)))
        row = next_row
    return row[-1]


def score(reference: str, prediction: str, skills: list[str]) -> dict:
    raw_ref = unicodedata.normalize("NFC", reference)
    raw_pred = unicodedata.normalize("NFC", prediction)
    ref, pred = normalize(reference), normalize(prediction)
    han_ref = "".join(re.findall(r"[\u3400-\u9fff]", raw_ref))
    han_pred = "".join(re.findall(r"[\u3400-\u9fff]", raw_pred))
    recognized = []
    for term in skills:
        if re.search(r"[\u3400-\u9fff]", term):
            found = normalize(term) in pred
        else:
            pattern = r"(?<![A-Za-z0-9_])" + r"\s*".join(re.escape(p) for p in term.split()) + r"(?![A-Za-z0-9_])"
            found = re.search(pattern, prediction, re.IGNORECASE) is not None
        if found:
            recognized.append(term)
    errors = edit_distance(ref, pred)
    raw_errors = edit_distance(raw_ref, raw_pred)
    return {"reference_chars": len(ref), "errors": errors, "cer": errors / max(1, len(ref)),
            "raw_reference_chars": len(raw_ref), "raw_errors": raw_errors,
            "raw_cer": raw_errors / max(1, len(raw_ref)),
            "han_reference_chars": len(han_ref), "han_errors": edit_distance(han_ref, han_pred),
            "skills_total": len(skills), "skills_recognized": len(recognized),
            "recognized": recognized, "missing": [s for s in skills if s not in recognized]}


def aggregate(rows: list[dict]) -> dict:
    chars = sum(r["reference_chars"] for r in rows)
    errors = sum(r["errors"] for r in rows)
    skills = sum(r["skills_total"] for r in rows)
    found = sum(r["skills_recognized"] for r in rows)
    raw_chars = sum(r["raw_reference_chars"] for r in rows)
    han_chars = sum(r["han_reference_chars"] for r in rows)
    times = sorted(r["elapsed_seconds"] for r in rows)
    return {"runs": len(rows), "reference_chars": chars, "errors": errors,
            "cer": errors / chars if chars else None, "character_accuracy": 1 - errors / chars if chars else None,
            "raw_cer": sum(r["raw_errors"] for r in rows) / raw_chars if raw_chars else None,
            "han_reference_chars": han_chars, "han_errors": sum(r["han_errors"] for r in rows),
            "han_cer": sum(r["han_errors"] for r in rows) / han_chars if han_chars else None,
            "skills_total": skills, "skills_recognized": found, "skill_recall": found / skills if skills else None,
            "elapsed_total_seconds": sum(times), "elapsed_median_seconds": statistics.median(times) if times else 0,
            "elapsed_p95_seconds": times[max(0, math.ceil(len(times) * .95) - 1)] if times else 0,
            "http_failures": sum(r.get("http_status", 200) != 200 for r in rows)}


def draw_text(text: str, font_name: str, size: int = 42):
    from PIL import Image, ImageDraw, ImageFont
    font_path = Path("C:/Windows/Fonts") / font_name
    font = ImageFont.truetype(str(font_path), size)
    lines = text.splitlines()
    width = int(max(font.getlength(line) for line in lines)) + 160
    height = 140 + len(lines) * (size + 26)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((80, 60 + i * (size + 26)), line, font=font, fill="black")
    return image, {"font": font_name, "font_sha256": sha256(font_path), "size": size,
                   "width": width, "height": height, "layout": "single-column, black-on-white"}


def prepare(out: Path) -> None:
    from PIL import Image
    import pdfplumber
    spec = json.loads(TRUTH.read_text(encoding="utf-8"))
    if (out / "manifest.json").exists():
        raise SystemExit("Manifest already exists: frozen inputs will not be overwritten")
    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    cases = []
    pdf_images = []
    for case in spec["cases"]:
        for variant, font in [("sans", "msyh.ttc"), ("serif", "simsun.ttc")]:
            image, render = draw_text(case["text"], font)
            name = case["id"] + "_" + variant
            path = inputs / (name + ".png")
            image.save(path, dpi=(300, 300))
            cases.append({**case, "id": name, "group": "core_image", "path": path.relative_to(out).as_posix(),
                          "input_sha256": sha256(path), "render": render})
            if variant == "sans" and case["source_id"] == "backend":
                pdf_path = inputs / (case["id"] + "_scan.pdf")
                image.save(pdf_path, "PDF", resolution=300.0)
                with pdfplumber.open(pdf_path) as doc:
                    native_chars = sum(len(p.extract_text() or "") for p in doc.pages)
                assert native_chars == 0, "Scanned PDF fixture must have no text layer"
                cases.append({**case, "id": case["id"] + "_scan", "group": "core_pdf",
                              "path": pdf_path.relative_to(out).as_posix(), "input_sha256": sha256(pdf_path),
                              "render": render, "native_text_chars": native_chars})
                pdf_images.append((case, image.copy()))
            if variant == "sans" and case["source_id"] == "legal":
                # Deliberately degraded holdout. Do not hide it in the clean-image aggregate.
                stress = image.resize((image.width // 3, image.height // 3), Image.Resampling.LANCZOS)
                stress = stress.rotate(3, expand=True, fillcolor="white")
                stress_path = inputs / (case["id"] + "_stress.jpg")
                stress.save(stress_path, quality=55, dpi=(100, 100))
                cases.append({**case, "id": case["id"] + "_stress", "group": "stress",
                              "path": stress_path.relative_to(out).as_posix(), "input_sha256": sha256(stress_path),
                              "render": {**render, "degradation": "100dpi-equivalent, 3-degree rotation, JPEG quality 55"}})
    multi = inputs / "multilingual_scan.pdf"
    pdf_images[0][1].save(multi, "PDF", resolution=300.0, save_all=True,
                           append_images=[item[1] for item in pdf_images[1:]])
    with pdfplumber.open(multi) as doc:
        assert len(doc.pages) == 3
        assert sum(len(p.extract_text() or "") for p in doc.pages) == 0
    cases.append({"id": "multilingual_scan", "language": "mixed", "source_id": "backend",
                  "text": "\n\n".join(c["text"] for c, _ in pdf_images),
                  "skills": sorted(set(s for c, _ in pdf_images for s in c["skills"])),
                  "group": "core_pdf", "path": multi.relative_to(out).as_posix(),
                  "input_sha256": sha256(multi), "native_text_chars": 0, "pages": 3})
    for case in cases:
        case["truth_sha256"] = hashlib.sha256(case["text"].encode()).hexdigest()
    manifest = {"schema_version": "ocr-inputs-v1", "ground_truth_sha256": sha256(TRUTH),
                "acceptance": spec["acceptance"], "sources": spec["sources"], "cases": cases}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"prepared": len(cases), "pdf_count": 4, "ground_truth_sha256": sha256(TRUTH)}), flush=True)


def run(out: Path, phase: str, repeats: int, selected: str, transport: str) -> None:
    import pytesseract
    from app.config import settings
    from app.parsers.resume_parser import extract_resume_text
    from app.parsers import ocr as local_ocr
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert sha256(TRUTH) == manifest["ground_truth_sha256"], "Ground truth changed"
    result_path = out / (phase + ".json")
    if result_path.exists():
        raise SystemExit("Choose a new phase name; never overwrite run evidence")
    rows = []
    engine_calls = []
    original_ocr = pytesseract.image_to_string
    def observed_ocr(*args, **kwargs):
        start = time.perf_counter()
        entry = {"lang": kwargs.get("lang"), "config": kwargs.get("config", ""),
                 "timeout": kwargs.get("timeout"), "case_id": current["id"]}
        try:
            text = original_ocr(*args, **kwargs)
            raw_path = phase_folder / f"engine-{len(engine_calls) + 1}.txt"
            raw_path.write_text(text, encoding="utf-8")
            entry["raw_output"] = raw_path.relative_to(out).as_posix()
            entry["raw_output_sha256"] = sha256(raw_path)
            return text
        finally:
            entry["elapsed_seconds"] = time.perf_counter() - start
            engine_calls.append(entry)
    pytesseract.image_to_string = observed_ocr
    original_engine = local_ocr.run_engine
    def observed_engine(image, languages=None):
        start = time.perf_counter()
        entry = {"lang": languages or settings.ocr_languages, "args": local_ocr.engine_args(languages),
                 "timeout": settings.ocr_timeout_seconds, "case_id": current["id"]}
        try:
            text = original_engine(image, languages=languages)
            raw_path = phase_folder / f"engine-{len(engine_calls) + 1}.txt"
            raw_path.write_text(text, encoding="utf-8")
            entry["raw_output"] = raw_path.relative_to(out).as_posix()
            entry["raw_output_sha256"] = sha256(raw_path)
            return text
        finally:
            entry["elapsed_seconds"] = time.perf_counter() - start
            engine_calls.append(entry)
    local_ocr.run_engine = observed_engine
    phase_folder = out / phase
    phase_folder.mkdir(exist_ok=False)
    try:
        with ExitStack() as stack:
            client = None
            if transport == "http":
                from unittest.mock import patch
                from fastapi.testclient import TestClient
                import api.deps as deps
                from api.routers import auth
                from api.main import app
                from app.security import AnonymousIdentityStore
                temp = stack.enter_context(tempfile.TemporaryDirectory(prefix="interview-ocr-benchmark-"))
                store = AnonymousIdentityStore(Path(temp) / "auth.db")
                stack.enter_context(patch.object(settings, "data_dir", temp))
                stack.enter_context(patch.object(auth, "identity_store", lambda: store))
                stack.enter_context(patch.object(deps, "identity_store", lambda: store))
                client = stack.enter_context(TestClient(app))
                token = client.post("/api/auth/anonymous").json()["token"]
                client.headers["Authorization"] = "Bearer " + token
            for current in manifest["cases"]:
                if selected and current["id"] not in selected.split(","):
                    continue
                path = out / current["path"]
                assert sha256(path) == current["input_sha256"], "Input changed"
                for repetition in range(1, repeats + 1):
                    started = time.perf_counter()
                    prediction, error, status, method = "", None, 200, ""
                    try:
                        if client:
                            endpoint = "resume" if path.suffix == ".pdf" else "jd-image"
                            # File only: no reference text field or model is used.
                            response = client.post("/api/uploads/" + endpoint,
                                                   files={"file": (path.name, path.read_bytes())})
                            status = response.status_code
                            if status == 200:
                                prediction = response.json()["text"]
                                method = response.json()["method"]
                            else:
                                error = response.json().get("detail")
                        else:
                            parsed = extract_resume_text(path)
                            prediction, method = parsed["text"], parsed["method"]
                    except Exception as exc:
                        error, status = type(exc).__name__, 0
                    elapsed = time.perf_counter() - started
                    output = phase_folder / (current["id"] + f"-{repetition}.txt")
                    output.write_text(prediction, encoding="utf-8")
                    row = {"id": current["id"], "language": current["language"], "group": current["group"],
                           "repetition": repetition, "elapsed_seconds": elapsed, "http_status": status,
                           "method": method, "error": error, "input_sha256": current["input_sha256"],
                           "truth_sha256": current["truth_sha256"], "output_sha256": sha256(output),
                           "output": output.relative_to(out).as_posix(), **score(current["text"], prediction, current["skills"])}
                    rows.append(row)
                    print(json.dumps({k: row[k] for k in ("id", "repetition", "http_status", "cer", "skills_recognized", "skills_total", "elapsed_seconds")}), flush=True)
    finally:
        pytesseract.image_to_string = original_ocr
        local_ocr.run_engine = original_engine
        core = [r for r in rows if r["group"] != "stress"]
        groups = {group: aggregate([r for r in rows if r["group"] == group]) for group in sorted({r["group"] for r in rows})}
        langs = {lang: aggregate([r for r in core if r["language"] == lang]) for lang in ("zh", "en", "mixed")}
        summary = {"phase": phase, "transport": transport, "repeats": repeats,
                   "llm_calls": 0, "search_api_calls": 0, "ground_truth_sent_to_parser": False,
                   "tesseract_version": str(pytesseract.get_tesseract_version()),
                   "parser_sha256": sha256(ROOT / "app/parsers/resume_parser.py"),
                   "ocr_module_sha256": sha256(ROOT / "app/parsers/ocr.py"),
                   "manifest_sha256": sha256(out / "manifest.json"), "engine_calls": engine_calls,
                   "core": aggregate(core), "overall_including_stress": aggregate(rows),
                   "by_group": groups, "core_by_language": langs, "rows": rows}
        summary["acceptance_passed"] = bool(core) and all(v["cer"] <= manifest["acceptance"]["core_normalized_cer_max"] and v["skill_recall"] >= manifest["acceptance"]["core_skill_recall_min"] and v["http_failures"] == 0 for v in langs.values() if v["runs"])
        summary["coverage_complete"] = len(rows) == len(manifest["cases"]) * repeats
        summary["character_threshold_passed"] = summary["coverage_complete"] and all(v["runs"] and v["cer"] <= manifest["acceptance"]["core_normalized_cer_max"] and v["http_failures"] == 0 for v in langs.values())
        summary["skill_threshold_passed"] = summary["coverage_complete"] and all(v["runs"] and v["skill_recall"] >= manifest["acceptance"]["core_skill_recall_min"] for v in langs.values())
        summary["acceptance_passed"] = summary["acceptance_passed"] and summary["coverage_complete"]
        result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"phase": phase, "core": summary["core"], "by_language": langs,
                          "acceptance_passed": summary["acceptance_passed"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--phase", default="final")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--selected", default="")
    parser.add_argument("--transport", choices=["direct", "http"], default="http")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("repeats must be 1..10")
    if args.prepare:
        prepare(args.out.resolve())
    else:
        run(args.out.resolve(), args.phase, args.repeats, args.selected, args.transport)


if __name__ == "__main__":
    main()
