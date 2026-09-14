"""Exercise real local OCR through HTTP using synthetic Chinese/English images.

No model or search request. Requires a running server and a Chinese font.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)


def sample(font_path: Path) -> Image.Image:
    image = Image.new("RGB", (1400, 420), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 44)
    for y, text in zip((40, 130, 220, 310), (
        "岗位：数据工程师", "技能：Python SQL",
        "职责：数据开发与数据分析", "要求：熟悉数据库和软件开发",
    )):
        draw.text((50, y), text, font=font, fill="black")
    return image


def run(base_url: str, font_path: Path) -> dict:
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=120, trust_env=False) as client:
        identity = client.post("/api/auth/anonymous")
        identity.raise_for_status()
        client.headers["Authorization"] = "Bearer " + identity.json()["token"]
        status = client.get("/api/self-check")
        status.raise_for_status()
        readiness = status.json()["ocr"]
        if not readiness["ready"] or not readiness["pdf"]["ready"]:
            raise RuntimeError("OCR dependency check failed: " + json.dumps(readiness))
        image = sample(font_path)
        results = []
        for route, name, fmt, media in (
            ("jd-image", "synthetic-jd.png", "PNG", "image/png"),
            ("resume", "synthetic-resume.png", "PNG", "image/png"),
            ("resume", "synthetic-scanned-resume.pdf", "PDF", "application/pdf"),
        ):
            data = io.BytesIO()
            image.save(data, format=fmt, resolution=150)
            response = client.post("/api/uploads/" + route,
                                   files={"file": (name, data.getvalue(), media)})
            response.raise_for_status()
            payload = response.json()
            text = "".join(payload["text"].split())
            expected = ("岗位", "Python", "SQL")
            if not all(word in text for word in expected) or "ocr" not in payload["method"]:
                raise RuntimeError("Synthetic OCR keywords or method missing for " + name)
            results.append({"file": name, "status": response.status_code,
                            "method": payload["method"], "characters": len(text),
                            "expected_keywords_present": True})
        return {"ocr": readiness, "uploads": results, "cloud_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    font = args.font or next((Path(p) for p in FONT_CANDIDATES if Path(p).is_file()), None)
    if font is None or not font.is_file():
        parser.error("Chinese font missing; specify --font /path/to/font.ttc")
    print(json.dumps(run(args.base_url, font), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
