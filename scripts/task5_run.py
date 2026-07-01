"""Task 5 — workflow: load images, call Ollama, save structured results."""
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import ollama

from task5_schema import Task5Output

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data" / "dataset"
OUT   = ROOT / "results" / "task5"
MODEL = "minicpm-v4.5:8b"
EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _clean(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return s


def run(prompt_text: str, max_images: int = 0):
    images = sorted(p for p in DATA.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    if max_images > 0:
        images = images[:max_images]
    if not images:
        print("[task5] No images found."); return

    jd = OUT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    jd.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img.name}")
        t0 = perf_counter()
        try:
            r = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system",  "content": prompt_text},
                    {"role": "user",    "content": "Analyze this street-view image. Return JSON only.",
                     "images": [str(img)]},
                ],
                format=Task5Output.model_json_schema(),
                options={"temperature": 0},
            )
            result = Task5Output.model_validate_json(_clean(r.message.content)).model_dump()
            err = None
        except Exception as e:
            result, err = None, str(e)

        dt = round(perf_counter() - t0, 2)
        (jd / f"{img.stem}.json").write_text(
            json.dumps({"image": img.name, "time_s": dt, "result": result, "error": err},
                       indent=2, ensure_ascii=False),
            encoding="utf-8")

    print(f"[task5] Done. {len(images)} images → {OUT}")
