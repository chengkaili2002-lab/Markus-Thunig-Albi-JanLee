import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROMPTS_DIR = ROOT_DIR / "prompts"
RESULTS_DIR = ROOT_DIR / "results"

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_sample_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        return []

    images = [
        path
        for path in sorted(images_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return images


def make_safe_relative_name(image_path: Path, base_dir: Path) -> str:
    """
    Build a filename-safe identifier from the image path relative to the input folder.

    This keeps outputs unique even when different subfolders contain images with the
    same filename.
    """

    relative_path = image_path.relative_to(base_dir)
    safe_parts = [part.replace(" ", "_") for part in relative_path.parts]
    return "__".join(safe_parts)


def get_default_model() -> str:
    return os.environ.get("OLLAMA_MODEL", "gemma3n:e2b")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def serialize_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def write_csv_file(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_cell(row.get(key)) for key in fieldnames})
