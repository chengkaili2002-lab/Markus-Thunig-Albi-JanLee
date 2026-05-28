import csv
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
from time import perf_counter

import ollama
from pydantic import BaseModel
from model_API import (
    Task1Result, Task2Lane, Task2Result, Task3Result, Task4Result, Task5Result, Task5StandardInformation,
    call_ollama_with_image, extract_json_text
)

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

def parse_json_response(text: str) -> Dict[str, Any]:
    json_text = extract_json_text(text)
    return json.loads(json_text)

# TODO: What is the reason for this function??? It seems as its not used
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


def build_task2_summary_row(
    image_path: Path,
    sample_dir: Path,
    output_path: Path,
    model: str,
    runtime_seconds: float,
    result_data: Optional[Dict[str, Any]],
    error_message: Optional[str],
) -> Dict[str, Any]:
    """
    Build a compact Task 2 CSV row that is easy to scan by eye.

    The nested lane list is preserved in JSON, while the CSV keeps one row per
    image with the main semantic signals flattened into readable columns.
    """

    relative_path = str(image_path.relative_to(sample_dir))
    row: Dict[str, Any] = {
        "source_image_name": image_path.name,
        "source_image_relative_path": relative_path,
        "source_image_absolute_path": str(image_path.resolve()),
        "model": model,
        "runtime_seconds": runtime_seconds,
        "output_json": str(output_path),
        "arrow_sign_visible": "",
        "lane_count": "",
        "lane_types": "",
        "allowed_movements": "",
        "special_regulations": "",
        "opposing_directions": "",
        "additional_lane_regulations": "",
        "overall_lane_semantics": "",
        "confidence": "",
        "error": error_message or "",
    }

    if not result_data:
        return row

    lanes = result_data.get("lanes") or []
    lane_types: List[str] = []
    allowed_movements: List[str] = []
    special_regs: List[str] = []

    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        lane_types.append(f"{lane.get('lane_index', '?')}:{lane.get('visible_arrow_type', 'unclear')}")
        movements = lane.get("allowed_movements") or []
        if isinstance(movements, list):
            allowed_movements.append(
                f"{lane.get('lane_index', '?')}:{'|'.join(str(item) for item in movements)}"
            )
        special_regs.append(f"{lane.get('lane_index', '?')}:{lane.get('special_regulation', 'unclear')}")

    additional_regs = result_data.get("additional_lane_regulations") or []

    row.update(
        {
            "arrow_sign_visible": result_data.get("arrow_sign_visible", ""),
            "lane_count": result_data.get("lane_count", ""),
            "lane_types": "; ".join(lane_types),
            "allowed_movements": "; ".join(allowed_movements),
            "special_regulations": "; ".join(special_regs),
            "opposing_directions": result_data.get("opposing_directions", ""),
            "additional_lane_regulations": "; ".join(str(item) for item in additional_regs),
            "overall_lane_semantics": result_data.get("overall_lane_semantics", ""),
            "confidence": result_data.get("confidence", ""),
        }
    )
    return row


def build_task5_summary_row(
    image_path: Path,
    sample_dir: Path,
    output_path: Path,
    model: str,
    runtime_seconds: float,
    result_data: Optional[Dict[str, Any]],
    error_message: Optional[str],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "source_image_name": image_path.name,
        "source_image_relative_path": str(image_path.relative_to(sample_dir)),
        "source_image_absolute_path": str(image_path.resolve()),
        "model": model,
        "runtime_seconds": runtime_seconds,
        "output_json": str(output_path),
        "scene_category": "",
        "traffic_sign_visible": "",
        "traffic_sign_condition_relevant": "",
        "arrow_or_lane_direction_sign_visible": "",
        "lane_detection_overlay_visible": "",
        "gps_risk_infrastructure_visible": "",
        "recommended_tasks": "",
        "priority": "",
        "resource_strategy": "",
        "trigger_reason": "",
        "confidence": "",
        "error": error_message or "",
    }

    if not result_data:
        return row

    row.update(
        {
            "scene_category": result_data.get("scene_category", ""),
            "traffic_sign_visible": result_data.get("traffic_sign_visible", ""),
            "traffic_sign_condition_relevant": result_data.get("traffic_sign_condition_relevant", ""),
            "arrow_or_lane_direction_sign_visible": result_data.get("arrow_or_lane_direction_sign_visible", ""),
            "lane_detection_overlay_visible": result_data.get("lane_detection_overlay_visible", ""),
            "gps_risk_infrastructure_visible": result_data.get("gps_risk_infrastructure_visible", ""),
            "recommended_tasks": "; ".join(result_data.get("recommended_tasks") or []),
            "priority": result_data.get("priority", ""),
            "resource_strategy": result_data.get("resource_strategy", ""),
            "trigger_reason": result_data.get("trigger_reason", ""),
            "confidence": result_data.get("confidence", ""),
        }
    )

    standard_info = result_data.get("standard_information") or {}
    if isinstance(standard_info, dict):
        row.update(
            {
                "road_type": standard_info.get("road_type", ""),
                "visible_traffic_signs": "; ".join(standard_info.get("visible_traffic_signs") or []),
                "visible_road_users": "; ".join(standard_info.get("visible_road_users") or []),
                "visible_infrastructure": "; ".join(standard_info.get("visible_infrastructure") or []),
                "image_quality_issues": "; ".join(standard_info.get("image_quality_issues") or []),
            }
        )

    return row


def serialize_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)

def run_task_batch(
    *,
    task_name: str,
    prompt_file: Path,
    output_dir: Path,
    schema_model: Type[BaseModel],
    model: str,
    images_dir: Optional[Path] = None,
    max_images: Optional[int] = None,
    summary_row_builder: Optional[
        Callable[[Path, Path, Path, str, float, Optional[Dict[str, Any]], Optional[str]], Dict[str, Any]]
    ] = None,
) -> None:
    prompt_text = load_text_file(prompt_file)
    sample_dir = images_dir if images_dir is not None else DATA_DIR / "sample_images"
    images = list_sample_images(sample_dir)
    if max_images is not None:
        images = images[:max_images]

    if not images:
        print(f"[{task_name}] No images found in {sample_dir}")
        return

    json_dir = ensure_directory(output_dir / "json")
    ensure_directory(output_dir)

    summary_rows: List[Dict[str, Any]] = []

    for image_path in images:
        print(f"[{task_name}] Processing {image_path.relative_to(sample_dir)}")
        runtime_seconds = 0.0
        raw_text = ""
        result_data: Optional[Dict[str, Any]] = None
        error_message: Optional[str] = None

        start_time = perf_counter()
        try:
            result_data, raw_text = call_ollama_with_image(
                model=model,
                prompt_text=prompt_text,
                image_path=image_path,
                schema_model=schema_model,
            )
        except Exception as exc:  # pragma: no cover - defensive baseline handling
            error_message = str(exc)
        runtime_seconds = round(perf_counter() - start_time, 3)

        output_name = f"{make_safe_relative_name(image_path, sample_dir)}.json"
        output_path = json_dir / output_name

        payload = {
            "task": task_name,
            "source_image_name": image_path.name,
            "source_image_relative_path": str(image_path.relative_to(sample_dir)),
            "source_image_absolute_path": str(image_path.resolve()),
            "image_file": image_path.name,
            "image_relative_path": str(image_path.relative_to(sample_dir)),
            "image_path": str(image_path),
            "model": model,
            "runtime_seconds": runtime_seconds,
            "result": result_data,
            "raw_response": raw_text,
            "error": error_message,
        }
        write_json_file(output_path, payload)

        if summary_row_builder is not None:
            row = summary_row_builder(
                image_path,
                sample_dir,
                output_path,
                model,
                runtime_seconds,
                result_data,
                error_message,
            )
        else:
            row = {
                "source_image_name": image_path.name,
                "source_image_relative_path": str(image_path.relative_to(sample_dir)),
                "source_image_absolute_path": str(image_path.resolve()),
                "image_file": image_path.name,
                "image_relative_path": str(image_path.relative_to(sample_dir)),
                "image_path": str(image_path),
                "model": model,
                "runtime_seconds": runtime_seconds,
                "output_json": str(output_path),
                "error": error_message or "",
            }
            if result_data is not None:
                row.update(result_data)
        summary_rows.append(row)

    write_csv_file(output_dir / f"{task_name}_summary.csv", summary_rows)
    print(f"[{task_name}] Finished. Wrote {len(summary_rows)} result files.")
