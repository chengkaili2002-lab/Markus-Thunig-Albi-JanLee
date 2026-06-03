import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel

from common import (
    DATA_DIR,
    ensure_directory,
    list_sample_images,
    load_text_file,
    make_safe_relative_name,
    write_csv_file,
    write_json_file,
)
from ollama_client import call_ollama_with_image


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

        from time import perf_counter

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
