import csv
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
from typing import Literal

import ollama
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROMPTS_DIR = ROOT_DIR / "prompts"
RESULTS_DIR = ROOT_DIR / "results"

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Task1Result(BaseModel):
    traffic_sign_visible: bool = Field(description="Whether a traffic sign is visible")
    sign_type: str = Field(description="Type of the sign")
    condition: Literal["clean", "dirty", "sticker", "vandalized", "invalidated", "unclear"]
    occlusion: Literal["none", "partial", "heavy", "unclear"]
    damage: Literal["none", "minor", "severe", "unclear"]
    readability: Literal["readable", "partly_readable", "unreadable", "unclear"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence between 0.0 and 1.0")
    explanation: str = Field(description="Short explanation")


class Task2Lane(BaseModel):
    lane_index: int = Field(ge=1, description="Lane number from left to right or top to bottom")
    visible_arrow_type: Literal[
        "straight",
        "left",
        "right",
        "straight+left",
        "straight+right",
        "merge_left",
        "merge_right",
        "unclear",
    ]
    allowed_movements: List[Literal["straight", "left", "right", "merge_left", "merge_right", "unclear"]]
    special_regulation: Literal["none", "bus_only", "truck_restriction", "turn_restriction", "unclear"]
    semantic_interpretation: str = Field(description="Short lane-level interpretation")


class Task2Result(BaseModel):
    arrow_sign_visible: bool = Field(description="Whether any relevant arrow sign is visible")
    lane_count: int = Field(ge=0, description="Number of lanes described in the output")
    lanes: List[Task2Lane] = Field(description="Lane-by-lane semantic interpretation")
    opposing_directions: bool = Field(description="Whether the sign contains opposing directions")
    additional_lane_regulations: List[str] = Field(description="Extra lane regulations as short phrases")
    overall_lane_semantics: str = Field(description="Short overall lane meaning")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence between 0.0 and 1.0")
    explanation: str = Field(description="Short explanation")


class Task3Result(BaseModel):
    lane_detection_assessment: Literal["correct", "incorrect", "unclear"]
    lane_marking_continuity: Literal["continuous", "discontinuous", "unclear"]
    likely_reason: Literal[
        "road_topology",
        "temporal_occlusion",
        "camera_interference",
        "poor_markings",
        "intersection_geometry",
        "unclear",
    ]
    supporting_signals: List[str] = Field(description="Short evidence phrases")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence between 0.0 and 1.0")
    explanation: str = Field(description="Short explanation")


class Task4Result(BaseModel):
    tunnel_visible: bool = Field(description="Whether a tunnel is visible")
    bridge_visible: bool = Field(description="Whether a bridge is visible")
    tall_buildings_visible: bool = Field(description="Whether tall buildings are visible")
    urban_canyon: bool = Field(description="Whether the scene looks like an urban canyon")
    gps_risk: Literal["low", "medium", "high", "unclear"]
    explanation: str = Field(description="Short explanation")


class Task5StandardInformation(BaseModel):
    road_type: str = Field(description="Short road type description")
    visible_traffic_signs: List[str] = Field(description="Visible traffic signs as short phrases")
    visible_road_users: List[str] = Field(description="Visible road users as short phrases")
    visible_infrastructure: List[str] = Field(description="Visible infrastructure as short phrases")
    image_quality_issues: List[str] = Field(description="Image quality issues as short phrases")


class Task5Result(BaseModel):
    scene_category: Literal[
        "urban_road",
        "highway",
        "intersection",
        "tunnel_or_underpass",
        "parking_or_private_area",
        "unclear",
    ]
    traffic_sign_visible: bool
    traffic_sign_condition_relevant: bool
    arrow_or_lane_direction_sign_visible: bool
    lane_detection_overlay_visible: bool
    gps_risk_infrastructure_visible: bool
    standard_information: Task5StandardInformation
    recommended_tasks: List[
        Literal[
            "task1_sign_condition",
            "task2_lane_semantics",
            "task3_lane_failure_validation",
            "task4_infrastructure",
            "skip",
        ]
    ]
    priority: Literal["low", "medium", "high", "unclear"]
    resource_strategy: Literal[
        "skip",
        "run_single_task",
        "run_multiple_tasks",
        "send_to_smartphone_validation",
        "unclear",
    ]
    trigger_reason: str = Field(description="Short reason for the recommended routing decision")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence between 0.0 and 1.0")


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


def extract_json_text(text: str) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]

    return cleaned


def parse_json_response(text: str) -> Dict[str, Any]:
    json_text = extract_json_text(text)
    return json.loads(json_text)


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


def call_ollama_with_image(
    *,
    model: str,
    prompt_text: str,
    image_path: Path,
    schema_model: Type[BaseModel],
) -> Tuple[Dict[str, Any], str]:
    """
    Send one image and one fixed prompt to a local Ollama model.

    The response is requested as structured JSON, then validated with Pydantic.
    """

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt_text},
            {
                "role": "user",
                "content": "Analyze the attached traffic-scene image and return JSON only.",
                "images": [str(image_path)],
            },
        ],
        format=schema_model.model_json_schema(),
        options={"temperature": 0},
        stream=False,
    )

    raw_text = response.message.content
    parsed = schema_model.model_validate_json(extract_json_text(raw_text))
    return parsed.model_dump(), raw_text


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
