from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

from common import SUPPORTED_IMAGE_EXTENSIONS, load_json_file, load_text_file, write_json_file
from ollama_client import call_ollama_with_images
from schemas import Task1Result


@dataclass(frozen=True)
class SignConfig:
    sign_key: str
    prompt_file: str
    standard_image: str
    dataset_prefix: str


SIGN_CONFIGS: Dict[str, SignConfig] = {
    "drive_right": SignConfig("drive_right", "drive_right.txt", "drive right.png", "driveRight"),
    "no_stop": SignConfig("no_stop", "no_stop.txt", "no stop.png", "noStop"),
    "priority": SignConfig("priority", "priority.txt", "priority road.png", "priorityRoad"),
    "priority_road": SignConfig("priority", "priority.txt", "priority road.png", "priorityRoad"),
    "speed_limit": SignConfig(
        "speed_limit_30",
        "speed_limit_30.txt",
        "speed limit 30.png",
        "speed30",
    ),
    "speed_limit_30": SignConfig(
        "speed_limit_30",
        "speed_limit_30.txt",
        "speed limit 30.png",
        "speed30",
    ),
    "stop": SignConfig("stop", "stop.txt", "stop.png", "stop"),
}

GROUND_TRUTH_NAME_PATTERN = re.compile(
    r"^(DriveRight|NoStop|PriorityRoad|SpeedLimit30|Stop)(Clean|Impurity)No\.(\d+)$"
)

GROUND_TRUTH_PREFIXES = {
    "DriveRight": "driveRight",
    "NoStop": "noStop",
    "PriorityRoad": "priorityRoad",
    "SpeedLimit30": "speed30",
    "Stop": "stop",
}


def render_task1_prompt(template_text: str, *, image_a: Path, image_b: Path) -> str:
    rendered = template_text.replace("{{IMAGE_A}}", str(image_a.resolve()))
    return rendered.replace("{{IMAGE_B}}", str(image_b.resolve()))


def expected_dataset_stem(image_name: str) -> str:
    match = GROUND_TRUTH_NAME_PATTERN.fullmatch(image_name)
    if not match:
        raise ValueError(f"Unsupported ground-truth image_name: {image_name}")

    sign_name, condition, number = match.groups()
    return f"{GROUND_TRUTH_PREFIXES[sign_name]}_{condition.lower()}_{number}"


def list_dataset_images(dataset_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in dataset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def resolve_dataset_image(image_name: str, dataset_images: Iterable[Path]) -> Path:
    expected_stem = expected_dataset_stem(image_name).lower()
    exact_matches = [
        image_path
        for image_path in dataset_images
        if image_path.stem.lower() == expected_stem
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        names = [path.name for path in exact_matches]
        raise ValueError(f"Multiple exact dataset matches for {image_name}: {names}")

    pattern = re.compile(rf"^{re.escape(expected_stem)}($|\D)")
    prefix_matches = [
        image_path
        for image_path in dataset_images
        if pattern.match(image_path.stem.lower())
    ]
    if len(prefix_matches) != 1:
        names = [path.name for path in prefix_matches]
        raise ValueError(
            f"Expected one dataset image for {image_name} ({expected_stem}), "
            f"found {len(prefix_matches)}: {names}"
        )
    return prefix_matches[0]


def normalize_task1_result(
    result_data: Dict[str, Any],
    *,
    image_name: str,
    model: str,
    runtime_seconds: float,
) -> Dict[str, Any]:
    normalized = dict(result_data)
    normalized["image_name"] = image_name
    normalized["model"] = model
    normalized["runtime_seconds"] = runtime_seconds

    impurity_types = list(normalized.get("impurity_types") or [])
    if normalized.get("impurity_present") == "no":
        normalized["impurity_types"] = []
    elif impurity_types:
        normalized["impurity_present"] = "yes"

    return normalized


def validate_task1_inputs(
    *,
    ground_truth: List[Dict[str, Any]],
    dataset_dir: Path,
    prompt_dir: Path,
    sign_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    dataset_images = list_dataset_images(dataset_dir)
    standard_dir = dataset_dir / "standard"
    resolved: List[Dict[str, Any]] = []
    selected_sign_types = set(sign_types or [])

    if not ground_truth:
        raise ValueError("Ground truth is empty.")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    if not standard_dir.exists():
        raise FileNotFoundError(f"Standard reference folder not found: {standard_dir}")

    for index, entry in enumerate(ground_truth, start=1):
        image_name = str(entry.get("image_name", ""))
        sign_type = str(entry.get("sign_type", ""))
        config = SIGN_CONFIGS.get(sign_type)
        if not config:
            raise ValueError(f"Unsupported sign_type at groundtruth item {index}: {sign_type}")
        if selected_sign_types and sign_type not in selected_sign_types and config.sign_key not in selected_sign_types:
            continue

        image_path = resolve_dataset_image(image_name, dataset_images)
        prompt_path = prompt_dir / config.prompt_file
        standard_image = standard_dir / config.standard_image

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        if not standard_image.exists():
            raise FileNotFoundError(f"Standard image not found: {standard_image}")

        resolved.append(
            {
                "order": index,
                "ground_truth": entry,
                "config": config,
                "image_path": image_path,
                "prompt_path": prompt_path,
                "standard_image": standard_image,
            }
        )

    return resolved


def build_output_entry(
    *,
    item: Dict[str, Any],
    model: str,
    runtime_seconds: float,
    result_data: Optional[Dict[str, Any]],
    raw_response: str,
    error: Optional[str],
) -> Dict[str, Any]:
    ground_truth = item["ground_truth"]

    entry: Dict[str, Any] = {
        "image_name": ground_truth["image_name"],
        "runtime_seconds": runtime_seconds,
    }

    if result_data:
        entry["result"] = result_data
        entry["raw_response"] = raw_response
    if error:
        entry["error"] = error

    return entry


def run_task1_groundtruth_ordered(
    *,
    ground_truth_path: Path,
    dataset_dir: Path,
    prompt_dir: Path,
    output_file: Path,
    model: str,
    retries: int = 2,
    dry_run: bool = False,
    sign_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    ground_truth = load_json_file(ground_truth_path)
    if not isinstance(ground_truth, list):
        raise ValueError(f"Ground truth must be a JSON array: {ground_truth_path}")

    resolved_items = validate_task1_inputs(
        ground_truth=ground_truth,
        dataset_dir=dataset_dir,
        prompt_dir=prompt_dir,
        sign_types=sign_types,
    )

    if dry_run:
        mapping = [
            {
                "order": item["order"],
                "image_name": item["ground_truth"]["image_name"],
                "sign_type": item["config"].sign_key,
                "prompt_file": str(item["prompt_path"]),
                "reference_image": str(item["standard_image"]),
                "source_image": str(item["image_path"]),
            }
            for item in resolved_items
        ]
        write_json_file(output_file, mapping)
        return mapping

    outputs: List[Dict[str, Any]] = []
    for item in resolved_items:
        prompt_template = load_text_file(item["prompt_path"])
        prompt_text = render_task1_prompt(
            prompt_template,
            image_a=item["standard_image"],
            image_b=item["image_path"],
        )

        result_data: Optional[Dict[str, Any]] = None
        raw_response = ""
        error: Optional[str] = None
        runtime_seconds = 0.0

        for attempt in range(retries + 1):
            start_time = perf_counter()
            try:
                result_data, raw_response = call_ollama_with_images(
                    model=model,
                    prompt_text=prompt_text,
                    image_paths=[item["standard_image"], item["image_path"]],
                    schema_model=Task1Result,
                )
                runtime_seconds = round(perf_counter() - start_time, 3)
                result_data = normalize_task1_result(
                    result_data,
                    image_name=item["ground_truth"]["image_name"],
                    model=model,
                    runtime_seconds=runtime_seconds,
                )
                error = None
                break
            except Exception as exc:
                error = str(exc)
                runtime_seconds = round(perf_counter() - start_time, 3)
                if attempt >= retries:
                    break

        outputs.append(
            build_output_entry(
                item=item,
                model=model,
                runtime_seconds=runtime_seconds,
                result_data=result_data,
                raw_response=raw_response,
                error=error,
            )
        )
        print(
            f"[task1] {item['order']}/{len(resolved_items)} "
            f"{item['ground_truth']['image_name']}",
            flush=True,
        )

    write_json_file(output_file, outputs)
    return outputs
