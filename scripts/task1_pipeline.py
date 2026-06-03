import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import (
    ensure_directory,
    list_sample_images,
    load_text_file,
    make_safe_relative_name,
    write_csv_file,
    write_json_file,
)
from ollama_client import call_ollama_with_images
from schemas import Task1Result


TASK1_SIGN_CONFIGS = [
    {
        "sign_key": "drive_right",
        "folder_name": "DRIVE_RIGHT",
        "prompt_file": "drive_right.txt",
        "result_folder": "drive right",
    },
    {
        "sign_key": "no_stop",
        "folder_name": "NO_STOP",
        "prompt_file": "no_stop.txt",
        "result_folder": "no stop",
    },
    {
        "sign_key": "priority_road",
        "folder_name": "PRIORITY_ROAD",
        "prompt_file": "priority.txt",
        "result_folder": "priority",
    },
    {
        "sign_key": "speed_limit_30",
        "folder_name": "SPEED_LIMIT_30",
        "prompt_file": "speed_limit_30.txt",
        "result_folder": "speed limit 30",
    },
    {
        "sign_key": "stop",
        "folder_name": "STOP",
        "prompt_file": "stop.txt",
        "result_folder": "stop",
    },
]


def render_task1_prompt(template_text: str, *, image_a: Path, image_b: Path) -> str:
    """
    Substitute the concrete Image A / Image B paths into a task 1 prompt template.

    The template uses simple placeholders so the same prompt file can be reused for
    the per-class 31.5.2026 test folders.
    """

    rendered = template_text.replace("{{IMAGE_A}}", str(image_a.resolve()))
    rendered = rendered.replace("{{IMAGE_B}}", str(image_b.resolve()))
    return rendered


def normalize_task1_result(result_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Keep Task 1 outputs internally consistent when the model returns conflicting fields.

    The prompt expects `dirty_count`, `overall_readability`, and `dirty_areas` to agree.
    We treat the listed `dirty_areas` as the source of truth because they are the most
    concrete part of the response.
    """

    normalized = dict(result_data)
    dirty_areas = list(normalized.get("dirty_areas") or [])
    actual_dirty_count = len(dirty_areas)
    reported_dirty_count = normalized.get("dirty_count", 0)
    surface_condition = normalized.get("surface_condition")

    warnings: List[str] = []

    if reported_dirty_count != actual_dirty_count:
        warnings.append(
            f"Adjusted dirty_count from {reported_dirty_count} to {actual_dirty_count} to match dirty_areas."
        )
        normalized["dirty_count"] = actual_dirty_count

    if surface_condition == "clean":
        if actual_dirty_count != 0:
            warnings.append("Reset dirty_areas for a clean result.")
            actual_dirty_count = 0
            normalized["dirty_areas"] = []
            normalized["dirty_count"] = 0
        normalized["overall_readability"] = "not_affected"
    elif surface_condition == "not_clean":
        if actual_dirty_count == 0:
            normalized["overall_readability"] = "uncertain"
            warnings.append("Set overall_readability to uncertain because no dirty areas were listed.")
        elif actual_dirty_count == 1:
            normalized["overall_readability"] = "partially_affected"
        else:
            normalized["overall_readability"] = "severely_affected"
    else:
        normalized["overall_readability"] = "uncertain"

    warning_text = " ".join(warnings) if warnings else None
    return normalized, warning_text


def build_task1_summary_row(
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
        "sign_type": "",
        "surface_condition": "",
        "dirty_count": "",
        "overall_readability": "",
        "evidence": "",
        "dirty_areas": "",
        "error": error_message or "",
    }

    if not result_data:
        return row

    row.update(
        {
            "sign_type": result_data.get("sign_type", ""),
            "surface_condition": result_data.get("surface_condition", ""),
            "dirty_count": result_data.get("dirty_count", ""),
            "overall_readability": result_data.get("overall_readability", ""),
            "evidence": result_data.get("evidence", ""),
            "dirty_areas": json.dumps(result_data.get("dirty_areas") or [], ensure_ascii=True),
        }
    )
    return row


def run_task1_pairwise_batch(
    *,
    output_dir: Path,
    model: str,
    dataset_root: Path,
    prompt_dir: Path,
    split_folder_name: str = "31.5.2026",
    max_images_per_class: Optional[int] = None,
    sign_keys: Optional[List[str]] = None,
) -> None:
    """
    Run Task 1 over the five sign folders in the custom 31.5.2026 dataset.

    For each sign type, the corresponding prompt file is paired with the sign's
    standard reference image and every test image inside the same 31.5.2026 folder.
    """

    json_dir = ensure_directory(output_dir / "json")
    ensure_directory(output_dir)

    summary_rows: List[Dict[str, Any]] = []

    selected_sign_keys = {item.strip().upper() for item in sign_keys} if sign_keys else None

    for config in TASK1_SIGN_CONFIGS:
        if selected_sign_keys is not None and config["folder_name"].upper() not in selected_sign_keys and config["sign_key"].upper() not in selected_sign_keys:
            continue

        sign_key = config["sign_key"]
        sign_folder = dataset_root / config["folder_name"]
        reference_image = sign_folder / "standard.png"
        test_dir = sign_folder / split_folder_name
        prompt_file = prompt_dir / config["prompt_file"]
        result_json_dir = ensure_directory(json_dir / config["result_folder"])

        if not prompt_file.exists():
            print(f"[task1] Missing prompt file: {prompt_file}")
            continue
        if not reference_image.exists():
            print(f"[task1] Missing reference image: {reference_image}")
            continue
        if not test_dir.exists():
            print(f"[task1] Missing test folder: {test_dir}")
            continue

        prompt_template = load_text_file(prompt_file)
        test_images = list_sample_images(test_dir)
        if max_images_per_class is not None:
            test_images = test_images[:max_images_per_class]

        for image_path in test_images:
            print(f"[task1] Processing {config['folder_name']}/{split_folder_name}/{image_path.name}")
            runtime_seconds = 0.0
            raw_text = ""
            result_data: Optional[Dict[str, Any]] = None
            error_message: Optional[str] = None
            warning_message: Optional[str] = None

            from time import perf_counter

            start_time = perf_counter()
            try:
                prompt_text = render_task1_prompt(
                    prompt_template,
                    image_a=reference_image,
                    image_b=image_path,
                )
                result_data, raw_text = call_ollama_with_images(
                    model=model,
                    prompt_text=prompt_text,
                    image_paths=[reference_image, image_path],
                    schema_model=Task1Result,
                )
                result_data, warning_message = normalize_task1_result(result_data)
            except Exception as exc:  # pragma: no cover - defensive baseline handling
                error_message = str(exc)
            runtime_seconds = round(perf_counter() - start_time, 3)

            output_name = f"{make_safe_relative_name(image_path, test_dir)}.json"
            output_path = result_json_dir / f"{sign_key}__{output_name}"

            payload = {
                "result": result_data,
                "task": "task1",
                "sign_key": sign_key,
                "test_image": image_path.name,
                "reference_image": reference_image.name,
            }
            if error_message:
                payload["error"] = error_message
            if warning_message:
                payload["warning"] = warning_message
            write_json_file(output_path, payload)

            row = build_task1_summary_row(
                image_path,
                test_dir,
                output_path,
                model,
                runtime_seconds,
                result_data,
                error_message,
            )
            if warning_message:
                row["warning"] = warning_message
            row["sign_key"] = sign_key
            row["reference_image_name"] = reference_image.name
            row["reference_image_absolute_path"] = str(reference_image.resolve())
            summary_rows.append(row)

    write_csv_file(output_dir / "task1_summary.csv", summary_rows)
    print(f"[task1] Finished. Wrote {len(summary_rows)} result files.")
