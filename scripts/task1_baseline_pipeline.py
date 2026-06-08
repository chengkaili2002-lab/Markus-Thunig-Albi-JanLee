from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

from common import load_json_file, load_text_file, write_json_file
from ollama_client import call_ollama_with_image
from schemas import Task1BaselineResult
from task1_pipeline import list_dataset_images, resolve_dataset_image


def validate_baseline_inputs(
    *,
    ground_truth: List[Dict[str, Any]],
    dataset_dir: Path,
    prompt_file: Path,
) -> List[Dict[str, Any]]:
    if not ground_truth:
        raise ValueError("Ground truth is empty.")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    if not prompt_file.exists():
        raise FileNotFoundError(f"Baseline prompt file not found: {prompt_file}")

    dataset_images = list_dataset_images(dataset_dir)
    resolved: List[Dict[str, Any]] = []
    for index, entry in enumerate(ground_truth, start=1):
        image_name = str(entry.get("image_name", ""))
        image_path = resolve_dataset_image(image_name, dataset_images)
        resolved.append(
            {
                "order": index,
                "ground_truth": entry,
                "image_path": image_path,
            }
        )
    return resolved


def build_baseline_output_entry(
    *,
    item: Dict[str, Any],
    runtime_seconds: float,
    result_data: Optional[Dict[str, Any]],
    raw_response: str,
    error: Optional[str],
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "image_name": item["ground_truth"]["image_name"],
        "runtime_seconds": runtime_seconds,
    }
    if result_data:
        entry["result"] = result_data
        entry["raw_response"] = raw_response
    if error:
        entry["error"] = error
    return entry


def run_task1_baseline_groundtruth_ordered(
    *,
    ground_truth_path: Path,
    dataset_dir: Path,
    prompt_file: Path,
    output_file: Path,
    model: str,
    retries: int = 2,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    ground_truth = load_json_file(ground_truth_path)
    if not isinstance(ground_truth, list):
        raise ValueError(f"Ground truth must be a JSON array: {ground_truth_path}")

    resolved_items = validate_baseline_inputs(
        ground_truth=ground_truth,
        dataset_dir=dataset_dir,
        prompt_file=prompt_file,
    )

    if dry_run:
        mapping = [
            {
                "order": item["order"],
                "image_name": item["ground_truth"]["image_name"],
                "source_image": str(item["image_path"]),
            }
            for item in resolved_items
        ]
        write_json_file(output_file, mapping)
        return mapping

    prompt_text = load_text_file(prompt_file)
    outputs: List[Dict[str, Any]] = []

    for item in resolved_items:
        result_data: Optional[Dict[str, Any]] = None
        raw_response = ""
        error: Optional[str] = None
        runtime_seconds = 0.0

        for attempt in range(retries + 1):
            start_time = perf_counter()
            try:
                result_data, raw_response = call_ollama_with_image(
                    model=model,
                    prompt_text=prompt_text,
                    image_path=item["image_path"],
                    schema_model=Task1BaselineResult,
                )
                runtime_seconds = round(perf_counter() - start_time, 3)
                error = None
                break
            except Exception as exc:
                error = str(exc)
                runtime_seconds = round(perf_counter() - start_time, 3)
                if attempt >= retries:
                    break

        outputs.append(
            build_baseline_output_entry(
                item=item,
                runtime_seconds=runtime_seconds,
                result_data=result_data,
                raw_response=raw_response,
                error=error,
            )
        )
        print(
            f"[baseline] {item['order']}/{len(resolved_items)} "
            f"{item['ground_truth']['image_name']}",
            flush=True,
        )

    write_json_file(output_file, outputs)
    return outputs
