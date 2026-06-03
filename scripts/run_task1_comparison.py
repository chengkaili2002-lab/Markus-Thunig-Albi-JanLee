import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from common import (
    PROMPTS_DIR,
    RESULTS_DIR,
    ROOT_DIR,
    ensure_directory,
    get_default_model,
    list_sample_images,
    load_text_file,
    make_safe_relative_name,
    write_csv_file,
    write_json_file,
)
from ollama_client import call_ollama_with_images
from task1_pipeline import (
    TASK1_SIGN_CONFIGS,
    run_task1_pairwise_batch,
)


class Task1BaselineResult(BaseModel):
    readability: str
    explanation: str


def build_task1_baseline_summary_row(
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
        "readability": "",
        "explanation": "",
        "error": error_message or "",
    }

    if not result_data:
        return row

    row.update(
        {
            "readability": result_data.get("readability", ""),
            "explanation": result_data.get("explanation", ""),
        }
    )
    return row


def run_task1_baseline_batch(
    *,
    output_dir: Path,
    model: str,
    dataset_root: Path,
    prompt_file: Path,
    split_folder_name: str = "31.5.2026",
    max_images_per_class: Optional[int] = None,
    sign_keys: Optional[List[str]] = None,
) -> None:
    """
    Run the single-image baseline prompt over the same Task1 image pairs.

    The improved path remains pairwise and uses the sign-specific prompt folder.
    The baseline path keeps the simpler readability schema and saves its own
    JSON and CSV files in a separate output tree.
    """

    json_dir = ensure_directory(output_dir / "json")
    ensure_directory(output_dir)

    if not prompt_file.exists():
        print(f"[task1-baseline] Missing prompt file: {prompt_file}")
        return

    prompt_text = load_text_file(prompt_file)
    summary_rows: List[Dict[str, Any]] = []
    selected_sign_keys = {item.strip().upper() for item in sign_keys} if sign_keys else None

    for config in TASK1_SIGN_CONFIGS:
        if selected_sign_keys is not None and config["folder_name"].upper() not in selected_sign_keys and config["sign_key"].upper() not in selected_sign_keys:
            continue

        sign_key = config["sign_key"]
        sign_folder = dataset_root / config["folder_name"]
        reference_image = sign_folder / "standard.png"
        test_dir = sign_folder / split_folder_name
        result_json_dir = ensure_directory(json_dir / config["result_folder"])

        if not reference_image.exists():
            print(f"[task1-baseline] Missing reference image: {reference_image}")
            continue
        if not test_dir.exists():
            print(f"[task1-baseline] Missing test folder: {test_dir}")
            continue

        test_images = list_sample_images(test_dir)
        if max_images_per_class is not None:
            test_images = test_images[:max_images_per_class]

        for image_path in test_images:
            print(f"[task1-baseline] Processing {config['folder_name']}/{split_folder_name}/{image_path.name}")
            runtime_seconds = 0.0
            result_data: Optional[Dict[str, Any]] = None
            error_message: Optional[str] = None

            from time import perf_counter

            start_time = perf_counter()
            try:
                result_data, raw_text = call_ollama_with_images(
                    model=model,
                    prompt_text=prompt_text,
                    image_paths=[reference_image, image_path],
                    schema_model=Task1BaselineResult,
                )
            except Exception as exc:  # pragma: no cover - defensive baseline handling
                error_message = str(exc)
                raw_text = ""

            runtime_seconds = round(perf_counter() - start_time, 3)

            output_name = f"{make_safe_relative_name(image_path, test_dir)}.json"
            output_path = result_json_dir / f"{sign_key}__{output_name}"

            payload: Dict[str, Any] = {
                "result": result_data,
                "task": "task1",
                "sign_key": sign_key,
                "test_image": image_path.name,
                "reference_image": reference_image.name,
            }
            if raw_text:
                payload["raw_response"] = raw_text
            if error_message:
                payload["error"] = error_message
            write_json_file(output_path, payload)

            row = build_task1_baseline_summary_row(
                image_path,
                test_dir,
                output_path,
                model,
                runtime_seconds,
                result_data,
                error_message,
            )
            row["sign_key"] = sign_key
            row["reference_image_name"] = reference_image.name
            row["reference_image_absolute_path"] = str(reference_image.resolve())
            summary_rows.append(row)

    write_csv_file(output_dir / "task1_summary.csv", summary_rows)
    print(f"[task1-baseline] Finished. Wrote {len(summary_rows)} result files.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Task 1 improved and baseline prompts")
    parser.add_argument(
        "--model",
        default=get_default_model(),
        help="Ollama model name, for example gemma4:e2b-it-q4_K_M",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(ROOT_DIR / "data" / "data" / "Task1_sign_impurity"),
        help="Root folder containing the five task 1 sign classes",
    )
    parser.add_argument(
        "--improved-prompt-dir",
        default=str(PROMPTS_DIR / "task1_sign_condition"),
        help="Folder containing the improved task 1 prompt templates",
    )
    parser.add_argument(
        "--baseline-prompt-file",
        default=str(PROMPTS_DIR / "task1_baseline.txt"),
        help="Baseline task 1 prompt file",
    )
    parser.add_argument(
        "--split-folder",
        default="31.5.2026",
        help="Subfolder inside each class folder that contains the test images",
    )
    parser.add_argument(
        "--output-root",
        default=str(RESULTS_DIR / "task1_comparison" / "2026-06-02"),
        help="Folder for the comparison run outputs",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=10,
        help="Maximum number of images to process per class folder",
    )
    parser.add_argument(
        "--sign-keys",
        default="DRIVE_RIGHT,NO_STOP,PRIORITY_ROAD,SPEED_LIMIT_30,STOP",
        help="Comma-separated sign keys or folder names to process",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    improved_output_dir = output_root / "improved"
    baseline_output_dir = output_root / "baseline"
    dataset_root = Path(args.dataset_root)
    sign_keys = [item.strip() for item in args.sign_keys.split(",") if item.strip()]

    # Keep the improved path exactly as before, only redirecting its output folder.
    run_task1_pairwise_batch(
        output_dir=improved_output_dir,
        model=args.model,
        dataset_root=dataset_root,
        prompt_dir=Path(args.improved_prompt_dir),
        split_folder_name=args.split_folder,
        max_images_per_class=args.max_images_per_class,
        sign_keys=sign_keys,
    )

    run_task1_baseline_batch(
        output_dir=baseline_output_dir,
        model=args.model,
        dataset_root=dataset_root,
        prompt_file=Path(args.baseline_prompt_file),
        split_folder_name=args.split_folder,
        max_images_per_class=args.max_images_per_class,
        sign_keys=sign_keys,
    )


if __name__ == "__main__":
    main()
