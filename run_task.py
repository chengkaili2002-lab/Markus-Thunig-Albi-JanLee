import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from utils import (
    PROMPTS_DIR,
    RESULTS_DIR,
    ROOT_DIR,
    Task1Result,
    Task2Result,
    Task3Result,
    Task4Result,
    build_task2_summary_row,
    get_default_model,
    run_task_batch,
)

TaskConfig = Dict[str, Optional[object]]

TASKS: Dict[str, TaskConfig] = {
    "task1": {
        "description": "Traffic sign condition analysis",
        "prompt_file": PROMPTS_DIR / "task1_sign_condition.txt",
        "schema_model": Task1Result,
        "default_images_dir": ROOT_DIR / "data" / "sample_images",
    },
    "task2": {
        "description": "Lane semantics analysis",
        "prompt_file": PROMPTS_DIR / "task2_lane_semantics.txt",
        "schema_model": Task2Result,
        "default_images_dir": ROOT_DIR / "data" / "sample_images",
        "summary_row_builder": build_task2_summary_row,
    },
    "task3": {
        "description": "Lane detection failure validation",
        "prompt_file": PROMPTS_DIR / "task3_lane_failure.txt",
        "schema_model": Task3Result,
        "default_images_dir": ROOT_DIR / "data" / "task3_test_5_20260512_141827",
    },
    "task4": {
        "description": "Infrastructure recognition",
        "prompt_file": PROMPTS_DIR / "task4_infrastructure.txt",
        "schema_model": Task4Result,
        "default_images_dir": ROOT_DIR / "data" / "sample_images",
    },
}


def get_task_config(task_name: str) -> TaskConfig:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}")
    return TASKS[task_name]


def build_default_output_dir(task_name: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return RESULTS_DIR / task_name / "runs" / timestamp


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run a traffic image analysis task")
    parser.add_argument(
        "--task",
        choices=list(TASKS.keys()),
        required=True,
        help="Which task to execute",
    )
    parser.add_argument(
        "--model",
        default=get_default_model(),
        help="Ollama model name, for example gemma3n:e2b",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Folder containing input images",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional custom folder for JSON and CSV outputs",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=5,
        help="Maximum number of images to process",
    )

    args = parser.parse_args(argv)
    config = get_task_config(args.task)

    images_dir = Path(args.images_dir) if args.images_dir is not None else config["default_images_dir"]
    output_dir = Path(args.output_dir) if args.output_dir is not None else build_default_output_dir(args.task)

    run_task_batch(
        task_name=args.task,
        prompt_file=config["prompt_file"],
        output_dir=output_dir,
        schema_model=config["schema_model"],
        model=args.model,
        images_dir=images_dir,
        max_images=args.max_images,
        summary_row_builder=config.get("summary_row_builder"),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
