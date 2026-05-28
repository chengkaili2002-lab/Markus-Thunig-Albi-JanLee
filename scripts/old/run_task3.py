import argparse
from datetime import datetime
from pathlib import Path

from utils import PROMPTS_DIR, RESULTS_DIR, ROOT_DIR, Task3Result, get_default_model, run_task_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 3 lane detection failure validation")
    parser.add_argument(
        "--model",
        default=get_default_model(),
        help="Ollama model name, for example gemma4:e2b-it-q4_K_M",
    )
    parser.add_argument(
        "--images-dir",
        default=str(ROOT_DIR / "data" / "task3_test_5_20260512_141827"),
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
    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = RESULTS_DIR / "task3" / "runs" / timestamp
    else:
        output_dir = Path(args.output_dir)

    run_task_batch(
        task_name="task3",
        prompt_file=PROMPTS_DIR / "task3_lane_failure.txt",
        output_dir=output_dir,
        schema_model=Task3Result,
        model=args.model,
        images_dir=Path(args.images_dir),
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
