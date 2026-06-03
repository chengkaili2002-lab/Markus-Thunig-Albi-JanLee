import argparse
from datetime import datetime
from pathlib import Path

from common import PROMPTS_DIR, RESULTS_DIR, ROOT_DIR, get_default_model
from schemas import Task5Result
from utils import build_task5_summary_row, run_task_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 5 open-ended routing and trigger analysis")
    parser.add_argument(
        "--model",
        default=get_default_model(),
        help="Ollama model name, for example gemma4:e2b-it-q4_K_M",
    )
    parser.add_argument(
        "--images-dir",
        default=str(ROOT_DIR / "data" / "sample_images"),
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
        output_dir = RESULTS_DIR / "task5" / "runs" / timestamp
    else:
        output_dir = Path(args.output_dir)

    run_task_batch(
        task_name="task5",
        prompt_file=PROMPTS_DIR / "task5_open_ended_router.txt",
        output_dir=output_dir,
        schema_model=Task5Result,
        model=args.model,
        images_dir=Path(args.images_dir),
        max_images=args.max_images,
        summary_row_builder=build_task5_summary_row,
    )


if __name__ == "__main__":
    main()
