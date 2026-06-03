import argparse
from pathlib import Path

from common import PROMPTS_DIR, RESULTS_DIR, ROOT_DIR, get_default_model
from task1_pipeline import run_task1_pairwise_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 1 traffic sign condition analysis")
    parser.add_argument(
        "--model",
        default=get_default_model(),
        help="Ollama model name, for example gemma3n:e2b",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(ROOT_DIR / "data" / "data" / "Task1_sign_impurity"),
        help="Root folder containing the five task 1 sign classes",
    )
    parser.add_argument(
        "--prompt-dir",
        default=str(PROMPTS_DIR / "task1_sign_condition"),
        help="Folder containing the five task 1 prompt templates",
    )
    parser.add_argument(
        "--split-folder",
        default="31.5.2026",
        help="Subfolder inside each class folder that contains the test images",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "31.5.2026"),
        help="Folder for JSON and CSV outputs",
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

    output_dir = Path(args.output_dir)
    sign_keys = [item.strip() for item in args.sign_keys.split(",") if item.strip()]

    run_task1_pairwise_batch(
        output_dir=output_dir,
        model=args.model,
        dataset_root=Path(args.dataset_root),
        prompt_dir=Path(args.prompt_dir),
        split_folder_name=args.split_folder,
        max_images_per_class=args.max_images_per_class,
        sign_keys=sign_keys,
    )


if __name__ == "__main__":
    main()
