import argparse
from pathlib import Path

from common import DATA_DIR, PROMPTS_DIR, RESULTS_DIR, get_default_model
from task1_pipeline import run_task1_groundtruth_ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Task1 one-shot anomaly detection for selected sign types."
    )
    parser.add_argument(
        "--sign-type",
        action="append",
        required=True,
        help="Sign type to run. Can be repeated. Example: --sign-type drive_right",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=RESULTS_DIR / "7.6.2026" / "groundtruth.json",
        help="Ground-truth JSON array. Output order follows this file exactly.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATA_DIR / "dataset",
        help="Dataset folder containing test images and the standard reference folder.",
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=PROMPTS_DIR / "task1_sign_condition",
        help="Folder containing the five Task1 prompt files.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=RESULTS_DIR / "7.6.2026" / "task1_sign_type_outputs.json",
        help="Combined JSON output file.",
    )
    parser.add_argument("--model", default=get_default_model(), help="Ollama model name.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_task1_groundtruth_ordered(
        ground_truth_path=args.ground_truth,
        dataset_dir=args.dataset,
        prompt_dir=args.prompt_dir,
        output_file=args.output_file,
        model=args.model,
        retries=args.retries,
        dry_run=args.dry_run,
        sign_types=args.sign_type,
    )
    print(f"[task1] Wrote {len(outputs)} entries to {args.output_file}")


if __name__ == "__main__":
    main()
