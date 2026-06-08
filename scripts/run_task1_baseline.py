import argparse
from pathlib import Path

from common import DATA_DIR, PROMPTS_DIR, RESULTS_DIR, get_default_model
from task1_baseline_pipeline import run_task1_baseline_groundtruth_ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Task1 baseline zero-shot anomaly detection in ground-truth order."
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
        help="Dataset folder containing test images.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=PROMPTS_DIR / "task1_baseline.txt",
        help="Baseline zero-shot prompt file.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=RESULTS_DIR / "7.6.2026" / "task1_baseline_outputs.json",
        help="Combined baseline JSON output file.",
    )
    parser.add_argument("--model", default=get_default_model(), help="Ollama model name.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_task1_baseline_groundtruth_ordered(
        ground_truth_path=args.ground_truth,
        dataset_dir=args.dataset,
        prompt_file=args.prompt_file,
        output_file=args.output_file,
        model=args.model,
        retries=args.retries,
        dry_run=args.dry_run,
    )
    print(f"[baseline] Wrote {len(outputs)} entries to {args.output_file}")


if __name__ == "__main__":
    main()
