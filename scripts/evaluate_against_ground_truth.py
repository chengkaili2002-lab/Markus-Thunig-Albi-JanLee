"""
Meeting-readable evaluation for manual ground truth vs. model outputs.

Outputs:
  - results/evaluation/evaluation_summary.md
  - results/evaluation/evaluation_details.csv

This script compares only the structured ground-truth fields requested for each
task. Confidence and explanation are excluded from automatic scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GROUND_TRUTH = ROOT_DIR / "data" / "ground_truth" / "ground_truth_manual.json"
OUTPUT_DIR = ROOT_DIR / "results" / "evaluation"
SUMMARY_PATH = OUTPUT_DIR / "evaluation_summary.md"
DETAILS_PATH = OUTPUT_DIR / "evaluation_details.csv"

SUPPORTED_TASKS = ("task1", "task2", "task3", "task4")


TASK_RULES: Dict[str, Dict[str, Any]] = {
    "task1": {
        "fields": [
            "traffic_sign_visible",
            "sign_type",
            "condition",
            "occlusion",
            "damage",
            "readability",
        ],
        "critical_fields": ["traffic_sign_visible", "sign_type", "readability"],
    },
    "task2": {
        "fields": [
            "lane_count",
            "visible_arrow_count",
            "visible_arrow_type",
            "allowed_movements",
            "merge_behavior",
            "special_regulation",
        ],
        "critical_fields": [
            "lane_count",
            "visible_arrow_count",
            "visible_arrow_type",
            "allowed_movements",
            "merge_behavior",
        ],
    },
    "task3": {
        "fields": [
            "detection_quality",
            "lane_markings_in_scene",
            "detection_failure_visible",
            "failure_reasons",
            "problematic_regions",
        ],
        "critical_fields": [
            "detection_quality",
            "lane_markings_in_scene",
            "detection_failure_visible",
        ],
    },
    "task4": {
        "fields": [
            "tunnel_visible",
            "bridge_visible",
            "tall_buildings_visible",
            "urban_canyon",
            "gps_risk",
        ],
        "critical_fields": [
            "tunnel_visible",
            "bridge_visible",
            "urban_canyon",
            "gps_risk",
        ],
    },
}


@dataclass
class Comparison:
    field: str
    expected: Any
    actual: Any
    match: bool


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "image_id",
                "model_output_found",
                "entry_status",
                "field_accuracy",
                "wrong_fields",
                "main_error_reason",
                "short_comment",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def repair_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    replacements = {
        "璇句欢": "课件",
        "灞忓箷鎴浘": "屏幕截图",
    }
    repaired = value
    for old, new in replacements.items():
        repaired = repaired.replace(old, new)
    return repaired


def normalize_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = text.replace("&", " and ")
    text = text.replace("+", "_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or None


def normalize_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    return bool(value)


def normalize_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_list_of_labels(value: Any) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        items = [part for part in value.split(";")]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
    else:
        return None
    normalized = [normalize_label(item) for item in items]
    normalized = [item for item in normalized if item]
    return tuple(sorted(normalized))


def compare_bool(expected: Any, actual: Any) -> bool:
    return normalize_bool(expected) == normalize_bool(actual)


def compare_int(expected: Any, actual: Any) -> bool:
    return normalize_int(expected) == normalize_int(actual)


def compare_label(expected: Any, actual: Any) -> bool:
    return normalize_label(expected) == normalize_label(actual)


def compare_label_set(expected: Any, actual: Any) -> bool:
    return normalize_list_of_labels(expected) == normalize_list_of_labels(actual)


def compare_list_set(expected: Any, actual: Any) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    if not isinstance(expected, Sequence) or not isinstance(actual, Sequence):
        return False
    if isinstance(expected, (str, bytes)) or isinstance(actual, (str, bytes)):
        return compare_label(expected, actual)
    expected_norm = sorted(normalize_label(item) for item in expected if normalize_label(item))
    actual_norm = sorted(normalize_label(item) for item in actual if normalize_label(item))
    return expected_norm == actual_norm


def load_ground_truth(path: Path) -> List[Dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError("ground_truth_manual.json must contain a JSON array")
    return payload


def resolve_model_output_path(task_id: str, image_id: str, explicit_path: Any) -> Optional[Path]:
    explicit = repair_text(explicit_path)
    if isinstance(explicit, str) and explicit.strip():
        candidate = Path(explicit)
        if candidate.exists():
            return candidate

    search_root = ROOT_DIR / "results" / task_id
    if not search_root.exists():
        return None

    digits = re.sub(r"\D+", "", str(image_id))
    stem = Path(str(image_id)).stem
    matches: List[Path] = []
    for candidate in search_root.rglob("*.json"):
        name = candidate.name
        candidate_digits = re.sub(r"\D+", "", name)
        if digits and digits == candidate_digits:
            matches.append(candidate)
            continue
        if digits and digits and digits in candidate_digits:
            matches.append(candidate)
            continue
        if stem and stem in name:
            matches.append(candidate)

    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def load_model_result(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    payload = load_json(path)
    if not isinstance(payload, dict):
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    return payload


def task2_actual_lane_values(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    lanes = result.get("lanes", [])
    actual_lanes: List[Dict[str, Any]] = []
    if not isinstance(lanes, list):
        return actual_lanes

    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        visible_arrow_type = lane.get("visible_arrow_type")
        if normalize_label(visible_arrow_type) == "merge_left":
            merge_behavior = "merge_left"
        elif normalize_label(visible_arrow_type) == "merge_right":
            merge_behavior = "merge_right"
        else:
            merge_behavior = "none"
        actual_lanes.append(
            {
                "visible_arrow_type": visible_arrow_type,
                "allowed_movements": lane.get("allowed_movements", []),
                "merge_behavior": merge_behavior,
                "special_regulation": lane.get("special_regulation"),
            }
        )
    return actual_lanes


def task3_derived_lists(result: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    likely_reason = normalize_label(result.get("likely_reason"))
    failure_reasons = [likely_reason] if likely_reason else []
    supporting = result.get("supporting_signals", [])
    problematic_regions: List[str] = []
    if isinstance(supporting, list):
        for item in supporting:
            label = normalize_label(item)
            if label:
                problematic_regions.append(label)
    return failure_reasons, problematic_regions


def compare_entry_field(field: str, expected: Any, actual: Any, task_id: str) -> bool:
    if task_id == "task1":
        if field == "traffic_sign_visible":
            return compare_bool(expected, actual)
        if field == "sign_type":
            return compare_label_set(expected, actual)
        return compare_label(expected, actual)

    if task_id == "task2":
        if field in {"lane_count", "visible_arrow_count"}:
            return compare_int(expected, actual)
        if field == "allowed_movements":
            return compare_list_set(expected, actual)
        return compare_label(expected, actual)

    if task_id == "task3":
        if field == "detection_failure_visible":
            return compare_bool(expected, actual)
        if field in {"failure_reasons", "problematic_regions"}:
            return compare_list_set(expected, actual)
        return compare_label(expected, actual)

    if task_id == "task4":
        if field in {"tunnel_visible", "bridge_visible", "tall_buildings_visible", "urban_canyon"}:
            return compare_bool(expected, actual)
        return compare_label(expected, actual)

    return False


def task3_partial_field_score(field: str, expected: Any, actual: Any) -> float:
    if compare_entry_field(field, expected, actual, "task3"):
        return 1.0

    if field == "detection_quality":
        ordered = {"incorrect": 0, "partially_correct": 1, "correct": 2}
        expected_label = normalize_label(expected)
        actual_label = normalize_label(actual)
        if expected_label in ordered and actual_label in ordered and abs(ordered[expected_label] - ordered[actual_label]) == 1:
            return 0.5
        return 0.0

    if field == "lane_markings_in_scene":
        expected_label = normalize_label(expected)
        actual_label = normalize_label(actual)
        if {expected_label, actual_label} == {"continuous", "discontinuous"}:
            return 0.5
        if expected_label in {"weak", "discontinuous"} and actual_label in {"continuous", "discontinuous", "weak"}:
            return 0.5
        return 0.0

    if field in {"failure_reasons", "problematic_regions"}:
        expected_set = set(normalize_list_of_labels(expected) or ())
        actual_set = set(normalize_list_of_labels(actual) or ())
        if expected_set and actual_set and expected_set.intersection(actual_set):
            return 0.5
        return 0.0

    return 0.0


def compare_record(record: Dict[str, Any]) -> Dict[str, Any]:
    task_id = str(record.get("task_id", "")).strip()
    image_id = repair_text(record.get("image_id", ""))
    image_path = repair_text(record.get("image_path", ""))
    model_output_json_path = record.get("model_output_json_path", "")
    gt = record.get("ground_truth", {})
    if task_id not in SUPPORTED_TASKS or not isinstance(gt, dict):
        return {
            "task_id": task_id,
            "image_id": image_id,
            "model_output_found": False,
            "entry_status": "incorrect",
            "field_accuracy": "",
            "wrong_fields": "invalid_ground_truth_entry",
            "main_error_reason": "invalid_ground_truth_entry",
            "short_comment": "Ground truth entry is malformed.",
            "field_count": 0,
            "matched_count": 0,
            "sort_key": (4, task_id, image_id),
        }

    resolved_path = resolve_model_output_path(task_id, image_id, model_output_json_path)
    model_payload = load_model_result(resolved_path)
    if model_payload is None:
        return {
            "task_id": task_id,
            "image_id": image_id,
            "model_output_found": False,
            "entry_status": "missing_model_output",
            "field_accuracy": "",
            "wrong_fields": "model_output_missing",
            "main_error_reason": "model_output_missing",
            "short_comment": "No model output JSON file found.",
            "field_count": 0,
            "matched_count": 0,
            "sort_key": (4, task_id, image_id),
        }

    result = model_payload["result"]
    comparisons: List[Comparison] = []
    expected_fields = TASK_RULES[task_id]["fields"]

    if task_id == "task1":
        for field in expected_fields:
            comparisons.append(
                Comparison(
                    field=field,
                    expected=gt.get(field),
                    actual=result.get(field),
                    match=compare_entry_field(field, gt.get(field), result.get(field), task_id),
                )
            )

    elif task_id == "task2":
        comparisons.append(
            Comparison(
                field="lane_count",
                expected=gt.get("lane_count"),
                actual=result.get("lane_count"),
                match=compare_entry_field("lane_count", gt.get("lane_count"), result.get("lane_count"), task_id),
            )
        )
        actual_visible_arrow_count = len(result.get("lanes", [])) if isinstance(result.get("lanes"), list) else 0
        comparisons.append(
            Comparison(
                field="visible_arrow_count",
                expected=gt.get("visible_arrow_count"),
                actual=actual_visible_arrow_count,
                match=compare_entry_field("visible_arrow_count", gt.get("visible_arrow_count"), actual_visible_arrow_count, task_id),
            )
        )

        expected_lanes = gt.get("lanes", [])
        actual_lanes = task2_actual_lane_values(result)
        max_lanes = max(len(expected_lanes) if isinstance(expected_lanes, list) else 0, len(actual_lanes))
        for idx in range(max_lanes):
            expected_lane = expected_lanes[idx] if isinstance(expected_lanes, list) and idx < len(expected_lanes) and isinstance(expected_lanes[idx], dict) else {}
            actual_lane = actual_lanes[idx] if idx < len(actual_lanes) else {}
            for field in ("visible_arrow_type", "allowed_movements", "merge_behavior", "special_regulation"):
                expected_value = expected_lane.get(field)
                actual_value = actual_lane.get(field)
                comparisons.append(
                    Comparison(
                        field=f"lanes[{idx}].{field}",
                        expected=expected_value,
                        actual=actual_value,
                        match=compare_entry_field(field, expected_value, actual_value, task_id),
                    )
                )

    elif task_id == "task3":
        actual_failure_reasons, actual_problematic_regions = task3_derived_lists(result)
        comparisons.append(
            Comparison(
                field="detection_quality",
                expected=gt.get("detection_quality"),
                actual=result.get("lane_detection_assessment"),
                match=compare_entry_field("detection_quality", gt.get("detection_quality"), result.get("lane_detection_assessment"), task_id),
            )
        )
        comparisons.append(
            Comparison(
                field="lane_markings_in_scene",
                expected=gt.get("lane_markings_in_scene"),
                actual=result.get("lane_marking_continuity"),
                match=compare_entry_field("lane_markings_in_scene", gt.get("lane_markings_in_scene"), result.get("lane_marking_continuity"), task_id),
            )
        )
        derived_failure_visible = normalize_label(result.get("lane_detection_assessment")) in {"partially_correct", "incorrect"}
        comparisons.append(
            Comparison(
                field="detection_failure_visible",
                expected=gt.get("detection_failure_visible"),
                actual=derived_failure_visible,
                match=compare_entry_field("detection_failure_visible", gt.get("detection_failure_visible"), derived_failure_visible, task_id),
            )
        )
        comparisons.append(
            Comparison(
                field="failure_reasons",
                expected=gt.get("failure_reasons"),
                actual=actual_failure_reasons,
                match=compare_entry_field("failure_reasons", gt.get("failure_reasons"), actual_failure_reasons, task_id),
            )
        )
        comparisons.append(
            Comparison(
                field="problematic_regions",
                expected=gt.get("problematic_regions"),
                actual=actual_problematic_regions,
                match=compare_entry_field("problematic_regions", gt.get("problematic_regions"), actual_problematic_regions, task_id),
            )
        )

    elif task_id == "task4":
        for field in expected_fields:
            comparisons.append(
                Comparison(
                    field=field,
                    expected=gt.get(field),
                    actual=result.get(field),
                    match=compare_entry_field(field, gt.get(field), result.get(field), task_id),
                )
            )

    matched_fields = [item for item in comparisons if item.match]
    wrong_fields = [item.field for item in comparisons if not item.match]
    field_accuracy = len(matched_fields) / len(comparisons) if comparisons else 0.0
    partial_credit_accuracy = ""
    partial_credit_support = 0

    critical_fields = TASK_RULES[task_id]["critical_fields"]
    critical_wrong = [item.field for item in comparisons if (item.field in critical_fields or any(item.field.startswith(f"lanes") for f in critical_fields)) and not item.match]

    if task_id == "task3" and comparisons:
        partial_scores = [task3_partial_field_score(item.field, item.expected, item.actual) for item in comparisons]
        partial_credit_accuracy = sum(partial_scores) / len(partial_scores)
        partial_credit_support = sum(1 for score in partial_scores if score > 0.0)

    if not comparisons:
        entry_status = "incorrect"
    elif len(matched_fields) == len(comparisons):
        entry_status = "correct"
    elif len(critical_wrong) >= 2 or field_accuracy < 0.4 or len(matched_fields) == 0:
        entry_status = "incorrect"
    else:
        entry_status = "partially_correct"

    if wrong_fields:
        main_error_reason = wrong_fields[0]
    else:
        main_error_reason = "none"

    if entry_status == "missing_model_output":
        short_comment = "No model output JSON file found."
    elif entry_status == "correct":
        short_comment = "All evaluated fields match."
    elif entry_status == "partially_correct":
        short_comment = f"Some key fields match, but {main_error_reason} is off."
    else:
        short_comment = f"Most important fields are off, starting with {main_error_reason}."

    model_output_found = True
    if resolved_path is None:
        model_output_found = False

    return {
        "task_id": task_id,
        "image_id": image_id,
        "image_path": image_path,
        "model_output_found": model_output_found,
        "entry_status": entry_status,
        "field_accuracy": field_accuracy,
        "wrong_fields": "; ".join(wrong_fields),
        "main_error_reason": main_error_reason,
        "short_comment": short_comment,
        "field_count": len(comparisons),
        "matched_count": len(matched_fields),
        "partial_credit_accuracy": partial_credit_accuracy,
        "partial_credit_support": partial_credit_support,
        "sort_key": (
            0 if entry_status == "correct" else 1 if entry_status == "partially_correct" else 2 if entry_status == "incorrect" else 3,
            task_id,
            image_id,
        ),
    }


def summarize(entries: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    compared_fields_display = {
        "task1": "traffic_sign_visible, sign_type, condition, occlusion, damage, readability",
        "task2": "lane_count, visible_arrow_count, visible_arrow_type, allowed_movements, merge_behavior, special_regulation",
        "task3": "detection_quality, lane_markings_in_scene, detection_failure_visible, failure_reasons, problematic_regions",
        "task4": "tunnel_visible, bridge_visible, tall_buildings_visible, urban_canyon, gps_risk",
    }

    def split_wrong_fields(value: str) -> List[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(";") if item.strip()]

    def base_field_name(field_name: str) -> str:
        if field_name.startswith("lanes[") and "." in field_name:
            return field_name.split(".", 1)[1]
        return field_name

    def top_wrong_fields(rows: List[Dict[str, Any]], limit: int = 3) -> List[str]:
        counter: Counter[str] = Counter()
        for row in rows:
            seen_in_row = set()
            for field_name in split_wrong_fields(row["wrong_fields"]):
                base_name = base_field_name(field_name)
                if base_name in {"model_output_missing", "invalid_ground_truth_entry"}:
                    continue
                seen_in_row.add(base_name)
            for base_name in seen_in_row:
                counter[base_name] += 1
        return [name for name, _ in counter.most_common(limit)]

    def interpret_task_error(task_id: str, wrong_fields: List[str]) -> str:
        wrong_set = set(wrong_fields)
        if task_id == "task1":
            if "sign_type" in wrong_set:
                return "Mostly sign-type normalization or naming mismatch, not a pure perception miss."
            if "traffic_sign_visible" in wrong_set:
                return "Traffic-sign visibility or detection miss."
            return "Traffic-sign attribute interpretation issue."
        if task_id == "task2":
            if {"merge_behavior", "allowed_movements", "visible_arrow_type"} & wrong_set:
                return "Lane semantics are simplified; merge-left and left-turn behavior are often confused."
            if "lane_count" in wrong_set or "visible_arrow_count" in wrong_set:
                return "Lane structure is miscounted before higher-level semantics are inferred."
            return "Lane-level semantic reasoning is unstable."
        if task_id == "task3":
            return "Strict exact-match scoring underestimates partial correctness on difficult failure cases."
        if task_id == "task4":
            if "bridge_visible" in wrong_set or "gps_risk" in wrong_set:
                return "Infrastructure reasoning confusion, especially guardrail/bridge and GPS-risk interpretation."
            return "Infrastructure context is mostly stable with a few reasoning mistakes."
        return "Mixed error pattern."

    def interpret_case_reason(task_id: str, wrong_fields_text: str, status: str) -> str:
        if status == "missing_model_output":
            return "No stored model output was available for comparison."
        wrong_fields = [base_field_name(name) for name in split_wrong_fields(wrong_fields_text)]
        wrong_set = set(wrong_fields)
        if task_id == "task1":
            if "sign_type" in wrong_set:
                return "The sign was seen, but the label looks mismatched or too coarsely normalized."
            if "readability" in wrong_set:
                return "The sign readability judgement differs from the manual label."
            return "Traffic-sign attributes do not fully align."
        if task_id == "task2":
            if "merge_behavior" in wrong_set:
                return "The model mixes merge behavior with a normal turning interpretation."
            if "allowed_movements" in wrong_set or "visible_arrow_type" in wrong_set:
                return "Arrow meaning is simplified and lane-level movement semantics are lost."
            if "lane_count" in wrong_set or "visible_arrow_count" in wrong_set:
                return "The lane structure is counted differently from the manual interpretation."
            return "Lane semantics do not match the expected structure."
        if task_id == "task3":
            return "The scene contains some valid detection evidence, but the failure definition does not align exactly."
        if task_id == "task4":
            if "bridge_visible" in wrong_set:
                return "Roadside structure or guardrail context is interpreted as bridge-related infrastructure."
            if "gps_risk" in wrong_set:
                return "The GPS-risk infrastructure cue is interpreted differently from the manual label."
            return "Infrastructure cues are interpreted differently."
        return "Semantic mismatch."

    total_entries = len(entries)
    evaluated_entries = sum(1 for row in entries if row["model_output_found"])
    missing_model_outputs = sum(1 for row in entries if row["entry_status"] == "missing_model_output")
    scored_rows = [row for row in entries if row["model_output_found"] and row["field_count"] > 0]
    total_scored_fields = sum(row["field_count"] for row in scored_rows)
    total_matched_fields = sum(row["matched_count"] for row in scored_rows)
    overall_field_accuracy = (total_matched_fields / total_scored_fields) if total_scored_fields else 0.0
    fully_correct_entries = sum(1 for row in entries if row["entry_status"] == "correct")
    overall_entry_accuracy = fully_correct_entries / total_entries if total_entries else 0.0

    per_task: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "entries": 0,
        "field_count": 0,
        "matched_count": 0,
        "correct": 0,
        "partial": 0,
        "incorrect": 0,
        "rows": [],
        "partial_credit_sum": 0.0,
        "partial_credit_rows": 0,
        "partial_credit_nonzero": 0,
    })

    for row in entries:
        bucket = per_task[row["task_id"]]
        bucket["entries"] += 1
        bucket["rows"].append(row)
        if row["model_output_found"]:
            bucket["field_count"] += row["field_count"]
            bucket["matched_count"] += row["matched_count"]
            if row.get("partial_credit_accuracy", "") != "":
                bucket["partial_credit_sum"] += float(row["partial_credit_accuracy"])
                bucket["partial_credit_rows"] += 1
                if float(row["partial_credit_accuracy"]) > 0.0:
                    bucket["partial_credit_nonzero"] += 1
        if row["entry_status"] == "correct":
            bucket["correct"] += 1
        elif row["entry_status"] == "partially_correct":
            bucket["partial"] += 1
        else:
            bucket["incorrect"] += 1

    main_errors_all = Counter()
    for row in entries:
        seen_in_row = set()
        for field_name in split_wrong_fields(row["wrong_fields"]):
            base_name = base_field_name(field_name)
            if base_name in {"model_output_missing", "invalid_ground_truth_entry"}:
                continue
            seen_in_row.add(base_name)
        for base_name in seen_in_row:
            main_errors_all[base_name] += 1

    top_representative_cases = [
        row
        for row in sorted(entries, key=lambda r: (r["sort_key"], -r["field_count"]))
        if row["entry_status"] in {"partially_correct", "incorrect", "missing_model_output"}
    ][:5]

    def fmt_pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    summary_lines: List[str] = []
    summary_lines.append("# Evaluation Summary")
    summary_lines.append("")
    summary_lines.append("This report compares the manual ground truth against the stored model outputs.")
    summary_lines.append("Confidence and explanation are excluded from automatic scoring.")
    summary_lines.append("")
    summary_lines.append("## Evaluation Setup")
    summary_lines.append(f"- Total ground truth entries: {total_entries}")
    summary_lines.append(f"- Evaluated entries: {evaluated_entries}")
    summary_lines.append(f"- Missing model outputs: {missing_model_outputs}")
    summary_lines.append("- Compared fields:")
    for task_id in SUPPORTED_TASKS:
        summary_lines.append(f"  - {task_id}: {compared_fields_display[task_id]}")
    summary_lines.append("- Excluded fields: confidence, explanation")
    summary_lines.append("")
    summary_lines.append("## Quantitative Result")
    summary_lines.append(f"- Overall field-level accuracy: {fmt_pct(overall_field_accuracy)}")
    summary_lines.append(f"- Overall entry-level accuracy: {fmt_pct(overall_entry_accuracy)}")
    summary_lines.append("")
    summary_lines.append("| task_id | entries | field-level accuracy | entry-level accuracy | main wrong fields | interpreted main error category |")
    summary_lines.append("| --- | ---: | ---: | ---: | --- | --- |")
    for task_id in SUPPORTED_TASKS:
        bucket = per_task.get(task_id, {
            "entries": 0,
            "field_count": 0,
            "matched_count": 0,
            "correct": 0,
            "partial": 0,
            "incorrect": 0,
            "rows": [],
        })
        field_accuracy = (bucket["matched_count"] / bucket["field_count"]) if bucket["field_count"] else 0.0
        entry_accuracy = (bucket["correct"] / bucket["entries"]) if bucket["entries"] else 0.0
        wrong_fields = top_wrong_fields(bucket["rows"], limit=3)
        main_wrong_fields = ", ".join(wrong_fields) if wrong_fields else "none"
        interpreted_error = interpret_task_error(task_id, wrong_fields)
        summary_lines.append(
            f"| {task_id} | {bucket['entries']} | {fmt_pct(field_accuracy)} | {fmt_pct(entry_accuracy)} | {main_wrong_fields} | {interpreted_error} |"
        )
    summary_lines.append("")
    summary_lines.append("## Task-wise Error Interpretation")
    summary_lines.append("- Task1: The dominant issue is sign-type normalization rather than basic sign visibility. Labels such as `speed_limit_30` versus a broader `speed limit sign` style category should be treated as schema or naming mismatches first, and only secondarily as perception errors.")
    summary_lines.append("- Task2: The main weakness is lane-level semantic reasoning. Errors are concentrated in `lane_count`, `visible_arrow_type`, `allowed_movements`, and `merge_behavior`, which suggests confusion between `merge_left` and a normal left-turn interpretation, plus simplification of complex multi-lane semantics.")
    summary_lines.append("- Task3: The errors look less like simple misses and more like definition problems. Near intersections, weak or discontinuous lane markings, incomplete predictions, and ambiguous road topology make it hard to decide whether the image shows a real detector failure or just a difficult scene.")
    task3_bucket = per_task.get("task3", {
        "entries": 0,
        "field_count": 0,
        "matched_count": 0,
        "correct": 0,
        "partial": 0,
        "incorrect": 0,
        "rows": [],
        "partial_credit_sum": 0.0,
        "partial_credit_rows": 0,
        "partial_credit_nonzero": 0,
    })
    task3_strict_accuracy = (task3_bucket["matched_count"] / task3_bucket["field_count"]) if task3_bucket["field_count"] else 0.0
    task3_partial_credit_accuracy = (task3_bucket["partial_credit_sum"] / task3_bucket["partial_credit_rows"]) if task3_bucket["partial_credit_rows"] else 0.0
    summary_lines.append("")
    summary_lines.append("### Task3 scoring limitation")
    summary_lines.append("Task3 currently receives a very low automatic score, but this should be interpreted carefully. The selected Task3 samples were intentionally difficult cases near intersections and traffic-light-controlled junctions, where lane markings are often weak, discontinuous, or ambiguous. In addition, the current evaluation uses strict field matching, so partially correct detections can still be counted as wrong if the exact labels do not match. Therefore, the Task3 result mainly indicates that the failure definition and scoring method need refinement, rather than proving that all lane-detection validation outputs are completely wrong.")
    summary_lines.append(f"- Task3 strict field-level accuracy: {fmt_pct(task3_strict_accuracy)}")
    summary_lines.append(f"- Task3 partial-credit field-level accuracy: {fmt_pct(task3_partial_credit_accuracy)}")
    summary_lines.append(f"- Task3 entries with some partial credit: {task3_bucket['partial_credit_nonzero']}/{task3_bucket['entries']}")
    summary_lines.append("- Interpretation: the current Task3 sample is deliberately biased toward difficult intersection and failure cases, so this number should be read as difficult-case performance rather than general Task3 performance.")
    summary_lines.append("")
    summary_lines.append("- Task4: The task is mostly stable, but the remaining errors are infrastructure reasoning issues. The main pattern is guardrail or roadside structure being interpreted as bridge-related context, together with some ambiguity in how GPS-risk infrastructure should be judged.")
    summary_lines.append("")
    summary_lines.append("## Representative Cases")
    summary_lines.append("")
    summary_lines.append("| task_id | image_id | status | wrong_fields | interpreted_reason | short_comment |")
    summary_lines.append("| --- | --- | --- | --- | --- | --- |")
    if top_representative_cases:
        for row in top_representative_cases:
            summary_lines.append(
                f"| {row['task_id']} | {row['image_id']} | {row['entry_status']} | {row['wrong_fields'] or 'none'} | {interpret_case_reason(row['task_id'], row['wrong_fields'], row['entry_status'])} | {row['short_comment']} |"
            )
    else:
        summary_lines.append("| none | none | none | none | none | none |")
    summary_lines.append("")
    summary_lines.append("## Main Findings")
    summary_lines.append("- Task1 is relatively stable for clear traffic signs.")
    summary_lines.append("- Task2 is weak for lane-level semantic reasoning.")
    summary_lines.append("- Task3 shows low automatic accuracy mainly because the selected cases are difficult intersection/failure cases and because strict exact-match scoring does not capture partial correctness.")
    summary_lines.append("- Task4 is mostly usable but can confuse guardrails or roadside structures with bridges.")
    summary_lines.append("- Overall, local lightweight VLMs are better for simple structured perception tasks than for high-level traffic semantics.")
    if main_errors_all:
        summary_lines.append(f"- Across all tasks, the most frequent raw wrong fields are: {', '.join(name for name, _ in main_errors_all.most_common(4))}.")

    return "\n".join(summary_lines), entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a concise evaluation report for meeting use.")
    parser.add_argument(
        "--ground-truth",
        default=str(DEFAULT_GROUND_TRUTH),
        help="Path to the manual ground truth JSON file",
    )
    args = parser.parse_args()

    ground_truth_path = Path(args.ground_truth)
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    ground_truth_records = load_ground_truth(ground_truth_path)
    entry_rows = [compare_record(record) for record in ground_truth_records]

    summary_md, detail_rows = summarize(entry_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_text(SUMMARY_PATH, summary_md)
    write_csv(
        DETAILS_PATH,
        [
            {
                "task_id": row["task_id"],
                "image_id": row["image_id"],
                "model_output_found": "True" if row["model_output_found"] else "False",
                "entry_status": row["entry_status"],
                "field_accuracy": "" if row["field_accuracy"] == "" else f"{row['field_accuracy']:.3f}",
                "wrong_fields": row["wrong_fields"],
                "main_error_reason": row["main_error_reason"],
                "short_comment": row["short_comment"],
            }
            for row in detail_rows
        ],
    )

    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {DETAILS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
