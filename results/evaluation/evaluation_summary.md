# Evaluation Summary

This report compares the manual ground truth against the stored model outputs.
Confidence and explanation are excluded from automatic scoring.

## Evaluation Setup
- Total ground truth entries: 12
- Evaluated entries: 11
- Missing model outputs: 1
- Compared fields:
  - task1: traffic_sign_visible, sign_type, condition, occlusion, damage, readability
  - task2: lane_count, visible_arrow_count, visible_arrow_type, allowed_movements, merge_behavior, special_regulation
  - task3: detection_quality, lane_markings_in_scene, detection_failure_visible, failure_reasons, problematic_regions
  - task4: tunnel_visible, bridge_visible, tall_buildings_visible, urban_canyon, gps_risk
- Excluded fields: confidence, explanation

## Quantitative Result
- Overall field-level accuracy: 48.5%
- Overall entry-level accuracy: 8.3%

| task_id | entries | field-level accuracy | entry-level accuracy | main wrong fields | interpreted main error category |
| --- | ---: | ---: | ---: | --- | --- |
| task1 | 3 | 77.8% | 0.0% | sign_type, readability | Mostly sign-type normalization or naming mismatch, not a pure perception miss. |
| task2 | 3 | 44.8% | 0.0% | merge_behavior, special_regulation, allowed_movements | Lane semantics are simplified; merge-left and left-turn behavior are often confused. |
| task3 | 3 | 0.0% | 0.0% | failure_reasons, detection_quality, detection_failure_visible | Strict exact-match scoring underestimates partial correctness on difficult failure cases. |
| task4 | 3 | 90.0% | 33.3% | bridge_visible | Infrastructure reasoning confusion, especially guardrail/bridge and GPS-risk interpretation. |

## Task-wise Error Interpretation
- Task1: The dominant issue is sign-type normalization rather than basic sign visibility. Labels such as `speed_limit_30` versus a broader `speed limit sign` style category should be treated as schema or naming mismatches first, and only secondarily as perception errors.
- Task2: The main weakness is lane-level semantic reasoning. Errors are concentrated in `lane_count`, `visible_arrow_type`, `allowed_movements`, and `merge_behavior`, which suggests confusion between `merge_left` and a normal left-turn interpretation, plus simplification of complex multi-lane semantics.
- Task3: The errors look less like simple misses and more like definition problems. Near intersections, weak or discontinuous lane markings, incomplete predictions, and ambiguous road topology make it hard to decide whether the image shows a real detector failure or just a difficult scene.

### Task3 scoring limitation
Task3 currently receives a very low automatic score, but this should be interpreted carefully. The selected Task3 samples were intentionally difficult cases near intersections and traffic-light-controlled junctions, where lane markings are often weak, discontinuous, or ambiguous. In addition, the current evaluation uses strict field matching, so partially correct detections can still be counted as wrong if the exact labels do not match. Therefore, the Task3 result mainly indicates that the failure definition and scoring method need refinement, rather than proving that all lane-detection validation outputs are completely wrong.
- Task3 strict field-level accuracy: 0.0%
- Task3 partial-credit field-level accuracy: 20.0%
- Task3 entries with some partial credit: 3/3
- Interpretation: the current Task3 sample is deliberately biased toward difficult intersection and failure cases, so this number should be read as difficult-case performance rather than general Task3 performance.

- Task4: The task is mostly stable, but the remaining errors are infrastructure reasoning issues. The main pattern is guardrail or roadside structure being interpreted as bridge-related context, together with some ambiguity in how GPS-risk infrastructure should be judged.

## Representative Cases

| task_id | image_id | status | wrong_fields | interpreted_reason | short_comment |
| --- | --- | --- | --- | --- | --- |
| task1 | 2018-05-30_08-12-16_911.jpg | partially_correct | sign_type | The sign was seen, but the label looks mismatched or too coarsely normalized. | Some key fields match, but sign_type is off. |
| task1 | 2018-05-30_08-21-27_017.jpg | partially_correct | sign_type | The sign was seen, but the label looks mismatched or too coarsely normalized. | Some key fields match, but sign_type is off. |
| task4 | 2018-05-30_08-12-16_911.jpg | partially_correct | bridge_visible | Roadside structure or guardrail context is interpreted as bridge-related infrastructure. | Some key fields match, but bridge_visible is off. |
| task1 | yaksamsungs23-20230703-0821__03_07_2023__09_11_19.png | incorrect | sign_type; readability | The sign was seen, but the label looks mismatched or too coarsely normalized. | Most important fields are off, starting with sign_type. |
| task2 | kism-samsung-ennoo-probe-03-20241126-1207_1732621868678_0.png | incorrect | lane_count; visible_arrow_count; lanes[1].visible_arrow_type; lanes[1].allowed_movements; lanes[2].visible_arrow_type; lanes[2].allowed_movements; lanes[2].merge_behavior; lanes[2].special_regulation | The model mixes merge behavior with a normal turning interpretation. | Most important fields are off, starting with lane_count. |

## Main Findings
- Task1 is relatively stable for clear traffic signs.
- Task2 is weak for lane-level semantic reasoning.
- Task3 shows low automatic accuracy mainly because the selected cases are difficult intersection/failure cases and because strict exact-match scoring does not capture partial correctness.
- Task4 is mostly usable but can confuse guardrails or roadside structures with bridges.
- Overall, local lightweight VLMs are better for simple structured perception tasks than for high-level traffic semantics.
- Across all tasks, the most frequent raw wrong fields are: sign_type, merge_behavior, special_regulation, allowed_movements.