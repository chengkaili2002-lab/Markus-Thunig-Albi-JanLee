# Ground Truth Annotation Guide

This guide follows the JSON output schemas defined in the task prompts.

## Scope

- Annotate `task1` to `task4` only.
- Do not annotate `task5` yet.
- Use one manual record per image per task.
- If the same image appears in multiple tasks, annotate it separately for each task with that task's schema.

## How To Organize Ground Truth

- Use the task-specific template that matches the prompt output schema.
- Keep the manual annotations as close as possible to the model JSON fields.
- Do not add extra reasoning fields or long explanations beyond the prompt schema.
- If you need bookkeeping for your own workflow, keep it outside the compared JSON object.

## Task 1 Schema

Source prompt:

- `prompts/task1_sign_condition.txt`

Expected JSON fields:

- `traffic_sign_visible`
- `sign_type`
- `condition`
- `occlusion`
- `damage`
- `readability`
- `confidence`
- `explanation`

Field notes:

- `traffic_sign_visible`: boolean
- `sign_type`: short string; use `none` if no sign is visible. If multiple signs are visible, keep one string with the most relevant sign first and separate additional signs with `;`
- `condition`: one of `clean`, `dirty`, `sticker`, `vandalized`, `invalidated`, `unclear`
- `occlusion`: one of `none`, `partial`, `heavy`, `unclear`
- `damage`: one of `none`, `minor`, `severe`, `unclear`
- `readability`: one of `readable`, `partly_readable`, `unreadable`, `unclear`
- `confidence`: number between `0.0` and `1.0`
- `explanation`: short string

Use:

- `data/ground_truth/task1_ground_truth_template.json`

## Task 2 Schema

Source prompt:

- `prompts/task2_lane_semantics.txt`

Expected JSON fields:

- `arrow_sign_visible`
- `visible_arrow_count`
- `lane_count`
- `lanes`
- `has_conflicting_or_opposite_movements`
- `additional_lane_regulations`
- `overall_lane_semantics`
- `confidence`
- `explanation`

Lane object fields inside `lanes`:

- `lane_index`
- `visible_arrow_type`
- `allowed_movements`
- `merge_behavior`
- `special_regulation`
- `semantic_interpretation`
- `lane_confidence`

Field notes:

- `arrow_sign_visible`: boolean
- `visible_arrow_count`: integer count of visible arrow symbols
- `lane_count`: integer count of lane entries in `lanes`
- `lanes`: array of lane objects
- `has_conflicting_or_opposite_movements`: boolean
- `additional_lane_regulations`: array of short strings
- `overall_lane_semantics`: short string
- `confidence`: number between `0.0` and `1.0`
- `explanation`: short string

Prompt-aligned value hints for task 2:

- `visible_arrow_type`: one of `straight`, `left`, `right`, `straight+left`, `straight+right`, `left+right`, `straight+left+right`, `u_turn`, `merge_left`, `merge_right`, `unclear`
- `allowed_movements`: list of movements that match the lane meaning, using short values such as `straight`, `left`, `right`, `merge_left`, `merge_right`, or `unclear`
- `merge_behavior`: one of `none`, `merge_left`, `merge_right`, `unclear`
- `special_regulation`: one of `none`, `bus_only`, `truck_only`, `truck_restriction`, `time_restriction`, `turn_restriction`, `unclear`

Use:

- `data/ground_truth/task2_ground_truth_template.json`

## Task 3 Schema

Source prompt:

- `prompts/task3_lane_failure.txt`

Expected JSON fields:

- `detection_quality`
- `lane_markings_in_scene`
- `detection_failure_visible`
- `failure_reasons`
- `problematic_regions`
- `confidence`
- `explanation`

Field notes:

- `detection_quality`: one of `correct`, `partially_correct`, `incorrect`, `unclear`
- `lane_markings_in_scene`: one of `continuous`, `discontinuous`, `unclear`
- `detection_failure_visible`: boolean
- `failure_reasons`: array of short strings
- `problematic_regions`: array of short strings
- `confidence`: number between `0.0` and `1.0`
- `explanation`: short string

Use:

- `data/ground_truth/task3_ground_truth_template.json`

## Task 4 Schema

Source prompt:

- `prompts/task4_infrastructure.txt`

Expected JSON fields:

- `tunnel_visible`
- `bridge_visible`
- `tall_buildings_visible`
- `urban_canyon`
- `gps_risk`
- `explanation`

Field notes:

- `tunnel_visible`: boolean
- `bridge_visible`: boolean
- `tall_buildings_visible`: boolean
- `urban_canyon`: boolean
- `gps_risk`: one of `low`, `medium`, `high`, `unclear`
- `explanation`: short string

Use:

- `data/ground_truth/task4_ground_truth_template.json`

## Suggested Small Batch Workflow

1. Pick 3 previously tested images for `task1`.
2. Pick 3 previously tested images for `task2`.
3. Pick 3 previously tested images for `task3`.
4. Pick 3 previously tested images for `task4`.
5. Fill the corresponding task-specific template for each selected image.

## Notes

- Keep the annotation values short and consistent.
- Do not annotate `task5` until it is explicitly brought back into scope.
- Do not change the prompts or the model outputs when preparing ground truth.
