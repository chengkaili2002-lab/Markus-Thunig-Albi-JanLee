# Experiment Log

Use this file to track quick notes while testing the local VLM pipeline.

## Template

- Date:
- Task:
- Model:
- Image set:
- Prompt version:
- Notes:
- Result summary:
- Next step:

## Run 002 - Task 2 Error Case Analysis

- Date: 2026-05-12
- Task: Task 2 - Lane Semantics from Arrow Signs
- Model: gemma4:e2b-it-q4_K_M
- Image set: Additional_Road_Course sample images
- Prompt version: prompts/task2_lane_semantics.txt
- Test image: kism-samsung-ennoo-probe-03-20241126-1207_1732621868678_0.png

- Notes:
  The local VLM was tested on lane-arrow traffic signs. The model was able to recognize that arrows are present, but it did not reliably infer lane-level traffic semantics. In one case, the model treated the arrow sign as an abstract path diagram instead of a traffic lane direction sign.

- Observed model issue:
  The model stated that it could not determine whether the leftmost arrow means straight, left turn, right turn, or lane merge. This is problematic because the arrow shapes on lane-direction signs already encode basic traffic semantics.

- Error type:
  Over-uncertainty / semantic misunderstanding.
  The model over-emphasized missing external context and failed to interpret visible traffic-sign semantics from the sign itself.

- Expected interpretation:
  The sign should be interpreted as a lane-direction sign. Vertical arrows indicate straight movement, curved arrows indicate turning or lane merging, and combined arrows indicate multiple allowed movements.

- Result summary:
  The current Task 2 prompt is not strong enough to force lane-level semantic interpretation. The output is too close to general image description or abstract reasoning.

- Next step:
  Improve the Task 2 prompt so that the model treats the image as a lane-direction traffic sign, infers lane-level movements, recognizes combined arrows and merge arrows, and returns structured JSON suitable for benchmark evaluation.






  
  ## Error Case - Task 2: Missed Side Arrows and Overconfident Simplification

- Date: 2026-05-12
- Task: Task 2 - Lane Semantics from Arrow Signs
- Model: gemma4:e2b-it-q4_K_M
- Image: lane-arrow sign with left-turn lane, two straight lanes, and right-side complex arrow / merge indication

### Expected interpretation

The sign should be interpreted as a multi-lane direction sign.

Expected lane semantics:
- Leftmost lane: left turn
- Middle lanes: straight
- Rightmost lane: straight/right and possible merge or lane-shift semantics

### Model output problem

The model returned:
- `lane_count: 3`
- all lanes as `straight`
- `confidence: 1.0`

### Observed errors

1. Wrong lane count  
   The sign likely contains 4 lane groups, but the model detected only 3.

2. Missed left-turn arrow  
   The leftmost curved arrow was incorrectly ignored or interpreted as straight.

3. Missed right-side complex arrow  
   The rightmost arrow contains straight/right and possible merge or lane-shift semantics, but the model simplified it to straight.

4. Overconfident output  
   The model returned `confidence: 1.0`, although important visual and semantic information was missed.

### Error type

- missed side arrows
- overconfident simplification
- weak lane topology understanding
- insufficient recognition of turn/merge semantics

### Research implication

This example shows that the local VLM can recognize simple straight arrows but struggles with complex lane-direction signs, especially side arrows and lane topology. This is important for Task 2 because the goal is not just arrow detection, but lane-level semantic understanding.

### Next step

Improve the Task 2 prompt and schema to explicitly focus on:
- side arrows
- combined arrows
- lane merge / lane shift semantics
- avoiding overconfident outputs when parts of the sign are unclear






## Error Case - Task 2: False Arrow-Sign Detection on Speed Limit Scene

- Date: 2026-05-12
- Task: Task 2 - Lane Semantics from Arrow Signs
- Model: gemma4:e2b-it-q4_K_M
- Image: traffic scene with a speed limit 30 sign, no arrow sign visible

### Expected interpretation

This image does not contain a lane-arrow sign.

Expected Task 2 output:
- `arrow_sign_visible`: false
- `lane_count`: 0 or not applicable
- `lanes`: []
- `additional_lane_regulations`: []
- `overall_lane_semantics`: no arrow-sign-based lane semantics available

The visible traffic sign is a speed limit sign, not an arrow sign.

### Model output problem

The model returned:
- `arrow_sign_visible: true`
- `lane_count: 1`
- `visible_arrow_type: straight`
- `additional_lane_regulations: ["Speed limit of 30"]`
- `confidence: 0.95`

### Observed errors

1. False positive arrow-sign detection  
   The model claimed that an arrow sign is visible, although the image only contains a speed limit sign.

2. Wrong task interpretation  
   The model mixed Task 2 lane-arrow semantics with general traffic-sign interpretation.

3. Wrong lane semantics  
   The model inferred a straight lane movement even though no lane-direction arrow sign is present.

4. Wrong lane count  
   The road scene contains at least two traffic directions / lanes: one ego-direction lane and one oncoming lane. The model simplified this into one lane.

5. Overconfident output  
   The model returned `confidence: 0.95` despite a clear task mismatch.

### Error type

- false positive arrow sign
- task confusion
- speed-limit sign misinterpreted as lane regulation
- wrong lane count
- overconfident hallucination

### Research implication

This error shows that the Task 2 prompt needs a stronger negative-case rule. The model must first decide whether a lane-arrow sign is visible. If no arrow sign is visible, it should stop and return an empty lane-semantics result instead of forcing lane interpretation.

### Prompt improvement suggestion

Add a rule to the Task 2 prompt:

If no lane-arrow sign is visible, return:
- `arrow_sign_visible: false`
- `lane_count: 0`
- `lanes: []`
- `additional_lane_regulations: []`
- `overall_lane_semantics: "No lane-arrow sign visible; lane semantics not applicable."`

Do not infer lane semantics from speed limit signs, road shape, vehicles, or general road context.





## Error Case - Task 4: Guardrail Misclassified as Bridge

- Task: Task 4 - Infrastructure Recognition
- Model: gemma4:e2b-it-q4_K_M
- Image: open road with trees, guardrail, and speed limit sign

### Expected interpretation
No bridge is clearly visible. The scene shows an open road with roadside vegetation and a guardrail.

### Model output problem
The model returned `bridge_visible: true`, probably because it associated the guardrail with a bridge or elevated structure.

### Error type
- false positive bridge detection
- infrastructure hallucination
- weak distinction between guardrail and bridge structure

### Prompt improvement
Add rule: Do not classify a scene as bridge_visible only because a guardrail is present. A bridge should require visible bridge-specific structures such as elevated roadway, bridge deck, underpass, river crossing, or large supporting structure.