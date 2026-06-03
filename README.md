# DCAITI Local VLM Traffic Scene Validation

This project is a simple Python baseline for running a local vision-language model on traffic-scene images with Ollama.

## What it does

- Reads images from `data/sample_images/`
- Loads a fixed prompt from `prompts/`
- Sends the image and prompt to a local Ollama model
- Parses a structured JSON response
- Saves one JSON file per image
- Writes one CSV summary per task

## Setup

1. Install Ollama and make sure it is running locally.
2. Pull a vision-capable model, for example:
   ```powershell
   ollama pull gemma4:e2b-it-q4_K_M
   ```
3. Install the Python dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Change into the project folder:
   ```powershell
   cd DCAITI
   ```

## Add images

Put traffic-scene images into:

```text
DCAITI/data/sample_images/
```

You can place images directly in that folder or inside subfolders. The scripts scan the whole tree recursively.

Supported common formats:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.webp`

## Run a task

Each task script loops over every image in the selected folder, calls Ollama, and stores the outputs in a run-specific folder under `results/<task>/runs/<timestamp>/`.

### Task 1: Traffic Sign Condition Analysis

```powershell
python scripts/run_task1.py --model gemma4:e2b-it-q4_K_M
```

By default, Task 1 now reads the five sign folders under `data/data/Task1_sign_impurity/` and uses each folder's `31.5.2026/` images as the test set, with the matching `standard.png` as the reference image.

### Task 2: Lane Semantics from Arrow Signs

```powershell
python scripts/run_task2.py --model gemma4:e2b-it-q4_K_M
```

### Task 3: Validation of Lane Detection Failure

```powershell
python scripts/run_task3.py --model gemma4:e2b-it-q4_K_M
```

Task 3 uses its own dedicated input folder under `data/` so you can keep a separate dataset for lane-detection-failure tests.

### Task 4: Infrastructure Recognition

```powershell
python scripts/run_task4.py --model gemma4:e2b-it-q4_K_M
```

### Task 5: Open-Ended Router

```powershell
python scripts/run_task5.py --model gemma4:e2b-it-q4_K_M
```

Task 5 is a lightweight pre-step. It checks which project-relevant triggers are visible and recommends whether Task 1, 2, 3, or 4 should run next.

## Model configuration

The model name is configurable with `--model`.

If you prefer, you can also set an environment variable:

```powershell
$env:OLLAMA_MODEL="gemma4:e2b-it-q4_K_M"
```

If no model is passed on the command line, the scripts use `OLLAMA_MODEL`, and then fall back to `gemma3n:e2b`.

## Output files

Task 1, Task 2, Task 3, Task 4, and Task 5 create a new timestamped run folder each time:

- `results/task1/runs/<timestamp>/`
- `results/task2/runs/<timestamp>/`
- `results/task3/runs/<timestamp>/`
- `results/task4/runs/<timestamp>/`
- `results/task5/runs/<timestamp>/`

Each run folder contains:

- `json/` with one JSON file per image
- a CSV summary for that run

Each JSON file includes:

- the source image name
- the source image relative path
- the source image absolute path
- the model name
- the runtime in seconds
- the structured model answer
- the raw response text

## Notes

- The pipeline is intentionally simple and beginner-friendly.
- The prompts are fixed and task-specific.
- Task 1 reports all visible traffic signs, not just one, when multiple signs appear.
- Task 5 is intended as a routing step to save compute before running more expensive downstream prompts.
- This is a baseline framework, not a full evaluation system yet.
