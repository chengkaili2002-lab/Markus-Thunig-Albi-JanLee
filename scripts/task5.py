"""Task 5 — entry point."""
from pathlib import Path
from task5_run import run

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "task5.txt"

# ---------- test parameters ----------
MAX_IMAGES = 10
# -------------------------------------

prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()
run(prompt_text=prompt, max_images=MAX_IMAGES)
