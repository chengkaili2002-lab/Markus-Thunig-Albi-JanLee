from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import ollama
from pydantic import BaseModel


def extract_json_text(text: str) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]

    return cleaned


def call_ollama_with_images(
    *,
    model: str,
    prompt_text: str,
    image_paths: List[Path],
    schema_model: Type[BaseModel],
) -> Tuple[Dict[str, Any], str]:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt_text},
            {
                "role": "user",
                "content": "Analyze Image A and Image B. Return JSON only.",
                "images": [str(image_path.resolve()) for image_path in image_paths],
            },
        ],
        format=schema_model.model_json_schema(),
        options={"temperature": 0},
        stream=False,
    )

    raw_text = response.message.content
    parsed = schema_model.model_validate_json(extract_json_text(raw_text))
    return parsed.model_dump(), raw_text


def call_ollama_with_image(
    *,
    model: str,
    prompt_text: str,
    image_path: Path,
    schema_model: Type[BaseModel],
) -> Tuple[Dict[str, Any], str]:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt_text},
            {
                "role": "user",
                "content": "Analyze the attached traffic-sign image. Return JSON only.",
                "images": [str(image_path.resolve())],
            },
        ],
        format=schema_model.model_json_schema(),
        options={"temperature": 0},
        stream=False,
    )

    raw_text = response.message.content
    parsed = schema_model.model_validate_json(extract_json_text(raw_text))
    return parsed.model_dump(), raw_text
