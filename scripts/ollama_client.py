import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import ollama
from pydantic import BaseModel


def extract_json_text(text: str) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]

    return cleaned


def parse_json_response(text: str) -> Dict[str, Any]:
    json_text = extract_json_text(text)
    return json.loads(json_text)


def call_ollama_with_image(
    *,
    model: str,
    prompt_text: str,
    image_path: Path,
    schema_model: Type[BaseModel],
) -> Tuple[Dict[str, Any], str]:
    """
    Send one image and one fixed prompt to a local Ollama model.

    The response is requested as structured JSON, then validated with Pydantic.
    """

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt_text},
            {
                "role": "user",
                "content": "Analyze the attached traffic-scene image and return JSON only.",
                "images": [str(image_path)],
            },
        ],
        format=schema_model.model_json_schema(),
        options={"temperature": 0},
        stream=False,
    )

    raw_text = response.message.content
    parsed = schema_model.model_validate_json(extract_json_text(raw_text))
    return parsed.model_dump(), raw_text


def call_ollama_with_images(
    *,
    model: str,
    prompt_text: str,
    image_paths: List[Path],
    schema_model: Type[BaseModel],
) -> Tuple[Dict[str, Any], str]:
    """
    Send multiple images and one fixed prompt to a local Ollama model.

    Task 1 uses Image A as the clean reference and Image B as the test patch.
    """

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt_text},
            {
                "role": "user",
                "content": "Analyze the attached pair of traffic-sign images and return JSON only.",
                "images": [str(image_path) for image_path in image_paths],
            },
        ],
        format=schema_model.model_json_schema(),
        options={"temperature": 0},
        stream=False,
    )

    raw_text = response.message.content
    parsed = schema_model.model_validate_json(extract_json_text(raw_text))
    return parsed.model_dump(), raw_text
