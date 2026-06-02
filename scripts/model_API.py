from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Literal
)

import ollama
from pydantic import BaseModel, Field

class Task1Result(BaseModel):
    traffic_sign_visible: bool = Field(description="Whether a traffic sign is visible")
    sign_type: str = Field(description="Type of the sign")
    condition: Literal["clean", "dirty", "sticker", "vandalized", "invalidated", "unclear"]
    occlusion: Literal["none", "partial", "heavy", "unclear"]
    damage: Literal["none", "minor", "severe", "unclear"]
    readability: Literal["readable", "partly_readable", "unreadable", "unclear"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence between 0.0 and 1.0")
    explanation: str = Field(description="Short explanation")


### TASK 2
# OLD
class Task2Lane(BaseModel):
    lane_index: int = Field(ge=1, description="Lane number from left to right or top to bottom")
    visible_arrow_type: Literal[
        "straight",
        "left",
        "right",
        "straight+left",
        "straight+right",
        "merge_left",
        "merge_right",
        "unclear",
    ]
    allowed_movements: List[Literal["straight", "left", "right", "merge_left", "merge_right", "unclear"]]
    special_regulation: Literal["none", "bus_only", "truck_restriction", "turn_restriction", "unclear"]
    semantic_interpretation: str = Field(description="Short lane-level interpretation")


class Task2Result(BaseModel):
    arrow_sign_visible: bool = Field(description="Whether any relevant arrow sign is visible")
    lane_count: int = Field(ge=0, description="Number of lanes described in the output")
    lanes: List[Task2Lane] = Field(description="Lane-by-lane semantic interpretation")
    opposing_directions: bool = Field(description="Whether the sign contains opposing directions")
    additional_lane_regulations: List[str] = Field(description="Extra lane regulations as short phrases")
    overall_lane_semantics: str = Field(description="Short overall lane meaning")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence between 0.0 and 1.0")
    explanation: str = Field(description="Short explanation")

# NEW
Direction = Literal[
    "left",
    "slight_left",
    "straight",
    "slight_right",
    "right",
    "uturn",
    "unknown",
]

class Task2Lane(BaseModel):
    lane_id: str = Field(description="Stable lane ID, e.g. lane_1")
    position_index: int = Field(description="1-based lane order from left to right")
    directions: list[Direction] = Field(description="Allowed driving directions")
    restriction_present: bool = Field(description="Whether any extra restriction symbol is present")
    merge_into: list[str] = Field(
        default_factory=list,
        description="Lane IDs this lane visually connects or merges into",
    )
    explanation: str = Field(description="Short explanation text describing the lane arrow belogning to this lane")


class Task2LaneSignAnalysis(BaseModel):
    lane_count: int = Field(description="Number of detected lane arrows")
    lanes: list[Task2Lane]
    needs_review: bool = Field(description="True if image is ambiguous or unclear")
    review_reason: str | None = Field(default=None)


### TASK3
class Task3Result(BaseModel):
    lane_detection_assessment: Literal["correct", "incorrect", "unclear"]
    lane_marking_continuity: Literal["continuous", "discontinuous", "unclear"]
    likely_reason: Literal[
        "road_topology",
        "temporal_occlusion",
        "camera_interference",
        "poor_markings",
        "intersection_geometry",
        "unclear",
    ]
    supporting_signals: List[str] = Field(description="Short evidence phrases")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence between 0.0 and 1.0")
    explanation: str = Field(description="Short explanation")


class Task4Result(BaseModel):
    tunnel_visible: bool = Field(description="Whether a tunnel is visible")
    bridge_visible: bool = Field(description="Whether a bridge is visible")
    tall_buildings_visible: bool = Field(description="Whether tall buildings are visible")
    urban_canyon: bool = Field(description="Whether the scene looks like an urban canyon")
    gps_risk: Literal["low", "medium", "high", "unclear"]
    explanation: str = Field(description="Short explanation")


class Task5StandardInformation(BaseModel):
    road_type: str = Field(description="Short road type description")
    visible_traffic_signs: List[str] = Field(description="Visible traffic signs as short phrases")
    visible_road_users: List[str] = Field(description="Visible road users as short phrases")
    visible_infrastructure: List[str] = Field(description="Visible infrastructure as short phrases")
    image_quality_issues: List[str] = Field(description="Image quality issues as short phrases")


class Task5Result(BaseModel):
    scene_category: Literal[
        "urban_road",
        "highway",
        "intersection",
        "tunnel_or_underpass",
        "parking_or_private_area",
        "unclear",
    ]
    traffic_sign_visible: bool
    traffic_sign_condition_relevant: bool
    arrow_or_lane_direction_sign_visible: bool
    lane_detection_overlay_visible: bool
    gps_risk_infrastructure_visible: bool
    standard_information: Task5StandardInformation
    recommended_tasks: List[
        Literal[
            "task1_sign_condition",
            "task2_lane_semantics",
            "task3_lane_failure_validation",
            "task4_infrastructure",
            "skip",
        ]
    ]
    priority: Literal["low", "medium", "high", "unclear"]
    resource_strategy: Literal[
        "skip",
        "run_single_task",
        "run_multiple_tasks",
        "send_to_smartphone_validation",
        "unclear",
    ]
    trigger_reason: str = Field(description="Short reason for the recommended routing decision")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence between 0.0 and 1.0")

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
        keep_alive="15m",
        stream=False,
    )

    raw_text = response.message.content
    parsed = schema_model.model_validate_json(extract_json_text(raw_text))
    return parsed.model_dump(), raw_text
