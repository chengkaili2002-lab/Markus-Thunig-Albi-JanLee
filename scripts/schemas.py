from typing import List
from typing import Literal

from pydantic import BaseModel, Field


class Task1DirtyArea(BaseModel):
    location: Literal[
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
        "center",
        "border",
        "arrow",
        "unclear",
    ]
    type: Literal[
        "sticker",
        "paint_loss",
        "discoloration",
        "burn_or_blackening",
        "graffiti",
        "contamination",
        "damage",
    ]


class Task1Result(BaseModel):
    sign_type: str = Field(description="Type of the sign")
    surface_condition: Literal["clean", "not_clean", "uncertain"]
    evidence: str = Field(description="Short visual evidence")
    dirty_count: int = Field(ge=0, description="Number of visible anomalies")
    overall_readability: Literal["not_affected", "partially_affected", "severely_affected", "uncertain"]
    dirty_areas: List[Task1DirtyArea] = Field(default_factory=list, description="Visible anomaly list")


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
