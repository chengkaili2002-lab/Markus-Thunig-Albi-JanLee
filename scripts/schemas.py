from typing import List, Literal

from pydantic import BaseModel, Field


class Task1Result(BaseModel):
    image_name: str = Field(description="Name of the analyzed test image")
    model: str = Field(description="Model name")
    runtime_seconds: float = Field(ge=0.0, description="Model runtime in seconds")
    sign_type: str = Field(description="Type of the sign")
    overall_readability: Literal[
        "not_affected",
        "partially_affected",
        "severely_affected",
    ]
    readability_evidence: str = Field(description="Short evidence explaining readability")
    impurity_present: Literal["yes", "no"]
    impurity_types: List[
        Literal[
            "sticker",
            "paint_loss",
            "discoloration",
            "burn_or_blackening",
            "graffiti",
            "contamination",
            "damage",
        ]
    ] = Field(default_factory=list)


class Task1BaselineResult(BaseModel):
    readability: Literal["readable", "partly_readable", "unreadable", "unclear"]
    explanation: str = Field(description="Short practical explanation")
