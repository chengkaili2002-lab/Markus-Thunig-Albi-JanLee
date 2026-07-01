"""Task 5 — Pydantic schema matching prompts/task5.txt."""
from typing import List
from pydantic import BaseModel, Field


class ImageQuality(BaseModel):
    class Fixed(BaseModel):
        is_clear:   str = Field(description="yes | no | uncertain")
        brightness: str = Field(description="normal | too_dark | overexposed | uncertain")
    fixed: Fixed
    open_observations: List[str] = []


class WeatherTime(BaseModel):
    class Fixed(BaseModel):
        weather:     str = Field(description="clear | rain | snow | fog | unclear")
        time_of_day: str = Field(description="day | night | twilight | unclear")
    fixed: Fixed
    open_observations: List[str] = []


class RoadVisibility(BaseModel):
    class Fixed(BaseModel):
        road_visible:                str = Field(description="yes | no | uncertain")
        oncoming_lane_present:       str = Field(description="yes | no | uncertain")
        intersection_visible:        str = Field(description="yes | no | uncertain")
        visible_lane_marking_count:  str = Field(description="none | one | multiple | unclear")
        road_surface_type:           str = Field(description="paved_road | dirt_road | cobblestone_road | unclear")
    fixed: Fixed
    open_observations: List[str] = []


class Surroundings(BaseModel):
    class Fixed(BaseModel):
        front_visible_range:       str = Field(description="within_50m | around_100m | 200m_or_more | unclear")
        left_side_visible_range:   str = Field(description="within_25m | around_50m | 100m_or_more | unclear")
        right_side_visible_range:  str = Field(description="within_25m | around_50m | 100m_or_more | unclear")
        buildings_present:         str = Field(description="yes | no | uncertain")
        buildings_close_to_road:   str = Field(description="yes | no | uncertain")
        building_floor_level:      str = Field(description="none | one_floor | two_floors | three_floors | four_or_more_floors | unclear")
        environment_type:          str = Field(description="urban | suburban | open_area | unclear")
    fixed: Fixed
    open_observations: List[str] = []


class TrafficContext(BaseModel):
    class Fixed(BaseModel):
        traffic_density:      str = Field(description="low | medium | high | uncertain")
        traffic_sign_visible: str = Field(description="yes | no | uncertain")
    fixed: Fixed
    open_observations: List[str] = []


class Task5Output(BaseModel):
    image_quality:   ImageQuality
    weather_time:    WeatherTime
    road_visibility: RoadVisibility
    surroundings:    Surroundings
    traffic_context: TrafficContext
