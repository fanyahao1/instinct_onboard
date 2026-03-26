"""instinct_onboard agents module.

This module provides various agent implementations for robot control:
- OnboardAgent: Base class for onboard agents
- ColdStartAgent: Agent for safe joint initialization
- ParkourAgent: Agent for parkour tasks with depth perception
- ParkourStandAgent: Agent for standing/balancing
- TrackerAgent: Agent for motion tracking from NPZ files
- PerceptiveTrackerAgent: TrackerAgent with depth perception
- ShadowingAgent: Agent for ROS topic-based motion reference
- WalkAgent: Agent for basic walking
- InteractionAgent: Agent for object interaction tasks
"""

from instinct_onboard.agents.base import ColdStartAgent, OnboardAgent
from instinct_onboard.agents.interaction_agent import InteractionAgent, MotionData
from instinct_onboard.agents.parkour_agent import ParkourAgent, ParkourStandAgent
from instinct_onboard.agents.shadowing_agent import MotionAsActAgent, ShadowingAgent
from instinct_onboard.agents.tracking_agent import PerceptiveTrackerAgent, TrackerAgent
from instinct_onboard.agents.walk_agent import WalkAgent

__all__ = [
    "OnboardAgent",
    "ColdStartAgent",
    "ParkourAgent",
    "ParkourStandAgent",
    "TrackerAgent",
    "PerceptiveTrackerAgent",
    "ShadowingAgent",
    "MotionAsActAgent",
    "WalkAgent",
    "InteractionAgent",
    "MotionData",
]
