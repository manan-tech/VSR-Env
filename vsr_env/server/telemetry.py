"""Telemetry tracking for VSR-Env.

Records all episode actions, Greeks, P&L traces, and rewards.
"""

from typing import Any, Dict, Optional
from datetime import datetime


class TelemetryTracker:
    """Global telemetry tracker for VSR-Env episodes."""

    def __init__(self):
        """Initialize the telemetry tracker."""
        self.episodes: Dict[str, Dict[str, Any]] = {}
        self.active_episode: Optional[str] = None

    def start_episode(
        self, episode_id: str, task_name: str, seed: Optional[int] = None
    ) -> None:
        """Start tracking a new episode."""
        self.active_episode = episode_id
        self.episodes[episode_id] = {
            "episode_id": episode_id,
            "task_name": task_name,
            "seed": seed,
            "start_time": datetime.utcnow().isoformat(),
            "end_time": None,
            "steps": [],
            "final_score": None,
            "completed": False,
        }

    def record_step(self, episode_id: str, step_data: Dict[str, Any]) -> None:
        """Record an action/observation step in the active episode."""
        if episode_id in self.episodes:
            self.episodes[episode_id]["steps"].append(step_data)

    def complete_episode(self, episode_id: str, final_score: float) -> None:
        """Mark an episode as completed and save final score."""
        if episode_id in self.episodes:
            self.episodes[episode_id]["end_time"] = datetime.utcnow().isoformat()
            self.episodes[episode_id]["final_score"] = final_score
            self.episodes[episode_id]["completed"] = True

            if self.active_episode == episode_id:
                self.active_episode = None

    def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Get the telemetry data for a specific episode."""
        return self.episodes.get(episode_id)

    def get_all_episodes(self) -> Dict[str, Dict[str, Any]]:
        """Get the telemetry data for all episodes."""
        return self.episodes


# Global telemetry instance
tracker = TelemetryTracker()
