from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
import json
import re
from typing import Any


FILENAME_PATTERN = re.compile(
    r"(?P<camera_id>camera[_-]?\d+)"
    r"(?:__|-)?"
    r"(?P<date>\d{8})?"
    r"(?:[_-]?(?P<clock>\d{6}))?",
    re.IGNORECASE,
)


def normalize_camera_id(value: str) -> str:
    """Normalize camera-1, camera01, Camera_01, etc. to camera_01."""
    value = value.strip().lower().replace("-", "_")
    match = re.search(r"camera_?(\d+)", value)
    if not match:
        return value
    return f"camera_{int(match.group(1)):02d}"


def parse_video_filename(filename: str) -> dict[str, Any]:
    """
    Expected pattern:
        camera_01__20260425_210000__north_entry_road.mp4

    Returns camera_id and recording_start if found. The app lets the user override.
    """
    stem = Path(filename).stem
    match = FILENAME_PATTERN.search(stem)

    parsed = {
        "filename": filename,
        "camera_id": None,
        "recording_start": None,
        "label": None,
    }

    if not match:
        return parsed

    if match.group("camera_id"):
        parsed["camera_id"] = normalize_camera_id(match.group("camera_id"))

    if match.group("date") and match.group("clock"):
        parsed["recording_start"] = datetime.strptime(
            f"{match.group('date')}{match.group('clock')}",
            "%Y%m%d%H%M%S",
        )

    # Optional descriptive label after the timestamp.
    parts = stem.split("__")
    if len(parts) >= 3:
        parsed["label"] = parts[-1].replace("_", " ")

    return parsed


def load_rules(path: str | Path = "camera_rules.json") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        rules = json.load(f)

    return {normalize_camera_id(k): v for k, v in rules.items()}


def _parse_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M:%S").time()


def is_within_time_window(moment: datetime, start_str: str, end_str: str) -> bool:
    """
    Handles normal and overnight windows.
    Example:
        18:00:00 to 06:00:00 means 6 PM through 6 AM.
    """
    start = _parse_time(start_str)
    end = _parse_time(end_str)
    t = moment.time()

    if start <= end:
        return start <= t <= end

    # Overnight window.
    return t >= start or t <= end


def event_absolute_times(event: dict[str, Any], recording_start: datetime) -> tuple[datetime, datetime]:
    start_offset = float(event.get("start_offset_sec") or 0)
    end_offset = float(event.get("end_offset_sec") or start_offset)
    start_dt = recording_start + timedelta(seconds=start_offset)
    end_dt = recording_start + timedelta(seconds=end_offset)
    return start_dt, end_dt


def _contains_all(text: str | None, required_parts: list[str] | None) -> bool:
    if not required_parts:
        return True

    text_norm = (text or "").lower()
    return all(part.lower() in text_norm for part in required_parts)


def _sameish(a: str | None, b: str | None) -> bool:
    if b is None:
        return True
    return (a or "").strip().lower() == b.strip().lower()


def check_event_against_camera_rules(
    event: dict[str, Any],
    camera_rule: dict[str, Any],
    recording_start: datetime,
) -> dict[str, Any]:
    """
    Returns a decision object:
        status: approved | violation | not_applicable
    """
    event_start, event_end = event_absolute_times(event, recording_start)

    active_windows = []
    for window in camera_rule.get("restricted_windows", []):
        if is_within_time_window(event_start, window["start"], window["end"]):
            active_windows.append(window)

    if not active_windows:
        return {
            "status": "approved",
            "reason": "Event occurred outside the restricted time window.",
            "event": event,
            "event_start": event_start,
            "event_end": event_end,
            "matched_rule": None,
            "matched_window": None,
        }

    for prohibited in camera_rule.get("prohibited_rules", []):
        object_type_match = _sameish(event.get("object_type"), prohibited.get("object_type"))
        classification_match = _sameish(event.get("classification"), prohibited.get("classification"))
        route_match = _contains_all(event.get("route"), prohibited.get("route_contains"))
        movement_match = _contains_all(event.get("movement"), prohibited.get("movement_contains"))

        if object_type_match and classification_match and route_match and movement_match:
            return {
                "status": "violation",
                "reason": prohibited.get("sar_reason", "Event matched a prohibited access rule."),
                "event": event,
                "event_start": event_start,
                "event_end": event_end,
                "matched_rule": prohibited,
                "matched_window": active_windows[0],
            }

    return {
        "status": "approved",
        "reason": "Event did not match a prohibited access rule.",
        "event": event,
        "event_start": event_start,
        "event_end": event_end,
        "matched_rule": None,
        "matched_window": active_windows[0],
    }


def evaluate_events(
    events: list[dict[str, Any]],
    camera_rule: dict[str, Any],
    recording_start: datetime,
) -> list[dict[str, Any]]:
    return [
        check_event_against_camera_rules(event, camera_rule, recording_start)
        for event in events
    ]
