from __future__ import annotations

from datetime import datetime
from typing import Any


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %I:%M:%S %p")


def _duration_seconds(start: datetime, end: datetime) -> int:
    return max(0, int((end - start).total_seconds()))


def format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} seconds"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} minute(s), {rem} second(s)"
    hours, rem_min = divmod(minutes, 60)
    return f"{hours} hour(s), {rem_min} minute(s), {rem} second(s)"


def generate_sar(decision: dict[str, Any], camera_id: str, camera_rule: dict[str, Any], video_filename: str) -> str:
    """
    Deterministic SAR template. Later, you can replace this with an LLM call,
    but only call that LLM when this function is reached for a violation.
    """
    event = decision["event"]
    start = decision["event_start"]
    end = decision["event_end"]
    duration = format_duration(_duration_seconds(start, end))
    matched_rule = decision.get("matched_rule") or {}
    window = decision.get("matched_window") or {}

    vehicle_bits = []
    if event.get("color"):
        vehicle_bits.append(str(event["color"]))
    if event.get("vehicle_type"):
        vehicle_bits.append(str(event["vehicle_type"]))
    vehicle_desc = " ".join(vehicle_bits) if vehicle_bits else "vehicle"

    confidence = event.get("confidence")
    confidence_line = f"\nModel confidence: {confidence:.2f}" if isinstance(confidence, (int, float)) else ""

    return f"""Suspicious Activity Report

Camera: {camera_id} - {camera_rule.get("display_name", camera_id)}
Location: {camera_rule.get("location", "Unknown")}
Source video: {video_filename}

Summary:
A {vehicle_desc} was observed entering campus during a restricted access period.

Time observed:
- First seen: {_fmt_dt(start)}
- Last seen: {_fmt_dt(end)}
- Duration visible: {duration}

Observed activity:
{event.get("summary", "No activity summary provided.")}

Rule triggered:
- {matched_rule.get("label", "Prohibited access rule")}
- Restricted window: {window.get("start", "unknown")} to {window.get("end", "unknown")}
- Reason: {decision.get("reason", "Matched a prohibited access condition.")}

Approved activities for this camera during the restricted period include:
{chr(10).join(f"- {item}" for item in camera_rule.get("approved_activity", []))}{confidence_line}

Disposition:
Generate SAR for human review. This report is based on automated video interpretation and should be verified against the source footage.
"""
