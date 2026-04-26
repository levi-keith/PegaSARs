from __future__ import annotations

import base64
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "events": {
            "type": "array",
            "description": "Only access-control-relevant events visible in the video.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "event_id": {"type": "string"},
                    "object_type": {
                        "type": "string",
                        "enum": ["vehicle", "person", "animal", "other"],
                    },
                    "classification": {
                        "type": "string",
                        "enum": [
                            "entering_campus",
                            "exiting_campus",
                            "foot_traffic",
                            "approved_pickup_dropoff",
                            "other",
                        ],
                    },
                    "route": {
                        "type": "string",
                        "description": "Named route/area, e.g. small road, service road, large circle drive.",
                    },
                    "movement": {
                        "type": "string",
                        "description": "Direction of travel relative to camera-specific landmarks.",
                    },
                    "vehicle_type": {
                        "type": ["string", "null"],
                        "description": "sedan, SUV, pickup truck, minivan, van, bus, truck, unknown, or null for non-vehicles.",
                    },
                    "color": {
                        "type": ["string", "null"],
                        "description": "Visible vehicle color or null for non-vehicles.",
                    },
                    "start_offset_sec": {
                        "type": "number",
                        "description": "Seconds from video start when the event first appears.",
                    },
                    "end_offset_sec": {
                        "type": "number",
                        "description": "Seconds from video start when the event ends or leaves view.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief factual description of the event.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": [
                    "event_id",
                    "object_type",
                    "classification",
                    "route",
                    "movement",
                    "vehicle_type",
                    "color",
                    "start_offset_sec",
                    "end_offset_sec",
                    "summary",
                    "confidence",
                ],
            },
        }
    },
    "required": ["events"],
}


def _aws_region(region_name: str | None = None) -> str:
    load_dotenv()
    return region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"


def _aws_session(region_name: str | None = None):
    """Use the same default credential chain that works in SageMaker Studio terminals/notebooks."""
    import boto3

    load_dotenv()
    session = boto3.Session(region_name=_aws_region(region_name))
    if session.get_credentials() is None:
        raise RuntimeError(
            "No AWS credentials found for this Python/Streamlit process. "
            "Run Streamlit from the SageMaker Studio terminal where `aws sts get-caller-identity` works."
        )
    return session


def _clean_bucket_owner(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if value in {"", "123456789012", "your-account-id", "YOUR_ACCOUNT_ID"}:
        return None
    return value


def _default_bucket_owner(session, explicit_bucket_owner: str | None) -> str | None:
    """
    TwelveLabs' Bedrock workshop passes bucketOwner explicitly. If the UI field is blank,
    default to the current STS account so S3 input works the same way as the notebook.
    """
    explicit_bucket_owner = _clean_bucket_owner(explicit_bucket_owner or os.getenv("AWS_ACCOUNT_ID"))
    if explicit_bucket_owner:
        return explicit_bucket_owner

    try:
        return session.client("sts").get_caller_identity().get("Account")
    except Exception:
        return None


def load_mock_events(path: str | Path = "sample_mock_events.json", camera_id: str = "camera_01") -> list[dict[str, Any]]:
    """Kept only for offline UI/rule debugging. Do not use for the AWS-backed run path."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(camera_id, [])


def build_pegasus_event_prompt(camera_id: str, camera_rule: dict[str, Any]) -> str:
    """
    Pegasus should extract only events needed for rule checking. The SAR is generated later,
    after the deterministic rule engine decides there is a violation.
    """
    approved = "\n".join(f"- {x}" for x in camera_rule.get("approved_activity", [])) or "- None listed"
    prohibited = "\n".join(
        f"- {rule.get('label')}: object_type={rule.get('object_type')}; "
        f"classification={rule.get('classification')}; "
        f"route must contain={rule.get('route_contains')}; "
        f"movement must contain={rule.get('movement_contains')}"
        for rule in camera_rule.get("prohibited_rules", [])
    ) or "- None listed"

    return f"""
You are analyzing campus security drone footage from {camera_id}.

Task:
Extract only structured access-control events needed to check the camera rules below.
Do not generate a Suspicious Activity Report.
Do not describe unrelated activity unless it helps classify vehicle entry, vehicle exit, foot traffic, or approved pickup/drop-off circulation.
Use offsets in seconds from the start of the video.

Classification labels you must use:
- entering_campus: automotive traffic moving in the camera-specific prohibited entry direction.
- exiting_campus: automotive traffic moving in the approved exit direction.
- foot_traffic: people walking; this is approved unless a rule says otherwise.
- approved_pickup_dropoff: vehicle activity confined to the approved pickup/drop-off/circle-drive route.
- other: visible activity that is relevant but cannot be confidently classified above.

Camera-specific location:
{camera_rule.get("location", "Unknown")}

Approved activity for this camera:
{approved}

Prohibited activity definitions for this camera:
{prohibited}

Important rules:
- If there are no access-control-relevant events, return {{"events": []}}.
- If direction is unclear, use classification "other" and explain uncertainty in summary.
- Keep summaries factual and brief.
- Return only JSON matching the schema.
""".strip()


def _safe_json_loads(text: str) -> dict[str, Any]:
    """Parse JSON from Pegasus message, with a fallback if it wraps JSON in prose/code fences."""
    text = text.strip()
    if not text:
        return {"events": []}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return json.loads(text[first:last + 1])

    raise ValueError(f"Pegasus response did not contain parseable JSON: {text[:500]}")


def parse_pegasus_response(response_body: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bedrock returns Pegasus output in response_body['message']; structured output is a JSON string."""
    message = response_body.get("message", "{}")
    if isinstance(message, str):
        data = _safe_json_loads(message)
    elif isinstance(message, dict):
        data = message
    else:
        data = {"events": []}

    events = data.get("events", [])
    if not isinstance(events, list):
        events = []

    return events, {
        "finishReason": response_body.get("finishReason"),
        "raw_message": message,
        "parsed": data,
    }


def upload_video_to_s3(
    local_video_path: str | Path,
    bucket: str,
    prefix: str = "sar-demo/videos",
    region_name: str | None = None,
) -> str:
    """Uploads a local video to S3 and returns the s3:// URI."""
    session = _aws_session(region_name)
    local_video_path = Path(local_video_path)
    key = f"{prefix.rstrip('/')}/{uuid.uuid4().hex}_{local_video_path.name}"

    s3 = session.client("s3")
    s3.upload_file(str(local_video_path), bucket, key)
    return f"s3://{bucket}/{key}"


def _media_source_from_s3(s3_uri: str, bucket_owner: str | None = None) -> dict[str, Any]:
    s3_location = {"uri": s3_uri}
    if bucket_owner:
        s3_location["bucketOwner"] = bucket_owner
    return {"s3Location": s3_location}


def _media_source_from_base64(local_video_path: str | Path) -> dict[str, Any]:
    video_bytes = Path(local_video_path).read_bytes()
    encoded = base64.b64encode(video_bytes).decode("utf-8")
    return {"base64String": encoded}


def extract_events_with_pegasus(
    *,
    camera_id: str,
    camera_rule: dict[str, Any],
    s3_uri: str | None = None,
    local_video_path: str | Path | None = None,
    bucket_owner: str | None = None,
    model_id: str | None = None,
    region_name: str | None = None,
    max_output_tokens: int = 4096,
    use_base64: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Invoke TwelveLabs Pegasus 1.2 on Amazon Bedrock and return structured events."""
    load_dotenv()
    region_name = _aws_region(region_name)
    model_id = model_id or os.getenv("PEGASUS_MODEL_ID") or "us.twelvelabs.pegasus-1-2-v1:0"
    session = _aws_session(region_name)

    if use_base64:
        if not local_video_path:
            raise ValueError("local_video_path is required when use_base64=True")
        media_source = _media_source_from_base64(local_video_path)
        resolved_bucket_owner = None
    else:
        if not s3_uri:
            raise ValueError("s3_uri is required when use_base64=False")
        resolved_bucket_owner = _default_bucket_owner(session, bucket_owner)
        media_source = _media_source_from_s3(s3_uri, bucket_owner=resolved_bucket_owner)

    # IMPORTANT: Pegasus structured outputs in the TwelveLabs Bedrock workshop use
    # responseFormat: {"jsonSchema": <schema>}. Do not use OpenAI-style
    # {"type": "json_schema", "json_schema": ...}; Bedrock/Pegasus rejects that.
    request_body = {
        "inputPrompt": build_pegasus_event_prompt(camera_id, camera_rule),
        "mediaSource": media_source,
        "temperature": 0,
        "maxOutputTokens": max_output_tokens,
        "responseFormat": {
            "jsonSchema": EVENT_SCHEMA,
        },
    }

    client = session.client("bedrock-runtime")
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    events, debug = parse_pegasus_response(response_body)
    debug["request"] = {
        "model_id": model_id,
        "region_name": region_name,
        "s3_uri": s3_uri,
        "bucket_owner": resolved_bucket_owner if not use_base64 else None,
        "used_base64": use_base64,
        "response_format_style": "jsonSchema",
    }
    return events, debug


def check_aws_identity(region_name: str | None = None) -> dict[str, Any]:
    """Small convenience function for the Streamlit sidebar."""
    session = _aws_session(region_name)
    ident = session.client("sts").get_caller_identity()
    return {
        "account": ident.get("Account"),
        "arn": ident.get("Arn"),
        "user_id": ident.get("UserId"),
        "region": _aws_region(region_name),
    }
