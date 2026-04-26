from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse, unquote

import boto3
import streamlit as st

from models import check_aws_identity, extract_events_with_pegasus
from rules import evaluate_events, load_rules, parse_video_filename
from sar import generate_sar

REGION = "us-east-1"
MODEL_ID = "us.twelvelabs.pegasus-1-2-v1:0"

VIDEO_CACHE_DIR = Path("video_cache")
CLIP_DIR = Path("sar_clips")
VIDEO_CACHE_DIR.mkdir(exist_ok=True)
CLIP_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Campus SAR Demo", layout="wide")
st.title("Campus Suspicious Activity Report Demo")
st.caption(
    "Enter an S3 video URI. The app parses camera ID and recording start time from the filename, "
    "loads the matching ruleset, and generates a SAR only if a rule is violated."
)


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    parsed = urlparse(s3_uri.strip())
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(
            "Expected an S3 URI like "
            "s3://bucket/camera_01__20260425_210000__north_entry_road.mp4"
        )
    return parsed.netloc, unquote(parsed.path.lstrip("/"))


def filename_from_s3_uri(s3_uri: str) -> str:
    _, key = parse_s3_uri(s3_uri)
    return Path(key).name


@st.cache_data
def get_rules():
    return load_rules("camera_rules.json")


def get_ffmpeg_executable() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg was not found, and imageio-ffmpeg is not installed. "
            "Run: pip install imageio-ffmpeg"
        ) from exc


def download_s3_video_once(s3_uri: str, region_name: str = REGION) -> Path:
    bucket, key = parse_s3_uri(s3_uri)
    suffix = Path(key).suffix or ".mp4"
    digest = hashlib.md5(s3_uri.encode("utf-8")).hexdigest()
    local_path = VIDEO_CACHE_DIR / f"{digest}{suffix}"

    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    s3 = boto3.Session(region_name=region_name).client("s3")
    s3.download_file(bucket, key, str(local_path))
    return local_path


def create_event_clip(
    *,
    source_video_path: Path,
    event: dict,
    clip_name: str,
    padding_sec: float = 5.0,
) -> tuple[Path, float, float]:
    """Create a browser-playable MP4 clip around the event offsets returned by Pegasus."""
    start_offset = float(event.get("start_offset_sec") or 0)
    end_offset = float(event.get("end_offset_sec") or start_offset)

    clip_start = max(0.0, start_offset - padding_sec)
    clip_end = max(clip_start + 1.0, end_offset + padding_sec)
    duration = clip_end - clip_start

    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in clip_name)
    output_path = CLIP_DIR / f"{safe_name}.mp4"

    # Always recreate the clip so stale/bad files do not hide the problem.
    if output_path.exists():
        output_path.unlink()

    ffmpeg_exe = get_ffmpeg_executable()

    # Re-encode to H.264 MP4 for browser-compatible Streamlit playback.
    # -an removes audio to avoid failures when source videos have no audio stream.
    cmd = [
        ffmpeg_exe,
        "-y",
        "-ss",
        str(clip_start),
        "-i",
        str(source_video_path),
        "-t",
        str(duration),
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg clip creation failed.\n\n"
            f"Command:\n{' '.join(cmd)}\n\n"
            f"STDOUT:\n{completed.stdout}\n\n"
            f"STDERR:\n{completed.stderr}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg finished, but no usable clip was created at {output_path}")

    return output_path, clip_start, clip_end


rules = get_rules()

s3_uri = st.text_input(
    "S3 video URI",
    value="",
    placeholder="s3://levis331/camera_01__20260425_210000__north_entry_road.mp4",
)

analyze = st.button("Analyze video", type="primary")

if not analyze:
    st.stop()

try:
    filename = filename_from_s3_uri(s3_uri)
    parsed_meta = parse_video_filename(filename)
except Exception as exc:
    st.error(f"Could not read S3 URI / filename: {exc}")
    st.stop()

camera_id = parsed_meta.get("camera_id")
recording_start = parsed_meta.get("recording_start")

if not camera_id:
    st.error(
        "Could not parse camera ID from filename. Expected a filename like: "
        "camera_01__20260425_210000__north_entry_road.mp4"
    )
    st.stop()

if not recording_start:
    st.error(
        "Could not parse recording start time from filename. Expected a filename like: "
        "camera_01__20260425_210000__north_entry_road.mp4"
    )
    st.stop()

if camera_id not in rules:
    st.error(f"Parsed {camera_id}, but no matching ruleset exists in camera_rules.json.")
    st.stop()

camera_rule = rules[camera_id]

st.subheader("Parsed video metadata")
st.write(
    {
        "s3_uri": s3_uri,
        "filename": filename,
        "camera_id": camera_id,
        "recording_start": recording_start.strftime("%Y-%m-%d %H:%M:%S"),
        "ruleset": camera_rule.get("display_name", camera_id),
    }
)

try:
    identity = check_aws_identity(REGION)
    st.caption(f"AWS identity confirmed: {identity.get('arn')}")
except Exception as exc:
    st.error(f"AWS credentials are not available to this Streamlit process: {exc}")
    st.stop()

with st.spinner("Calling AWS Bedrock / TwelveLabs Pegasus for structured event extraction..."):
    try:
        events, model_debug = extract_events_with_pegasus(
            camera_id=camera_id,
            camera_rule=camera_rule,
            s3_uri=s3_uri,
            region_name=REGION,
            model_id=MODEL_ID,
        )
    except Exception as exc:
        st.error(f"Pegasus extraction failed: {exc}")
        st.stop()

# Deterministic rule engine runs after Pegasus.
decisions = evaluate_events(events, camera_rule, recording_start)
violations = [d for d in decisions if d["status"] == "violation"]

st.markdown("---")

if not violations:
    st.success("No suspicious activity detected. No SAR generated.")
    st.caption(f"Reviewed {len(events)} access-control-relevant event(s) against {camera_id} rules.")
else:
    st.error(f"{len(violations)} suspicious event(s) detected. SAR generated only for rule violations.")

    source_video_path: Path | None = None
    with st.spinner("Downloading source video from S3 for SAR clips..."):
        try:
            source_video_path = download_s3_video_once(s3_uri, region_name=REGION)
            st.caption(
                f"Source video cached locally: {source_video_path} "
                f"({source_video_path.stat().st_size / 1024 / 1024:.1f} MB)"
            )
        except Exception as exc:
            st.warning(f"SAR reports were generated, but the source video could not be downloaded for clipping: {exc}")

    for idx, violation in enumerate(violations, start=1):
        event = violation["event"]
        event_id = event.get("event_id", idx)

        report = generate_sar(
            decision=violation,
            camera_id=camera_id,
            camera_rule=camera_rule,
            video_filename=filename,
        )

        st.subheader(f"SAR {idx}")
        report_col, video_col = st.columns([1, 1])

        with report_col:
            st.text_area("Report", value=report, height=420, key=f"sar_{idx}")
            st.download_button(
                label=f"Download SAR {idx}",
                data=report,
                file_name=f"SAR_{camera_id}_{event_id}.txt",
                mime="text/plain",
            )

        with video_col:
            st.markdown("**Triggered video segment**")

            if source_video_path is None:
                st.info("Video clip unavailable because the source video download failed.")
                continue

            try:
                uri_hash = hashlib.md5(s3_uri.encode("utf-8")).hexdigest()[:8]
                clip_path, clip_start, clip_end = create_event_clip(
                    source_video_path=source_video_path,
                    event=event,
                    clip_name=f"{camera_id}_{event_id}_{uri_hash}",
                    padding_sec=5.0,
                )

                clip_size_mb = clip_path.stat().st_size / 1024 / 1024
                st.caption(
                    f"Created clip: {clip_path} ({clip_size_mb:.2f} MB). "
                    f"Clip window: {clip_start:.1f}s to {clip_end:.1f}s."
                )

                # Bytes display is more reliable than passing a local server path in proxied environments.
                st.video(clip_path.read_bytes(), format="video/mp4")

                with open(clip_path, "rb") as f:
                    st.download_button(
                        label=f"Download SAR {idx} video clip",
                        data=f,
                        file_name=clip_path.name,
                        mime="video/mp4",
                    )

            except Exception as exc:
                st.error(f"Could not create/display video clip for SAR {idx}: {exc}")
