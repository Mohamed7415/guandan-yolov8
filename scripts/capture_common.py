"""Shared validation for real, locally saved PNG/JPEG screenshots.

This module never captures a screen or creates image data.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from pathlib import Path

FIELDS = (
    "filename", "session_id", "group_id", "scene", "captured_at",
    "sha256", "width", "height",
)
SCENES = ("self_play", "others_play", "pass", "fan", "sorted", "negative")
IMAGE_FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}


def parse_scenes(value: str) -> list[str]:
    scenes = [part.strip() for part in value.split("|") if part.strip()]
    if not scenes or any(scene not in SCENES for scene in scenes):
        raise ValueError(f"scene must use one or more of {', '.join(SCENES)} separated by |")
    return [scene for scene in SCENES if scene in scenes]


def validate_timestamp(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("captured_at must contain a timezone, e.g. 2026-10-16T14:30:00+08:00")
    return parsed.isoformat(timespec="seconds")


def manifest_path(raw_dir: Path, filename: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts or not filename.strip():
        raise ValueError("filename must be a path relative to raw_dir without '..'")
    path = (raw_dir / relative).resolve()
    if not path.is_relative_to(raw_dir.resolve()):
        raise ValueError("filename resolves outside raw_dir")
    return path


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], ["manifest_missing"]
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        missing = [field for field in FIELDS if field not in headers]
        rows = [dict(row) for row in reader]
    errors = [f"manifest_missing_column:{field}" for field in missing]
    if len(headers) != len(set(headers)):
        errors.append("manifest_duplicate_columns")
    return rows, errors


def inspect_image(path: Path) -> dict:
    """Validate magic/extension and decode an original PNG/JPEG without rewriting it."""
    result = {"sha256": None, "pixel_sha256": None, "format": None, "width": None, "height": None, "dhash": None, "error": None}
    if path.stat().st_size == 0:
        result["error"] = "empty_file"
        return result
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        magic = stream.read(16)
        stream.seek(0)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    result["sha256"] = digest.hexdigest()
    detected_format = ("PNG" if magic.startswith(b"\x89PNG\r\n\x1a\n")
                       else "JPEG" if magic.startswith(b"\xff\xd8\xff") else None)
    result["format"] = detected_format
    expected_format = IMAGE_FORMATS.get(path.suffix.lower())
    if expected_format is None:
        result["error"] = "unsupported_extension"
        return result
    if detected_format is None:
        result["error"] = "unsupported_image_magic"
        return result
    if detected_format != expected_format:
        result["error"] = f"extension_format_mismatch:{path.suffix.lower()}:{detected_format}"
        return result
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required. Use the project's training environment after dependency setup.") from exc
    try:
        with Image.open(path) as original:
            if original.format != detected_format:
                result["error"] = "decoder_format_mismatch"
                return result
            original.load()
            result.update(width=original.width, height=original.height)
            if original.width < 2 or original.height < 2:
                result["error"] = "empty_image"
                return result
            rgba = original.convert("RGBA")
            if rgba.getchannel("A").getextrema()[1] == 0:
                result["error"] = "transparent_image"
                return result
            image = Image.new("RGB", rgba.size, "white")
            image.paste(rgba, mask=rgba.getchannel("A"))
            if all(high - low <= 1 for low, high in image.getextrema()):
                result["error"] = "blank_image"
                return result
            # Lossless resaves can have different file hashes but identical pixels.
            pixel_digest = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode("ascii"))
            pixel_digest.update(image.tobytes())
            result["pixel_sha256"] = pixel_digest.hexdigest()
            pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
            dhash = 0
            for y in range(8):
                for x in range(8):
                    dhash = (dhash << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
            result["dhash"] = f"{dhash:016x}"
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        result["error"] = f"decode_error:{type(exc).__name__}"
    return result
