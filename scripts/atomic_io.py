"""Durable JSON publication shared by training and local recovery tooling."""
import json
import os
from pathlib import Path
import tempfile
import shutil


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def publish_new_file(source: Path, target: Path) -> None:
    """Publish a complete file without ever replacing an existing target."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + target.name, dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as original:
            shutil.copyfileobj(original, out)
            out.flush(); os.fsync(out.fileno())
        os.link(temporary, target)
        if os.name == "posix":
            directory = os.open(target.parent, os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
