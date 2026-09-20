"""Copy persistent model weights to local Colab disk when requested."""

import hashlib
import shutil
from pathlib import Path
from uuid import uuid4


def stage_model(
    source: Path, cache_dir: Path, *, enabled: bool, digest: str = "unresolved"
) -> Path:
    if not enabled:
        return source
    if not source.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {source}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in source.rglob("*") if path.is_file())
    metadata = "|".join(
        f"{path.relative_to(source)}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
        for path in files
    )
    identity = hashlib.sha256(
        f"{source.resolve()}:{digest}:{metadata}".encode()
    ).hexdigest()[:10]
    target = cache_dir / f"{source.name}-{identity}"
    marker = target / ".arex-copy-complete"
    if marker.exists():
        return target
    if target.exists():
        raise FileExistsError(f"incomplete staged model directory: {target}")
    size = sum(path.stat().st_size for path in files)
    free = shutil.disk_usage(cache_dir).free
    if free < size * 1.1:
        raise OSError(f"model staging needs {size} bytes; only {free} bytes free")
    temporary = cache_dir / f".{target.name}-{uuid4().hex}.partial"
    try:
        shutil.copytree(source, temporary)
        (temporary / marker.name).write_text("complete\n")
        temporary.rename(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target
