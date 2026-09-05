"""Download public assets atomically and verify their published checksums."""

import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_verified(url: str, destination: str | Path, sha256: str) -> Path:
    """Reuse a verified asset or download it without overwriting unknown files."""
    destination = Path(destination)
    if destination.exists():
        if sha256_file(destination) != sha256:
            raise ValueError(f"Checksum mismatch for existing file: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = Request(url, headers={"User-Agent": "VocalRep/1.0"})
        with (
            urlopen(request, timeout=120) as response,
            tempfile.NamedTemporaryFile(
                dir=destination.parent, suffix=".tmp", delete=False
            ) as stream,
        ):
            temporary = Path(stream.name)
            digest = hashlib.sha256()
            for block in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(block)
                digest.update(block)
        if digest.hexdigest() != sha256:
            raise ValueError(f"Downloaded asset failed SHA-256 verification: {url}")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
