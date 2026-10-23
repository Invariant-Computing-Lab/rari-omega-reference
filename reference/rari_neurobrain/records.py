"""Canonical, durable records. Immutable receipts never silently replace evidence."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def durable_bytes(path, data, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)  # Atomic fail-if-exists, also with concurrent writers.
        d = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(d)
        finally:
            os.close(d)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write(path, value, *, replace=False):
    durable_bytes(path, canonical(value), replace=replace)


def verify_files(root, manifest):
    root = Path(root).resolve()
    for name, expected in manifest.items():
        p = root / name
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError("MANIFEST_PATH_ESCAPE")
        if file_hash(p) != expected:
            raise ValueError("MANIFEST_HASH_MISMATCH:" + name)

