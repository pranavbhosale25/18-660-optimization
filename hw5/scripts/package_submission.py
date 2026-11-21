"""Package portable sources and measured outputs, excluding local environments."""

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Optimization_HW5_Submission.zip"
CONTENTS = [
    "Application",
    "Implementation",
    "Theory/hw5.pdf",
    "scripts",
    "README.md",
    "reproduce.py",
    "requirements-main.txt",
    "requirements-warmstart.txt",
    "Optimization_HW5.pdf",
    "Optimization_Syllabus_ECE_Template_Converted_Short.pdf",
    "lecture-18.pdf",
]
EXCLUDED = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv", ".git"}


def files_to_include():
    for name in CONTENTS:
        source = ROOT / name
        paths = source.rglob("*") if source.is_dir() else [source]
        for path in paths:
            relative = path.relative_to(ROOT)
            if not path.is_file() or path.name == ".DS_Store":
                continue
            if any(
                part in EXCLUDED or part.endswith(".egg-info")
                for part in relative.parts
            ):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            yield relative


if __name__ == "__main__":
    files = sorted(files_to_include())
    manifest = "".join(
        f"{hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}  {path.as_posix()}\n"
        for path in files
    )
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(ROOT / path, f"Optimization_HW5_Submission/{path.as_posix()}")
        archive.writestr("Optimization_HW5_Submission/SHA256SUMS.txt", manifest)
    with zipfile.ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive integrity verification failed")
    print(
        f"Packaged {len(files)} files: {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.2f} MB)"
    )
