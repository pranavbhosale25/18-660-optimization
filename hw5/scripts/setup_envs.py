"""Install the two independently pinned homework environments with Python 3.12."""

import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def install_environment(folder, requirements, projects):
    path = ROOT / folder
    venv.EnvBuilder(with_pip=True).create(path)
    python = path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    base = [str(python), "-m", "pip", "install"]
    subprocess.run(base + ["-r", str(ROOT / requirements)], check=True)
    if folder == ".venv":
        subprocess.run(
            base
            + [
                "--no-index",
                "--no-deps",
                "--find-links",
                str(ROOT / "Application/P1/jaxbo_review/wheels"),
                "hw5-bo-simulator-runtime==0.2.0",
            ],
            check=True,
        )
    for project in projects:
        subprocess.run(base + ["--no-deps", "-e", str(ROOT / project)], check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)


if __name__ == "__main__":
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(
            "Run this script with Python 3.12 (required by the supplied simulator wheel)."
        )
    install_environment(
        ".venv",
        "requirements-main.txt",
        ["Application/P1/jaxbo_review", "Implementation/jax_reinforce"],
    )
    install_environment(
        ".venv-l2ws", "requirements-warmstart.txt", ["Application/P2/l2ws_jax"]
    )
