"""Typeset the Markdown writeups and append them to the handwritten theory PDF.

Requires Pandoc, Tectonic, and the submission's main Python environment.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/pranavbhosale25/18-660-optimization"
CASES = {
    "FrozenLake",
    "Contextual bandit",
    "CartPole",
    "Pendulum",
    "Coating",
    "Pipeline",
    "Production",
    "Materials",
    "Controller",
}
HEADER = r"""
\usepackage{fancyhdr,float,pdflscape,caption,etoolbox,needspace}
\definecolor{heading}{HTML}{173F50}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textcolor{heading}{18-660: Optimization / HW5}}
\fancyhead[R]{\small Q2 and Q3 writeups}
\fancyfoot[C]{\small\thepage}
\setlength{\headheight}{14pt}
\setlength{\emergencystretch}{2em}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\AtBeginEnvironment{longtable}{\small}
\floatplacement{figure}{H}
\captionsetup{font=small,labelfont=bf}
\AtBeginDocument{\hypersetup{pdftitle={Optimization HW5: Q2 and Q3 writeups}}}
"""


def inline_text(value):
    if isinstance(value, dict):
        if value.get("t") == "Str":
            return value["c"]
        return inline_text(value.get("c", []))
    if isinstance(value, list):
        return " ".join(inline_text(item) for item in value)
    return ""


def transform(value, source, in_table=False):
    if isinstance(value, list):
        return [transform(item, source, in_table) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("t")
    in_table = in_table or kind == "Table"
    if kind == "Image":
        value["c"][2][0] = str((source.parent / value["c"][2][0]).resolve())
    elif kind == "Link":
        target = value["c"][2][0]
        if not target.startswith(("https:", "http:", "#")):
            path, _, anchor = target.partition("#")
            local = (source.parent / path).resolve()
            route = "tree" if local.is_dir() else "blob"
            relative = local.relative_to(ROOT).as_posix()
            value["c"][2][0] = f"{GITHUB}/{route}/main/hw5/{quote(relative)}"
            if anchor:
                value["c"][2][0] += "#" + anchor
    elif kind == "Code" and in_table:
        words = value["c"][1].replace("_", " ").split()
        return {
            "t": "Span",
            "c": [
                ["", [], []],
                [
                    element
                    for index, word in enumerate(words)
                    for element in ([{"t": "Space"}] if index else [])
                    + [{"t": "Str", "c": word}]
                ],
            ],
        }
    if kind == "Table":
        columns = value["c"][2]
        title = inline_text(value)
        weights = {
            3: [14, 43, 43],
            4: [17, 18, 30, 35],
            5: [16, 14, 28, 18, 24],
            8: [10, 25, 10, 7, 6, 11, 7, 11],
        }.get(len(columns), [14, 22, 16, 16, 16, 16])
        if len(columns) == 6 and "Candidate" in title:
            weights = [23, 9, 14, 12, 21, 21]
        for column, weight in zip(columns, weights):
            column[1] = {"t": "ColWidth", "c": weight / sum(weights)}
    return {key: transform(item, source, in_table) for key, item in value.items()}


def raw(text):
    return {"t": "RawBlock", "c": ["latex", text]}


def writeup_blocks(source, title):
    markdown = source.read_text().replace("–", "-").replace("—", "-")
    if source.parent.name == "Application":
        # Keep the shared plot legend with the experimental protocol.
        notes = [
            line
            for line in markdown.splitlines()
            if line.startswith(("- In every figure,", "- These are single-instance,"))
        ]
        for line in notes:
            markdown = markdown.replace(line + "\n", "")
        markdown = markdown.replace(
            "### Results\n", "### Results\n\n" + "\n".join(notes) + "\n", 1
        )
    parsed = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "json"],
        input=markdown,
        capture_output=True,
        text=True,
        check=True,
    )
    document = json.loads(parsed.stdout)
    blocks, landscape = [], False
    for block in document["blocks"]:
        block = transform(block, source)
        if block["t"] == "Header":
            level, attributes, content = block["c"]
            text = " ".join(inline_text(content).split())
            if level == 1:
                block["c"][2] = [{"t": "Str", "c": title}]
            elif level == 2 and text.startswith("Exercise 3.2"):
                blocks.append(raw(r"\clearpage"))
            elif source.parent.name == "Application" and text in {
                "Results",
                "Test results",
            }:
                blocks.append(raw(r"\clearpage"))
            elif level == 3 and text in CASES:
                blocks.append(raw(r"\begin{landscape}"))
                landscape = True
            else:
                blocks.append(raw(r"\Needspace{5\baselineskip}"))
        if (
            block["t"] == "Figure"
            and not landscape
            and "Kinematics" in inline_text(block)
        ):
            blocks.append(raw(r"\clearpage"))
        blocks.append(block)
        if block["t"] == "Figure" and landscape:
            blocks.append(raw(r"\end{landscape}"))
            landscape = False
    return document["pandoc-api-version"], blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latex-engine", default=shutil.which("tectonic") or "tectonic"
    )
    args = parser.parse_args()
    temporary = ROOT / "tmp/pdfs/writeups"
    output = ROOT / "output/pdf"
    temporary.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    theory = ROOT / "Theory/hw5.pdf"
    original_digest = hashlib.sha256(theory.read_bytes()).hexdigest()
    version, implementation = writeup_blocks(
        ROOT / "Implementation/writeup.md", "Q2: Implementation"
    )
    _, application = writeup_blocks(ROOT / "Application/writeup.md", "Q3: Application")
    ast = {
        "pandoc-api-version": version,
        "meta": {},
        "blocks": implementation + [raw(r"\clearpage")] + application,
    }
    header = temporary / "header.tex"
    header.write_text(HEADER)
    tex = temporary / "q2_q3_writeups.tex"
    subprocess.run(
        [
            "pandoc",
            "-f",
            "json",
            "-t",
            "latex",
            "--standalone",
            "-o",
            str(tex),
            "--include-in-header",
            str(header),
            "-V",
            "fontsize=10pt",
            "-V",
            "papersize=letter",
            "-V",
            "geometry:margin=0.65in",
            "-V",
            "colorlinks=true",
            "-V",
            "urlcolor=heading",
        ],
        input=json.dumps(ast),
        text=True,
        check=True,
    )
    subprocess.run(
        [args.latex_engine, "-X", "compile", str(tex), "--outdir", str(output)],
        check=True,
    )
    typed = output / "q2_q3_writeups.pdf"
    combined = output / "hw5_complete_submission.pdf"
    writer = PdfWriter()
    writer.append(str(theory), outline_item="Q1: Handwritten theory")
    writer.append(str(typed), outline_item="Q2 and Q3: Implementation and application")
    writer.add_metadata({"/Title": "Optimization HW5: Complete submission"})
    with combined.open("wb") as stream:
        writer.write(stream)
    assert hashlib.sha256(theory.read_bytes()).hexdigest() == original_digest
    assert len(PdfReader(combined).pages) == len(PdfReader(theory).pages) + len(
        PdfReader(typed).pages
    )
    print(f"Created {typed} and {combined}")


if __name__ == "__main__":
    main()
