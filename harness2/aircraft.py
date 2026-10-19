# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fine-grained aircraft recognition on FGVC-Aircraft test photos.

Each photo shows one aircraft; the task is to name its type among nine regional airliners
and turboprops that spotters confuse (ATR-42 vs ATR-72, CRJ-200 vs CRJ-700, ...). Rows come
from the Hugging Face mirror of the FGVC-Aircraft test split. The two tasks are a seeded
random split of 54 probe photos (six per type), not a hand-picked set.

FGVC-Aircraft images are made available by airplane spotters for non-commercial research
only; the photographers keep copyright. Photos are downloaded at run time, never committed.
"""

from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "Multimodal-Fatima/FGVC_Aircraft_test"
PAGE = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/"
LICENSE = "non-commercial research only"
TYPES = [
    "ATR-42",
    "ATR-72",
    "CRJ-200",
    "CRJ-700",
    "ERJ 145",
    "E-190",
    "Fokker 100",
    "Saab 340",
    "DHC-8-300",
]


def article(name: str) -> str:
    return f"{'an' if name[0] in 'AEIOU' else 'a'} {name}"


PLAIN = {name: article(name) for name in TYPES}
# What the prompt-rewriting tool writes: the cues spotters use to separate these types.
CUES = {
    "ATR-42": "ATR 42: high-wing twin turboprop, short fuselage",
    "ATR-72": "ATR 72: high-wing twin turboprop like the ATR 42 but clearly longer fuselage",
    "CRJ-200": "CRJ-200: small twin jet, engines on the rear fuselage, T-tail, short slim body, "
    "no wing leading-edge kink",
    "CRJ-700": "CRJ-700: like the CRJ-200 but longer, with a kinked wing leading edge and larger "
    "rear-mounted engines",
    "ERJ 145": "ERJ 145: slim twin jet with rear-mounted engines and T-tail, long pointed nose, "
    "very long thin fuselage",
    "E-190": "E-190: twin jet with engines under the wings, low tail (no T-tail), rounded nose",
    "Fokker 100": "Fokker 100: twin jet with rear-mounted engines and a T-tail, longer and wider "
    "than a CRJ",
    "Saab 340": "Saab 340: low-wing twin turboprop, small, with a tall tail",
    "DHC-8-300": "Dash 8-300: high-wing twin turboprop with a T-tail and engines in long nacelles",
}
# FGVC-Aircraft test-split row ids (variant label in parentheses is not used by any model).
TASKS = {
    "regional-a-01": (
        "Regional aircraft A",
        [1178, 1520, 2178, 3148, 2326, 1177, 1472, 2302, 2310, 2190, 2050, 2300],
    ),
    "regional-b-02": (
        "Regional aircraft B",
        [2551, 2168, 2301, 2062, 1475, 1147, 1162, 3154, 1502, 3133, 2559, 2539],
    ),
}
TAXONOMY = """# Aircraft types

Each photo shows one aircraft. Label it with exactly one of these types:

""" + "".join(f"- `{name}`\n" for name in TYPES)
ATTRIBUTION = f"""# Image attribution

Photos: FGVC-Aircraft ({PAGE}), test split, via the Hugging Face mirror {DATASET}.
Images are provided by airplane spotters for non-commercial research purposes only; the
photographers retain copyright. Do not redistribute.
"""
_ROWS = "https://datasets-server.huggingface.co/rows"
_INFO = "https://datasets-server.huggingface.co/info"


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def _variants() -> list[str]:
    info = json.loads(_get(f"{_INFO}?dataset={urllib.parse.quote(DATASET)}"))
    return info["dataset_info"]["default"]["features"]["label"]["names"]


def _row(index: int) -> dict:
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": "test", "offset": index, "length": 1}
    )
    return json.loads(_get(f"{_ROWS}?{query}"))["rows"][0]["row"]


def fetch(cache: Path) -> Path:
    """Download the task photos and their variant labels once into `cache`."""
    from PIL import Image

    cache = Path(cache)
    manifest = cache / "manifest.json"
    if manifest.is_file():
        return manifest
    cache.mkdir(parents=True, exist_ok=True)
    variants = _variants()
    rows = {}
    for _task, (_title, ids) in TASKS.items():
        for index in ids:
            row = _row(index)
            picture = Image.open(io.BytesIO(_get(row["image"]["src"]))).convert("RGB")
            picture.thumbnail((1024, 1024))
            picture.save(cache / f"{index}.jpg", "JPEG", quality=90)
            rows[str(index)] = variants[row["label"]]
    manifest.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return manifest


def install(root: Path, manifest: Path) -> dict[str, str]:
    """Write each task's photos, taxonomy and answer key; return task id -> title."""
    labels = json.loads(Path(manifest).read_text(encoding="utf-8"))
    for task_id, (_title, ids) in TASKS.items():
        source = Path(root) / "sources" / task_id
        (source / "images").mkdir(parents=True, exist_ok=True)
        (source / "taxonomy.md").write_text(TAXONOMY, encoding="utf-8")
        truth, origin = {}, {}
        for position, index in enumerate(ids):
            name = f"img-{position + 1:02d}.jpg"
            (source / "images" / name).write_bytes(
                (Path(manifest).parent / f"{index}.jpg").read_bytes()
            )
            truth[name] = [labels[str(index)]]
            origin[name] = index
        answers = Path(root) / "answers" / task_id
        answers.mkdir(parents=True, exist_ok=True)
        (answers / "reference_labels.json").write_text(
            json.dumps(truth, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        (answers / "origin.json").write_text(json.dumps(origin) + "\n", encoding="utf-8")
    (Path(root) / "sources" / "ATTRIBUTION.md").write_text(ATTRIBUTION, encoding="utf-8")
    return {task_id: title for task_id, (title, _ids) in TASKS.items()}


def claim_question(labels: tuple[str, ...]) -> str:
    """The fallback selector's yes/no question for one candidate answer."""
    if len(labels) != 1:
        return "Does `image` show no aircraft at all?"
    return f"Is the aircraft in `image` {article(labels[0])}?"
