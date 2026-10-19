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

"""Real street photos for the traffic-sign stream, with their published annotations.

Source: the validation split of Sign_Road_Detection_Dataset on Hugging Face (a Roboflow
export with YOLO boxes, CC BY 4.0). The 20 images below are phone photos of streets from its
`road*` subset; watermarked stock photos and simulator frames were left out. Class ids follow
Roboflow's alphabetical order, which was checked against the images. The annotations cover
only these classes, so other sign types in a photo are not labels.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

DATASET = "ayoubsa/Sign_Road_Detection_Dataset"
URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/valid.zip"
PAGE = f"https://huggingface.co/datasets/{DATASET}"
LICENSE = "CC BY 4.0"
CLASSES = [
    "light-green",
    "light-red",
    "speed-10",
    "speed-100",
    "speed-110",
    "speed-120",
    "speed-20",
    "speed-30",
    "speed-40",
    "speed-50",
    "speed-60",
    "speed-70",
    "speed-80",
    "speed-90",
    "stop",
]
_SUFFIX = "_png.rf."
# Two harder sets of ten. The starting harness gets 6 of 10 right in each; the remaining
# errors are Jev's own mistakes on real photos, not missing capabilities.
TASKS = {
    "speed-limits-01": (
        "Speed-limit signs",
        [
            "road551_png.rf.85136b3b7cb2a56ce38e6ac16d1d9ac0",
            "road602_png.rf.318d3ef4a1ae690772af4c7a45056e3d",
            "road627_png.rf.c7b56d6b9d184e7eda577bc92098970a",
            "road642_png.rf.25d90a69f2f25043d4816446bc06d57d",
            "road695_png.rf.aa31ed894fa0b3796c4865db5e00ebac",
            "road789_png.rf.e0563b696bba55c1e0ec034219c9b2e2",
            "road247_png.rf.fcf9d60e9599271c2725b9cc16153d8b",
            "road246_png.rf.e0ea3c61697e875432eab5f80f8f0b77",
            "road510_png.rf.c526c717645235b0ba526972f91dcd8c",
            "road769_png.rf.b448ee15665aa9198508c0007ecfff13",
        ],
    ),
    "intersections-02": (
        "Intersections",
        [
            "road57_png.rf.36dd69f2bf2bb71fae4bb5f71976de87",
            "road5_png.rf.14d45e774c14dff080bb5d8977b83b16",
            "road819_png.rf.102970997e42717b71177acc1f71f554",
            "road24_png.rf.dffe5ca5e68e8016570446d1b29c0425",
            "road176_png.rf.bbb9ca924014ecf54c43d0813e953a37",
            "road8_png.rf.46c42ca2cb0550a9319955aba6abede9",
            "road845_png.rf.a17d514f4ef79850eee0d0564166493c",
            "road195_png.rf.da6efe0c944836e494ef731b0c83f40e",
            "road820_png.rf.84c419beb4bfd13b6e9363920f97e752",
            "road825_png.rf.edd11b214c98691900d7161d1d445086",
        ],
    ),
}
TAXONOMY = """# Sign taxonomy (street photos)

Label each kind of sign or signal visible in the photo once, using exactly these labels:

- `stop`: red octagonal stop sign
- `speed-<n>`: round speed-limit sign showing `<n>` (10 to 120); two different limits are
  two labels
- `light-red`, `light-green`: a traffic light showing that lit signal

Other sign types (warning triangles, crossings, direction or parking signs) are not labels.
"""
ATTRIBUTION = f"""# Image attribution

Photos and annotations: {DATASET} ({PAGE}), validation split, licensed {LICENSE}.
Images were exported by Roboflow at 416x416. File names map to the original export.
"""


def fetch(cache: Path) -> Path:
    """The dataset's validation archive, downloaded once into `cache`."""
    path = Path(cache) / "valid.zip"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".zip.part")
        with urllib.request.urlopen(URL, timeout=120) as response, tmp.open("wb") as out:
            out.write(response.read())
        tmp.replace(path)
    return path


def _boxes(text: str, width: int, height: int) -> list[dict]:
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:])
        boxes.append(
            {
                "label": CLASSES[int(parts[0])],
                "bbox": [
                    round((cx - w / 2) * width, 1),
                    round((cy - h / 2) * height, 1),
                    round((cx + w / 2) * width, 1),
                    round((cy + h / 2) * height, 1),
                ],
            }
        )
    return boxes


def install(root: Path, archive: Path) -> dict[str, str]:
    """Write each task's photos, taxonomy and answer key; return task id -> title."""
    from PIL import Image

    with zipfile.ZipFile(archive) as bundle:
        names = {Path(n).name: n for n in bundle.namelist()}
        for task_id, (_title, stems) in TASKS.items():
            source = Path(root) / "sources" / task_id
            (source / "images").mkdir(parents=True, exist_ok=True)
            (source / "taxonomy.md").write_text(TAXONOMY, encoding="utf-8")
            truth, boxes, origin = {}, {}, {}
            for index, stem in enumerate(stems):
                data = bundle.read(names[f"{stem}.jpg"])
                width, height = Image.open(io.BytesIO(data)).size
                name = f"img-{index + 1:02d}.jpg"
                (source / "images" / name).write_bytes(data)
                found = _boxes(bundle.read(names[f"{stem}.txt"]).decode(), width, height)
                truth[name] = sorted({b["label"] for b in found})
                boxes[name] = found
                origin[name] = stem.split(_SUFFIX)[0]
            answers = Path(root) / "answers" / task_id
            answers.mkdir(parents=True, exist_ok=True)
            (answers / "reference_labels.json").write_text(
                json.dumps(truth, indent=1, sort_keys=True) + "\n", encoding="utf-8"
            )
            (answers / "boxes.json").write_text(json.dumps(boxes) + "\n", encoding="utf-8")
            (answers / "origin.json").write_text(json.dumps(origin) + "\n", encoding="utf-8")
    (Path(root) / "sources" / "ATTRIBUTION.md").write_text(ATTRIBUTION, encoding="utf-8")
    return {task_id: title for task_id, (title, _stems) in TASKS.items()}
