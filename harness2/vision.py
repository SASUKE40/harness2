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

"""Jev as the labeler: send each image to Jev and turn its typed answers into detections.

Jev's weights never change; the harness decides what Jev is asked. Its learned prompt rules
go into every question, and each tool's vision() hook can add a label, ask for a posted
speed or a signal color, re-check an uncertain detection on a crop, or filter boxes.

Three modes:
- SVG scenes are rendered to PNG and split into three vertical tiles, one sign per tile.
- Street photos (JPG/PNG) are asked about whole: one yes/no and one Box question per label.
- Aircraft photos get one type question (or one yes/no per type) and a Box for the aircraft.

Images and Box questions are TypeSafe opt-in features ("dreamer-ready"): requests send the
images header, and Box answers need jev-1.14.0. Identical requests are cached, so repeated
rollouts of the same harness cost nothing.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

from . import config
from .core import jev

IMAGES_HEADER = "TypeSafe-Danger-Zone-Images"
TILES = ("left", "middle", "right")
WIDTH, HEIGHT = 160.0, 120.0
TILE = WIDTH / len(TILES)
ROAD_LINE = 0.7
VALUES = ("30", "40", "50", "60", "80")
BASE_CLASSES = {
    "stop": "a red octagon (stop sign)",
    "yield": "an inverted white triangle with a red rim",
    "no-entry": "a solid red disc with a white horizontal bar",
    "speed-limit": "a white disc with a red rim showing a number",
    "warning": "a yellow diamond",
    "other": "a traffic sign that is none of the options above",
    "none": "no traffic sign: only scenery such as sky, sun, moon, buildings, trees or road",
}
PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png")
PHOTO_CLASSES = {
    "stop": "a red octagonal stop sign",
    "speed-limit": "a round speed-limit sign (a white disc with a red rim and a number)",
    "traffic-light": "a traffic light with a lit signal",
}
SPEEDS = ("10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "120")
LIGHTS = ("red", "green")


def _resvg():
    try:
        import resvg_py
    except ImportError as exc:
        raise config.ConfigError("The Jev executor requires: uv pip install -e '.[jev]'") from exc
    return resvg_py


def _pil():
    try:
        from PIL import Image
    except ImportError as exc:
        raise config.ConfigError("The Jev executor requires: uv pip install -e '.[jev]'") from exc
    return Image


def validate() -> None:
    """Check credentials and optional packages before any rollout."""
    config.require("TYPESAFE_API_KEY", needed_by="the Jev executor")
    jev._sdk()
    _resvg()
    _pil()


def settings() -> dict:
    return {
        "model": str(config.env("HARNESS2_JEV_MODEL", "jev-1.14.0")),
        "min_confidence": float(config.env("HARNESS2_JEV_MIN_CONFIDENCE", "0.6")),
    }


def render_tile(svg: str, index: int, width: int = 200) -> bytes:
    """PNG of one vertical third of the scene, cropped through the SVG viewBox."""
    tile = re.sub(
        r'viewBox="[^"]*"', f'viewBox="{index * TILE:.4f} 0 {TILE:.4f} {HEIGHT:g}"', svg, count=1
    )
    return _resvg().svg_to_bytes(svg_string=tile, width=width)


def harness_spec(harness: Path, mode: str = "tiles") -> tuple[dict, list[str]]:
    """What the harness asks Jev: base labels, learned rules, then each tool's vision hook."""
    from .aircraft import PLAIN
    from .image_demo import _read_files, prompt_rules

    files = _read_files(Path(harness))
    classes = {"tiles": BASE_CLASSES, "photo": PHOTO_CLASSES, "aircraft": PLAIN}[mode]
    spec = {
        "mode": mode,
        "classes": dict(classes),
        "rules": [rule.split(" (tool:")[0] for rule in prompt_rules(files)],
        "read_value": False,
        "read_light": False,
        "light_default": None,
        "verify": False,
        "drop_uncertain": False,
        "one_value": False,
        "single_value": False,
        "per_type": False,
        "first_yes": False,
        "max_box_top": None,
    }
    tools = []
    for rel in sorted(files):
        if not rel.startswith("scripts/") or rel == "scripts/label.py":
            continue
        namespace: dict = {"__name__": Path(rel).stem}
        exec(compile(files[rel], rel, "exec"), namespace)
        if callable(namespace.get("vision")):
            namespace["vision"](spec)
            tools.append(Path(rel).stem)
    return spec, tools


def _rules(spec: dict) -> str:
    body = "".join(f"\n- {rule}" for rule in spec["rules"])
    return f"\nRules:{body}" if body else ""


def questions(spec: dict) -> dict:
    """Tile-mode questions: which sign is in each third, and its box."""
    rules = _rules(spec)
    out = {}
    for tile in TILES:
        out[f"{tile}_sign"] = {
            "type": "choice",
            "instructions": (
                f"Which traffic sign is mounted on a pole in `{tile}`? Count only a physical "
                f"sign above the road.{rules}"
            ),
            "criteria": dict(spec["classes"]),
        }
        out[f"{tile}_box"] = {"type": "box", "instructions": f"the traffic sign in `{tile}`"}
        if spec["read_value"]:
            out[f"{tile}_value"] = {
                "type": "choice",
                "instructions": f"Which number is shown on the speed-limit sign in `{tile}`?",
                "criteria": {v: f"the number {v}" for v in VALUES} | {"none": "no number"},
            }
    return out


def _target(label: str) -> str:
    """The noun phrase Jev is asked about for one output label."""
    if label.startswith("speed-") and label[6:].isdigit():
        return f"a speed-limit sign showing {label[6:]}"
    if label.startswith("light-") and label[6:] in LIGHTS:
        return f"a traffic light showing a lit {label[6:]} signal"
    return PHOTO_CLASSES.get(label, label)


def photo_questions(spec: dict) -> dict:
    """Photo-mode questions: a yes/no and a Box for each label the harness looks for."""
    rules = _rules(spec)
    out = {}

    def ask(qid: str, target: str) -> None:
        out[qid] = {"type": "noul", "instructions": f"Is there {target} in `image`?{rules}"}
        out[f"{qid}__box"] = {
            "type": "box",
            "instructions": f"the {target.split(' ', 1)[1]} in `image`",
        }

    for name, target in spec["classes"].items():
        if name == "speed-limit" and spec["read_value"]:
            for value in SPEEDS:
                ask(f"speed_{value}", _target(f"speed-{value}"))
        elif name == "traffic-light" and spec["read_light"]:
            for color in LIGHTS:
                ask(f"light_{color}", _target(f"light-{color}"))
        else:
            ask(name.replace("-", "_"), target)
    return out


def _ask(tiles: dict[str, bytes], asked: dict, model: str) -> dict:
    api_key = config.require("TYPESAFE_API_KEY", needed_by="the Jev executor")
    sdk = jev._sdk()
    kinds = {"box": sdk.Box, "noul": sdk.Noul}
    built = {
        qid: kinds[q["type"]](instructions=q["instructions"])
        if q["type"] in kinds
        else sdk.Choice(instructions=q["instructions"], criteria=q["criteria"])
        for qid, q in asked.items()
    }
    try:
        with sdk.TypeSafeClient(
            api_key=api_key,
            base_url="https://api.typesafe.ai",
            model=model,
            timeout=60.0,
            retry=sdk.RetryPolicy(max_retries=2, timeout=120.0),
        ) as client:
            return client.system_one(
                state={name: sdk.Image(png) for name, png in tiles.items()},
                questions=built,
                extra_headers={IMAGES_HEADER: "true"},
            ).model_dump(mode="json")
    except sdk.TypeSafeError as exc:
        status = getattr(exc, "status_code", None)
        raise RuntimeError(
            f"TypeSafe vision request failed ({type(exc).__name__}"
            + (f", HTTP {status}" if status is not None else "")
            + ")"
        ) from None


def _cached(cache_dir: Path, tiles: dict[str, bytes], asked: dict, model: str) -> dict:
    digest = hashlib.sha256(json.dumps([model, asked], sort_keys=True).encode())
    for name in sorted(tiles):
        digest.update(name.encode() + hashlib.sha256(tiles[name]).digest())
    path = Path(cache_dir) / f"{digest.hexdigest()}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    response = _ask(tiles, asked, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{id(tiles)}.tmp")
    tmp.write_text(json.dumps(response), encoding="utf-8")
    tmp.replace(path)
    return response


def read_answers(answers: dict, spec: dict, threshold: float) -> tuple[list[dict], list[str]]:
    """Turn one tiled image's answers into detections in scene coordinates, plus warnings."""
    found, warnings = [], []
    for index, tile in enumerate(TILES):
        sign = answers.get(f"{tile}_sign") or {}
        choice = sign.get("choice")
        if choice in (None, "none"):
            continue
        if choice == "other":
            warnings.append(f"unlisted sign type in the {tile} third")
            continue
        edges = (answers.get(f"{tile}_box") or {}).get("box") or {}
        top, bottom = float(edges.get("top", 0)), float(edges.get("bottom", 1))
        left, right = float(edges.get("left", 0)), float(edges.get("right", 1))
        if spec["max_box_top"] is not None and top > spec["max_box_top"]:
            continue
        if top > ROAD_LINE:
            warnings.append(f"sign box below the road line in the {tile} third")
        label = choice
        if choice == "speed-limit":
            value = (answers.get(f"{tile}_value") or {}).get("choice")
            if spec["read_value"] and value in VALUES:
                label = f"speed-{value}"
            else:
                label = "speed-unknown"
                warnings.append(f"speed-limit sign without a posted value in the {tile} third")
        confidence = min(
            float(sign.get("confidence") or 0),
            float((answers.get(f"{tile}_box") or {}).get("confidence") or 0),
        )
        if confidence < threshold:
            warnings.append(f"low-confidence {label} ({confidence:.2f}) in the {tile} third")
        x0 = index * TILE
        bbox = [x0 + left * TILE, top * HEIGHT, x0 + right * TILE, bottom * HEIGHT]
        found.append(
            {
                "label": label,
                "bbox": [round(v, 1) for v in bbox],
                "confidence": round(confidence, 3),
            }
        )
    return found, warnings


def _photo_label(qid: str, spec: dict) -> tuple[str, str]:
    """Output label for a yes/no question, plus a warning when the harness cannot say more."""
    if qid == "speed_limit":
        return "speed-unknown", "speed-limit sign without a posted value"
    if qid == "traffic_light":
        if spec["light_default"]:
            return f"light-{spec['light_default']}", ""
        return "light-unknown", "traffic light without a signal color"
    if qid.startswith("speed_"):
        return f"speed-{qid[6:]}", ""
    if qid.startswith("light_"):
        return f"light-{qid[6:]}", ""
    return qid.replace("_", "-"), ""


# Measured limits of the images endpoint: longest side up to about 1568 px; the shortest
# side must exceed 256 px (256 fails, 280 passes).
MAX_SIDE, MIN_SIDE = 1536, 320


def normalize(image: bytes) -> bytes:
    """An image Jev accepts: downscaled to MAX_SIDE, then padded up to MIN_SIDE if needed.

    Images already within the limits are returned unchanged, so cached answers stay valid.
    """
    picture = _pil().open(io.BytesIO(image))
    if MIN_SIDE <= min(picture.size) and max(picture.size) <= MAX_SIDE:
        return image
    picture = picture.convert("RGB")
    picture.thumbnail((MAX_SIDE, MAX_SIDE))
    if min(picture.size) < MIN_SIDE:
        canvas = _pil().new(
            "RGB", (max(picture.width, MIN_SIDE), max(picture.height, MIN_SIDE)), (128, 128, 128)
        )
        canvas.paste(
            picture, ((canvas.width - picture.width) // 2, (canvas.height - picture.height) // 2)
        )
        picture = canvas
    buffer = io.BytesIO()
    picture.save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


def crop(image: bytes, bbox: list[float], size: int = 416) -> bytes:
    """A PNG of the region around a box, with half a box of margin, scaled up for a re-check."""
    picture = _pil().open(io.BytesIO(image)).convert("RGB")
    x0, y0, x1, y1 = bbox
    mx, my = (x1 - x0) / 2, (y1 - y0) / 2
    region = picture.crop(
        (
            max(0, int(x0 - mx)),
            max(0, int(y0 - my)),
            min(picture.width, int(x1 + mx) + 1),
            min(picture.height, int(y1 + my) + 1),
        )
    )
    scale = size / max(region.size)
    region = region.resize(
        (max(1, round(region.width * scale)), max(1, round(region.height * scale)))
    )
    buffer = io.BytesIO()
    region.save(buffer, "PNG")
    return normalize(buffer.getvalue())


def _iou(a: list[float], b: list[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union


def _speed_values(found: list[dict], spec: dict, warnings: list[str]) -> list[dict]:
    """Several values boxed on one sign are a contradiction: report it, or resolve it."""
    speeds = sorted(
        (d for d in found if d["label"].startswith("speed-") and d["label"] != "speed-unknown"),
        key=lambda d: -d["p"],
    )
    others = [d for d in found if d not in speeds]
    if spec["single_value"]:
        return others + speeds[:1]
    kept: list[dict] = []
    stacked = False
    for detection in speeds:
        if any(_iou(detection["bbox"], k["bbox"]) >= 0.4 for k in kept):
            stacked = True
            if spec["one_value"]:
                continue
        kept.append(detection)
    if stacked and not spec["one_value"]:
        warnings.append("several speed values on one sign")
    return others + kept


def read_photo(
    answers: dict,
    spec: dict,
    threshold: float,
    size: tuple[int, int],
    recheck=None,
) -> tuple[list[dict], list[str]]:
    """Turn one photo's answers into pixel-space detections, plus warnings.

    `recheck(label, bbox)` asks Jev again about an uncertain detection on a crop; it is only
    used when the harness has the verify tool.
    """
    width, height = size
    found, warnings = [], []
    for qid, answer in answers.items():
        if qid.endswith("__box") or (answer or {}).get("type") != "noul":
            continue
        yes = float(answer.get("noul") or 0)
        if yes < 0.5:
            continue
        label, warning = _photo_label(qid, spec)
        if warning:
            warnings.append(warning)
        box = answers.get(f"{qid}__box") or {}
        edges = box.get("box") or {}
        confidence = float(box.get("confidence") or 0)
        bbox = [
            float(edges.get("left", 0)) * width,
            float(edges.get("top", 0)) * height,
            float(edges.get("right", 1)) * width,
            float(edges.get("bottom", 1)) * height,
        ]
        detection = {
            "label": label,
            "bbox": [round(v, 1) for v in bbox],
            "confidence": round(confidence, 3),
            "p": round(yes, 3),
        }
        if confidence < threshold:
            if spec["drop_uncertain"]:
                continue
            if spec["verify"] and recheck is not None:
                kept = recheck(label, bbox)
                detection["verified"] = kept
                if not kept:
                    continue
            else:
                warnings.append(f"low-confidence {label} ({confidence:.2f})")
        found.append(detection)
    return _speed_values(found, spec, warnings), warnings


def aircraft_questions(spec: dict) -> dict:
    """One type question (or one yes/no per type with the tool) and a Box for the aircraft."""
    rules = _rules(spec)
    out: dict = {}
    if spec["per_type"] or spec["first_yes"]:
        from .aircraft import article

        for i, target in enumerate(spec["classes"].values()):
            head, _, cue = target.partition(": ")
            phrase = f"{article(head)} ({cue})" if cue else target
            out[f"type_{i}"] = {
                "type": "noul",
                "instructions": f"Is the aircraft in `image` {phrase}?{rules}",
            }
    else:
        out["type"] = {
            "type": "choice",
            "instructions": f"Which aircraft type is shown in `image`?{rules}",
            "criteria": dict(spec["classes"]),
        }
    out["plane__box"] = {"type": "box", "instructions": "the aircraft in `image`"}
    return out


def read_aircraft(
    answers: dict, spec: dict, threshold: float, size: tuple[int, int]
) -> tuple[list[dict], list[str]]:
    names = list(spec["classes"])
    warnings = []
    if "type" in answers:
        answer = answers["type"] or {}
        label = answer.get("choice") or names[0]
        confidence = float(answer.get("confidence") or 0)
        ranked = sorted((answer.get("probabilities") or {}).items(), key=lambda kv: -kv[1])
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.2:
            warnings.append(f"close call between {ranked[0][0]} and {ranked[1][0]}")
    else:
        yes = [float((answers.get(f"type_{i}") or {}).get("noul") or 0) for i in range(len(names))]
        best = max(range(len(names)), key=yes.__getitem__)
        if spec["first_yes"]:
            best = next((i for i, value in enumerate(yes) if value >= 0.5), best)
        label, confidence = names[best], yes[best]
    if confidence < threshold:
        warnings.append(f"uncertain aircraft type ({label}, {confidence:.2f})")
    width, height = size
    edges = (answers.get("plane__box") or {}).get("box") or {}
    bbox = [
        float(edges.get("left", 0)) * width,
        float(edges.get("top", 0)) * height,
        float(edges.get("right", 1)) * width,
        float(edges.get("bottom", 1)) * height,
    ]
    detection = {
        "label": label,
        "bbox": [round(v, 1) for v in bbox],
        "confidence": round(confidence, 3),
    }
    return [detection], warnings


def _photo_size(image: bytes) -> tuple[int, int]:
    return _pil().open(io.BytesIO(image)).size


def run_vision(
    images: Path, harness: Path, run_dir: Path, *, cache_dir: Path, mode: str | None = None
) -> Path:
    """One rollout: the harness shapes the questions, Jev answers, code writes the labels."""
    chosen = settings()
    paths = sorted(
        p for p in Path(images).iterdir() if p.suffix.lower() in (".svg", *PHOTO_SUFFIXES)
    )
    mode = mode or ("tiles" if all(p.suffix.lower() == ".svg" for p in paths) else "photo")
    spec, tools = harness_spec(harness, mode)
    asked = {"tiles": questions, "photo": photo_questions, "aircraft": aircraft_questions}[mode](
        spec
    )
    lines = [
        f"[ACTION] Jev vision labeling with {chosen['model']} ({mode} mode): "
        f"{len(asked)} questions per image",
        "[OBSERVATION]",
        *(f"[tool] {name}" for name in tools),
    ]
    labels, detections = {}, {}
    for path in paths:
        if mode == "tiles":
            svg = path.read_text(encoding="utf-8")
            tiles = {tile: render_tile(svg, i) for i, tile in enumerate(TILES)}
            response = _cached(cache_dir, tiles, asked, chosen["model"])
            found, warnings = read_answers(
                response.get("answers") or {}, spec, chosen["min_confidence"]
            )
        elif mode == "aircraft":
            image = normalize(path.read_bytes())
            response = _cached(cache_dir, {"image": image}, asked, chosen["model"])
            found, warnings = read_aircraft(
                response.get("answers") or {}, spec, chosen["min_confidence"], _photo_size(image)
            )
        else:
            image = normalize(path.read_bytes())

            def recheck(label: str, bbox: list[float], image: bytes = image) -> bool:
                asked_crop = {
                    "check": {
                        "type": "noul",
                        "instructions": f"Does `crop` show {_target(label)}?",
                    }
                }
                reply = _cached(cache_dir, {"crop": crop(image, bbox)}, asked_crop, chosen["model"])
                return (
                    float(((reply.get("answers") or {}).get("check") or {}).get("noul") or 0) >= 0.5
                )

            response = _cached(cache_dir, {"image": image}, asked, chosen["model"])
            found, warnings = read_photo(
                response.get("answers") or {},
                spec,
                chosen["min_confidence"],
                _photo_size(image),
                recheck,
            )
        # A photo answer names each label once; tiles can repeat a label.
        names = [d["label"] for d in found]
        labels[path.name] = sorted(names if mode == "tiles" else set(names))
        detections[path.name] = found
        lines.append(f"{path.name}: {labels[path.name]}")
        lines += [f"[warn] {path.name}: {w}" for w in warnings]
    run_dir = Path(run_dir)
    (run_dir / "labels.json").write_text(
        json.dumps(labels, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "detections.json").write_text(
        json.dumps(detections, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "trajectory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir
