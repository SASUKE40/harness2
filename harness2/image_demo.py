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

"""A deterministic traffic-sign labeling workload for the real recursion and Jev selection.

The harness the recursion edits has three kinds of components, mirroring what Harness²
changes around a frozen model: the system prompt (learned rules), tools (one script per
capability under scripts/, loaded by the labeler), and the experience library. The base
labeler has five perception gaps. The scripted editor reads only what a model editor
would: the trajectories' warnings, the candidate harnesses, and the library. Some probes
are plausible but wrong, so the final selection has a real choice to make. Reference
labels never enter any context.

Images are SVG scenes. The default executor parses them with the harness's labeler; the
`jev` executor (harness2.vision) renders them to PNG and asks Jev to see them instead.
"""

from __future__ import annotations

import difflib
import json
import os
import random
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .core import jev
from .core import tree as harness_tree
from .core.proposer import Proposer, Reply
from .core.spec import BenchPrompts, BenchSpec, ChannelSpec
from .substrate.opencode import OpencodeSubstrate

DOMAIN = "traffic-signs"
TAXONOMY = """# Sign taxonomy

Label every physical traffic sign in each image. Use exactly these labels:

- `stop`: red octagon
- `yield`: inverted white triangle with a red rim
- `no-entry`: solid red disc with a white bar
- `speed-<n>`: white disc with a red rim showing the posted limit `<n>`
- `warning`: yellow diamond
- `school`: fluorescent yellow-green pentagon

Reflections on the road are not signs. A reflection is drawn with opacity below 0.5.
Faded but physical signs still count. The sun and the moon are not signs.
"""

# id -> (title, features). Every task shows each tempting wrong rule's failure case.
TASKS = {
    "downtown-01": ("Downtown at night", ("groups", "speed", "warning", "reflections")),
    "rainy-highway-02": ("Rainy highway", ("groups", "speed", "warning", "reflections")),
    "suburban-03": ("Suburban dusk", ("groups", "speed", "warning", "reflections")),
    "school-zone-04": ("School zone", ("groups", "speed", "warning", "reflections", "school")),
}
IMAGES_PER_TASK = 6
SCHOOL = "#9d3"


@dataclass(frozen=True)
class Fix:
    title: str
    kind: str
    slot: str
    lesson: str
    tool: str = ""
    trigger: str = ""
    warning: str = ""
    vision_warning: str = ""
    good: bool = True
    alternative: str = ""
    install: str = ""
    vision: str = ""
    guardrail: str = ""


LABELER = r"""import importlib.util
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://www.w3.org/2000/svg}"
RED, WHITE, YELLOW = "#d00", "#fff", "#fc0"
WARNINGS = []
FOUND = []
CURRENT = [None, 0.0, 0.0]


def num(el, name):
    return float(el.get(name, 0))


def box(el, dx, dy):
    tag = el.tag[len(NS) :]
    if tag == "circle":
        cx, cy, r = num(el, "cx") + dx, num(el, "cy") + dy, num(el, "r")
        return [cx - r, cy - r, cx + r, cy + r]
    if tag == "polygon":
        pts = [p.split(",") for p in el.get("points", "").split()]
        xs, ys = [float(x) + dx for x, _ in pts], [float(y) + dy for _, y in pts]
        return [min(xs), min(ys), max(xs), max(ys)]
    x, y, w, h = num(el, "x") + dx, num(el, "y") + dy, num(el, "width"), num(el, "height")
    if "rotate" in el.get("transform", ""):
        cx, cy, half = x + w / 2, y + h / 2, w * 0.7071
        return [cx - half, cy - half, cx + half, cy + half]
    return [x, y, x + w, y + h]


class Labels(list):
    def append(self, label):
        super().append(label)
        FOUND.append({"label": label, "bbox": [round(v, 1) for v in box(*CURRENT)]})


def elements(root):
    for el in root:
        if el.tag == NS + "g":
            WARNINGS.append(f"nested <g> ignored ({len(list(el.iter())) - 1} elements)")
            continue
        yield el, 0.0, 0.0


def skip(el, tag):
    if float(el.get("opacity", 1)) < 0.5:
        WARNINGS.append(f"low-opacity {tag} counted as a sign")
    return False


def special(el, tag, fill, labels):
    return False


def polygon(el, fill, stroke, sides, labels):
    if sides == 8 and fill == RED:
        labels.append("stop")
    elif sides == 3 and stroke == RED:
        labels.append("yield")
    elif sides == 4 and fill == YELLOW:
        labels.append("warning")
    elif sides == 5:
        WARNINGS.append(f"unrecognized 5-sided polygon (fill {fill})")


def circle(el, fill, stroke, digits, labels):
    if fill == RED or stroke == RED:
        if digits:
            WARNINGS.append(f"circle with numeral text {digits[0]!r} labeled no-entry")
        labels.append("no-entry")


def rect(el, fill, labels):
    WARNINGS.append(f"rotated rect ignored (fill {fill})")


# Tools in this directory extend these hooks; see install() in each tool.
HOOKS = {
    "elements": elements,
    "skip": skip,
    "special": special,
    "polygon": polygon,
    "circle": circle,
    "rect": rect,
}


def classify(svg):
    items = list(HOOKS["elements"](ET.fromstring(svg)))
    texts = [
        (num(el, "x") + dx, num(el, "y") + dy, (el.text or "").strip())
        for el, dx, dy in items
        if el.tag == NS + "text"
    ]
    labels = Labels()
    for el, dx, dy in items:
        CURRENT[:] = [el, dx, dy]
        tag, fill, stroke = el.tag[len(NS) :], el.get("fill", ""), el.get("stroke", "")
        if HOOKS["skip"](el, tag) or HOOKS["special"](el, tag, fill, labels):
            continue
        if tag == "polygon":
            HOOKS["polygon"](el, fill, stroke, len(el.get("points", "").split()), labels)
        elif tag == "circle":
            cx, cy, r = num(el, "cx") + dx, num(el, "cy") + dy, num(el, "r")
            near = [s for x, y, s in texts if abs(x - cx) <= r and abs(y - cy) <= r]
            HOOKS["circle"](el, fill, stroke, [s for s in near if s.isdigit()], labels)
        elif tag == "rect" and el.get("transform", "").startswith("rotate"):
            HOOKS["rect"](el, fill, labels)
    return sorted(labels)


def load_tools(folder):
    loaded = []
    for path in sorted(Path(folder).glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "install"):
            module.install(HOOKS, globals())
            loaded.append(path.stem)
    return loaded


def main():
    images, out = Path(sys.argv[1]), Path(sys.argv[2])
    for name in load_tools(Path(__file__).parent):
        print(f"[tool] {name}")
    labels, detections = {}, {}
    for path in sorted(images.glob("*.svg")):
        WARNINGS.clear()
        FOUND.clear()
        labels[path.name] = classify(path.read_text())
        detections[path.name] = list(FOUND)
        print(f"{path.name}: {labels[path.name]}")
        for warning in WARNINGS:
            print(f"[warn] {path.name}: {warning}")
    out.write_text(json.dumps(labels, indent=1, sort_keys=True) + "\n")
    out.with_name("detections.json").write_text(json.dumps(detections, sort_keys=True) + "\n")
    print(f"wrote labels for {len(labels)} images to {out.name}")


if __name__ == "__main__":
    main()
"""


def _tool_source(fix: Fix) -> str:
    """One tool file: install() extends the SVG labeler, vision() shapes Jev requests."""
    return (
        f'"""Tool: {fix.title}.\n\n{fix.lesson}\n"""\n\n\n'
        f"def install(hooks, api):\n{fix.install}\n\n"
        f"def vision(spec):\n{(fix.vision or '    pass').rstrip()}\n"
    )


def _cues_vision() -> str:
    from .aircraft import CUES

    lines = "".join(f"            {name!r}: {text!r},\n" for name, text in CUES.items())
    return (
        '    if spec["mode"] == "aircraft":\n'
        '        spec["classes"].update(\n'
        "            {\n" + lines.replace("            ", "                ") + "            }\n"
        "        )\n"
    )


_CUES_VISION = _cues_vision()

FIXES: dict[str, Fix] = {
    "groups": Fix(
        title="Recurse into grouped signs",
        kind="INV",
        slot="groups",
        tool="recurse_offset",
        trigger="nested_group",
        warning="nested <g> ignored",
        lesson="Signs can sit inside <g transform=translate> groups; recurse and add the offset.",
        install="""    import re

    ns = api["NS"]

    def elements(root, dx=0.0, dy=0.0):
        for el in root:
            if el.tag == ns + "g":
                m = re.search(r"translate\\(([-\\d.]+)[ ,]+([-\\d.]+)\\)", el.get("transform", ""))
                ox, oy = (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)
                yield from elements(el, dx + ox, dy + oy)
                continue
            yield el, dx, dy

    hooks["elements"] = elements
""",
    ),
    "speed": Fix(
        title="Read speed-limit numerals",
        kind="DOM",
        slot="circle",
        tool="read_speed_limit",
        trigger="numeral_disc",
        warning="circle with numeral text",
        vision_warning="speed-limit sign without a posted value",
        lesson="A white disc with a red rim and a numeral is speed-<n>; a solid red disc is no-entry.",
        install="""    prev = hooks["circle"]

    def circle(el, fill, stroke, digits, labels):
        if stroke == api["RED"] and fill == api["WHITE"] and digits:
            labels.append(f"speed-{digits[0]}")
            return
        prev(el, fill, stroke, digits, labels)

    hooks["circle"] = circle
""",
        vision="""    spec["read_value"] = True
""",
    ),
    "speed_bad": Fix(
        title="Every red-rimmed disc is a speed limit",
        kind="DOM",
        slot="circle",
        tool="speed_unknown",
        trigger="red_rim",
        good=False,
        alternative="speed",
        lesson="Any red-rimmed disc is a speed limit; use speed-unknown when no numeral is visible.",
        install="""    prev = hooks["circle"]

    def circle(el, fill, stroke, digits, labels):
        if stroke == api["RED"]:
            labels.append("speed-" + (digits[0] if digits else "unknown"))
            return
        prev(el, fill, stroke, digits, labels)

    hooks["circle"] = circle
""",
        vision="""    spec["classes"]["speed-limit"] = "any disc with a red rim, including solid red discs"
""",
    ),
    "warning": Fix(
        title="Rotated-square warning diamonds",
        kind="DOM",
        slot="warning",
        tool="warning_diamond",
        trigger="rotated_square",
        warning="rotated rect ignored",
        lesson="Warning diamonds may be drawn as yellow squares rotated 45 degrees.",
        install="""    prev = hooks["rect"]

    def rect(el, fill, labels):
        if fill == api["YELLOW"] and "rotate(45" in el.get("transform", ""):
            labels.append("warning")
            return
        prev(el, fill, labels)

    hooks["rect"] = rect
""",
        vision="""    if "warning" in spec["classes"]:
        spec["classes"]["warning"] += ", including a yellow square turned 45 degrees"
""",
    ),
    "warning_bad": Fix(
        title="Every yellow shape is a warning",
        kind="DOM",
        slot="warning",
        tool="warning_any",
        trigger="yellow_shape",
        good=False,
        alternative="warning",
        lesson="Treat every yellow shape as a warning sign.",
        install="""    def special(el, tag, fill, labels):
        if fill == api["YELLOW"]:
            labels.append("warning")
            return True
        return False

    hooks["special"] = special
""",
        vision="""    if spec["mode"] == "tiles":
        spec["classes"]["warning"] = "any yellow shape, including round ones"
""",
    ),
    "reflections": Fix(
        title="Skip road reflections",
        kind="DOM",
        slot="opacity",
        tool="skip_reflection",
        trigger="low_opacity",
        warning="low-opacity",
        vision_warning="sign box below the road line",
        lesson="Shapes with opacity below 0.5 are reflections, not signs; skip them.",
        install="""    prev = hooks["skip"]

    def skip(el, tag):
        if float(el.get("opacity", 1)) < 0.5:
            return True
        return prev(el, tag)

    hooks["skip"] = skip
""",
        vision="""    spec["rules"].append("A sign mirrored in a puddle or on wet road is a reflection, not a sign.")
    if spec["mode"] == "tiles":
        spec["max_box_top"] = 0.7
""",
    ),
    "reflections_bad": Fix(
        title="Skip anything with an opacity attribute",
        kind="DOM",
        slot="opacity",
        tool="skip_any_opacity",
        trigger="opacity_attr",
        good=False,
        alternative="reflections",
        lesson="Ignore any element that sets an opacity attribute.",
        install="""    prev = hooks["skip"]

    def skip(el, tag):
        return bool(el.get("opacity")) or prev(el, tag)

    hooks["skip"] = skip
""",
        vision="""    spec["rules"].append("Ignore signs that look faded or partly transparent.")
    if spec["mode"] == "tiles":
        spec["max_box_top"] = 0.45
""",
    ),
    "school": Fix(
        title="School-zone pentagons",
        kind="DOM",
        slot="pentagon",
        tool="school_sign",
        trigger="five_sided",
        warning="unrecognized 5-sided polygon",
        vision_warning="unlisted sign type",
        lesson="A fluorescent yellow-green pentagon is a school sign; label it school.",
        install=f"""    prev = hooks["polygon"]

    def polygon(el, fill, stroke, sides, labels):
        if sides == 5 and fill == "{SCHOOL}":
            labels.append("school")
            return
        prev(el, fill, stroke, sides, labels)

    hooks["polygon"] = polygon
""",
        vision="""    if spec["mode"] == "tiles":
        spec["classes"]["school"] = "a fluorescent yellow-green pentagon (school zone)"
""",
    ),
    "school_bad": Fix(
        title="Every pentagon is a warning",
        kind="DOM",
        slot="pentagon",
        tool="pentagon_warning",
        trigger="five_sided",
        good=False,
        alternative="school",
        lesson="Treat any five-sided sign as a warning sign.",
        install="""    prev = hooks["polygon"]

    def polygon(el, fill, stroke, sides, labels):
        if sides == 5:
            labels.append("warning")
            return
        prev(el, fill, stroke, sides, labels)

    hooks["polygon"] = polygon
""",
        vision="""    if "warning" in spec["classes"]:
        spec["classes"]["warning"] += " or any five-sided sign"
""",
    ),
    "lights": Fix(
        title="Read traffic-light colors",
        kind="DOM",
        slot="light",
        tool="read_light_color",
        trigger="unlit_light",
        vision_warning="traffic light without a signal color",
        lesson="Ask which signal a traffic light shows; label it light-red or light-green.",
        install="    pass\n",
        vision="""    spec["read_light"] = True
""",
    ),
    "lights_bad": Fix(
        title="Every traffic light is red",
        kind="DOM",
        slot="light",
        tool="lights_red",
        trigger="unlit_light",
        good=False,
        alternative="lights",
        lesson="Treat every traffic light as showing red.",
        install="    pass\n",
        vision="""    spec["light_default"] = "red"
""",
    ),
    "dedupe": Fix(
        title="Keep one speed value per sign",
        kind="DOM",
        slot="speedvalues",
        tool="one_value_per_sign",
        trigger="stacked_values",
        vision_warning="several speed values on one sign",
        lesson="When several speed values overlap on one sign, keep only the most likely value.",
        install="    pass\n",
        vision="""    spec["one_value"] = True
""",
    ),
    "dedupe_bad": Fix(
        title="Keep one speed value per photo",
        kind="DOM",
        slot="speedvalues",
        tool="single_speed_value",
        trigger="stacked_values",
        good=False,
        alternative="dedupe",
        lesson="Report at most one speed limit per photo.",
        install="    pass\n",
        vision="""    spec["single_value"] = True
""",
    ),
    "cues": Fix(
        title="Describe each aircraft type by spotter cues",
        kind="DOM",
        slot="typecues",
        tool="spotter_cues",
        trigger="uncertain_type",
        vision_warning="uncertain aircraft type",
        lesson="Describe each aircraft type by the cues spotters use: wing and engine position, "
        "tail shape and fuselage length.",
        install="    pass\n",
        vision=_CUES_VISION,
    ),
    "cues_bad": Fix(
        title="Tell aircraft apart by airline livery",
        kind="DOM",
        slot="typecues",
        tool="livery_hints",
        trigger="uncertain_type",
        good=False,
        alternative="cues",
        lesson="Use the airline's livery and colors to tell aircraft types apart.",
        install="    pass\n",
        vision="""    spec["rules"].append("Use the airline's livery and colors to tell the types apart.")
""",
    ),
    "per_type": Fix(
        title="Ask about each aircraft type separately",
        kind="DOM",
        slot="typequestion",
        tool="ask_each_type",
        trigger="close_call",
        vision_warning="close call between",
        lesson="When two types are close, ask a yes/no question for each type and keep the "
        "most likely one.",
        install="    pass\n",
        vision="""    spec["per_type"] = True
""",
    ),
    "per_type_bad": Fix(
        title="Take the first type Jev says yes to",
        kind="DOM",
        slot="typequestion",
        tool="first_yes_type",
        trigger="close_call",
        good=False,
        alternative="per_type",
        lesson="Ask about each type in order and stop at the first yes.",
        install="    pass\n",
        vision="""    spec["first_yes"] = True
""",
    ),
    "verify": Fix(
        title="Re-check uncertain detections on a crop",
        kind="DOM",
        slot="verify",
        tool="verify_crop",
        trigger="low_confidence",
        vision_warning="low-confidence",
        lesson="Crop around an uncertain detection and ask again before keeping it.",
        install="    pass\n",
        vision="""    spec["verify"] = True
""",
    ),
    "verify_bad": Fix(
        title="Drop every uncertain detection",
        kind="DOM",
        slot="verify",
        tool="drop_uncertain",
        trigger="low_confidence",
        good=False,
        alternative="verify",
        lesson="Discard any detection below the confidence threshold.",
        install="    pass\n",
        vision="""    spec["drop_uncertain"] = True
""",
    ),
    "complete": Fix(
        title="Deliver only complete label files",
        kind="INV",
        slot="complete",
        trigger="partial_output",
        lesson="Deliver labels.json only after every image under images/ has an entry.",
        guardrail="- Deliver labels.json only after every image under images/ has an entry.",
    ),
}
GOOD = frozenset(name for name, fix in FIXES.items() if fix.good and fix.tool)
ORDER = list(FIXES)
TOOLS = {f"scripts/{fix.tool}.py": name for name, fix in FIXES.items() if fix.tool}

BASE_PROMPT = (
    "You label traffic signs in street-scene images.\n"
    "Run `python scripts/label.py images labels.json` from the task directory, check the\n"
    "result against taxonomy.md, and deliver labels.json. label.py loads every tool in\n"
    "scripts/ automatically.\n"
)
RULES_HEADER = "\n## Learned rules\n\n"
BASE_GUARDRAILS = "# Guardrails\n\n- Use only the labels in taxonomy.md.\n"


def render_tree(base: dict[str, str], fixes) -> dict[str, str]:
    """The harness for a fix set: one tool file per fix and a prompt rule for each."""
    files = {rel: body for rel, body in base.items() if rel not in TOOLS}
    files["scripts/label.py"] = LABELER
    ordered = [f for f in sorted(fixes, key=ORDER.index) if FIXES[f].tool]
    for name in ordered:
        files[f"scripts/{FIXES[name].tool}.py"] = _tool_source(FIXES[name])
    files["systemprompt.md"] = BASE_PROMPT + (
        RULES_HEADER
        + "".join(f"- {FIXES[f].lesson} (tool: scripts/{FIXES[f].tool}.py)\n" for f in ordered)
        if ordered
        else ""
    )
    files["guardrails.md"] = BASE_GUARDRAILS + (
        FIXES["complete"].guardrail + "\n" if "complete" in fixes else ""
    )
    return files


def fixes_of(files: dict[str, str]) -> set[str]:
    """Recover the applied fixes from the harness: its tool files and guardrails."""
    found = {name for rel, name in TOOLS.items() if rel in files}
    if FIXES["complete"].guardrail in files.get("guardrails.md", ""):
        found.add("complete")
    return found


def prompt_rules(files: dict[str, str]) -> list[str]:
    prompt = files.get("systemprompt.md", "")
    if RULES_HEADER not in prompt:
        return []
    body = prompt.split(RULES_HEADER, 1)[1]
    return [line[2:] for line in body.splitlines() if line.startswith("- ")]


def _read_files(directory: Path) -> dict[str, str]:
    paths = [directory / "guardrails.md", directory / "systemprompt.md"]
    paths += sorted((directory / "scripts").glob("*.py"))
    return {
        p.relative_to(directory).as_posix(): p.read_text(encoding="utf-8")
        for p in paths
        if p.is_file()
    }


def build_labeler(files: dict[str, str]) -> dict:
    """Load a harness's labeler and tools in-process, exactly as label.py's main() does."""
    ns: dict = {"__name__": "labeler", "__file__": "label.py"}
    exec(compile(files["scripts/label.py"], "label.py", "exec"), ns)
    for rel in sorted(files):
        if rel.startswith("scripts/") and rel != "scripts/label.py" and rel.endswith(".py"):
            tool: dict = {"__name__": Path(rel).stem}
            exec(compile(files[rel], rel, "exec"), tool)
            tool["install"](ns["HOOKS"], ns)
    return ns


_PERFECT: dict = {}


def _perfect() -> dict:
    if not _PERFECT:
        _PERFECT.update(build_labeler(render_tree({}, GOOD)))
    _PERFECT["FOUND"].clear()
    return _PERFECT


def perceive(svg: str) -> list[str]:
    """Labels from the fully corrected labeler: the deterministic stand-ins' perception."""
    return _perfect()["classify"](svg)


def perceive_boxes(svg: str) -> list[dict]:
    """Sign boxes from the fully corrected labeler, used only to annotate exported runs."""
    ns = _perfect()
    ns["classify"](svg)
    return list(ns["FOUND"])


# ---------------------------------------------------------------------------------------
# Scenes


def _octagon(cx: float, cy: float, r: float) -> str:
    pts = [(-0.41, -1), (0.41, -1), (1, -0.41), (1, 0.41), (0.41, 1), (-0.41, 1), (-1, 0.41)]
    pts.append((-1, -0.41))
    return " ".join(f"{cx + x * r:g},{cy + y * r:g}" for x, y in pts)


def _sign(kind: str, cx: float, cy: float, *, value: int = 0, rotated: bool = False, o: str = ""):
    text = 'text-anchor="middle" font-family="sans-serif"'
    if kind == "stop":
        return [
            f'<polygon points="{_octagon(cx, cy, 11)}" fill="#d00" stroke="#fff"{o}/>',
            f'<text x="{cx}" y="{cy + 2}" font-size="5" fill="#fff" {text}>STOP</text>',
        ]
    if kind == "yield":
        return [
            f'<polygon points="{cx - 11},{cy - 8} {cx + 11},{cy - 8} {cx},{cy + 11}" '
            f'fill="#fff" stroke="#d00" stroke-width="3"{o}/>'
        ]
    if kind == "no-entry":
        return [
            f'<circle cx="{cx}" cy="{cy}" r="10" fill="#d00" stroke="#d00"{o}/>',
            f'<rect x="{cx - 6}" y="{cy - 1.5}" width="12" height="3" fill="#fff"/>',
        ]
    if kind == "speed":
        return [
            f'<circle cx="{cx}" cy="{cy}" r="10" fill="#fff" stroke="#d00" stroke-width="3"{o}/>',
            f'<text x="{cx}" y="{cy + 3}" font-size="8" fill="#111" {text}>{value}</text>',
        ]
    if kind == "school":
        return [
            f'<polygon points="{cx},{cy - 11} {cx + 10},{cy - 4} {cx + 10},{cy + 10} '
            f'{cx - 10},{cy + 10} {cx - 10},{cy - 4}" fill="{SCHOOL}" stroke="#222"{o}/>',
            f'<text x="{cx}" y="{cy + 5}" font-size="3.6" fill="#222" {text}>SCHOOL</text>',
        ]
    shape = (
        f'<rect x="{cx - 7}" y="{cy - 7}" width="14" height="14" fill="#fc0" stroke="#222" '
        f'transform="rotate(45 {cx} {cy})"{o}/>'
        if rotated
        else f'<polygon points="{cx},{cy - 10} {cx + 10},{cy} {cx},{cy + 10} {cx - 10},{cy}" '
        f'fill="#fc0" stroke="#222"{o}/>'
    )
    return [shape, f'<text x="{cx}" y="{cy + 3}" font-size="9" fill="#222" {text}>!</text>']


# Scenery is decoration only. It uses <defs>/<use>, <path>, <ellipse>, plain rects and
# fill-opacity, never a polygon, rotated rect, sign color, `opacity` or <g>, so no scenery
# element can match a labeler rule or produce a labeler warning.
_COMMON_DEFS = (
    '<radialGradient id="v"><stop offset=".62" stop-opacity="0"/>'
    '<stop offset="1" stop-opacity=".42"/></radialGradient>'
    '<filter id="b" x="-1" y="-1" width="3" height="3"><feGaussianBlur stdDeviation="1.4"/></filter>'
)


def _grad(name: str, *stops: str) -> str:
    n = len(stops) - 1
    return (
        f'<linearGradient id="{name}" x2="0" y2="1">'
        + "".join(f'<stop offset="{i / n:.2g}" stop-color="{c}"/>' for i, c in enumerate(stops))
        + "</linearGradient>"
    )


def _skyline(rng: random.Random, base: int, low: int, high: int) -> str:
    d, x = f"M0 {base}", 0
    while x < 160:
        d += f"V{rng.randint(low, high)}h{rng.randint(12, 22)}"
        x = int(d.rsplit("h", 1)[1]) + x
    return d + f"V{base}z"


def _night(rng: random.Random) -> tuple[str, list[str], list[str]]:
    defs = (
        _grad("s", "#060a1d", "#262a5c", "#5a3a6e")
        + _grad("r", "#1d1f2c", "#0a0b11")
        + '<pattern id="w" width="5" height="6" patternUnits="userSpaceOnUse">'
        '<rect x="1" y="1.2" width="2" height="2.6" fill="#ffcf6e" fill-opacity=".8"/></pattern>'
        '<symbol id="c" width="30" height="12" viewBox="0 0 30 12">'
        '<path d="M2 9 3 5 8 4 11 1h9l4 3 4 1 1 4z" fill="#cfd6e6"/>'
        '<path d="M12 2h7l3 3H10z" fill="#4c5d82"/>'
        '<ellipse cx="8" cy="10" rx="2" ry="2" fill="#111"/>'
        '<ellipse cx="23" cy="10" rx="2" ry="2" fill="#111"/></symbol>'
    )
    city = _skyline(rng, 86, 34, 62)
    hx = rng.choice([10, 48, 100])
    car = rng.choice([4, 44, 96])
    stars = "".join(f"M{rng.randint(2, 158)} {rng.randint(2, 30)}h.1" for _ in range(10))
    streaks = "".join(f"M{rng.randint(5, 150)} 86h3v30h-3z" for _ in range(3))
    back = [
        '<rect width="160" height="86" fill="url(#s)"/>',
        f'<path d="{stars}" stroke="#fffbe6" stroke-width=".9" stroke-linecap="round"/>',
        f'<path d="{city}" fill="#121733"/>',
        f'<path d="{city}" fill="url(#w)"/>',
        f'<text x="{hx}" y="47" font-size="6" font-family="sans-serif" fill="#ffd1f4" '
        'stroke="#ff5fd2" stroke-width=".35">HOTEL</text>',
        '<rect y="84" width="160" height="36" fill="url(#r)"/>',
        f'<path d="{streaks}" fill="#ffcf6e" fill-opacity=".22" filter="url(#b)"/>',
        '<path d="M0 102H160" stroke="#c9ced8" stroke-width="1.2" stroke-dasharray="9 7"/>',
        f'<ellipse cx="{car + 34}" cy="100.5" rx="9" ry="2.5" fill="#fff3b0" fill-opacity=".45" '
        'filter="url(#b)"/>',
        f'<use href="#c" x="{car}" y="92"/>',
    ]
    return defs, back, []


def _dusk(rng: random.Random) -> tuple[str, list[str], list[str]]:
    defs = (
        _grad("s", "#3b4796", "#f08a6b", "#ffd59a")
        + '<symbol id="h" width="22" height="18" viewBox="0 0 22 18">'
        '<path d="M1 8 11 1l10 7z" fill="#5a3d4d"/><path d="M3 8h16v10H3z" fill="#7b5a63"/>'
        '<path d="M6 10h4v4H6zm7 0h4v4h-4z" fill="#ffd27a"/></symbol>'
        '<symbol id="t" width="14" height="22" viewBox="0 0 14 22">'
        '<path d="M6 12h2v10H6z" fill="#4a3426"/>'
        '<ellipse cx="7" cy="8" rx="7" ry="8" fill="#3e5d3c"/></symbol>'
    )
    back = [
        '<rect width="160" height="86" fill="url(#s)"/>',
        '<path d="M0 70Q40 52 80 66T160 60V86H0z" fill="#a46b7c"/>',
        '<path d="M0 78Q50 66 100 76T160 72V86H0z" fill="#5d6c48"/>',
        *(
            f'<use href="#h" x="{x}" y="{rng.randint(62, 66)}"/>'
            for x in rng.sample([2, 46, 98, 136], 2)
        ),
        *(
            f'<use href="#t" x="{x}" y="{rng.randint(60, 64)}"/>'
            for x in rng.sample([14, 58, 108, 146], 3)
        ),
        '<rect y="84" width="160" height="36" fill="#4b4a55"/>',
        '<rect y="84" width="160" height="2" fill="#8e8a86"/>',
        '<path d="M0 102H160" stroke="#f3e2b8" stroke-width="1.2" stroke-dasharray="9 7"/>',
    ]
    return defs, back, ['<rect width="160" height="120" fill="#ff9a5a" fill-opacity=".08"/>']


def _rain(rng: random.Random) -> tuple[str, list[str], list[str]]:
    defs = (
        _grad("s", "#566477", "#aab5c2")
        + _grad("r", "#3a414b", "#20252c")
        + '<linearGradient id="f" x2="0" y2="1"><stop stop-color="#e6ebf0" stop-opacity="0"/>'
        '<stop offset=".6" stop-color="#e6ebf0" stop-opacity=".6"/>'
        '<stop offset="1" stop-color="#e6ebf0" stop-opacity="0"/></linearGradient>'
        '<pattern id="n" width="7" height="9" patternUnits="userSpaceOnUse">'
        '<path d="M5 0 3 6" stroke="#e3eaf2" stroke-width=".4" stroke-opacity=".6"/></pattern>'
    )
    lights = rng.choice([30, 70, 110])
    back = [
        '<rect width="160" height="86" fill="url(#s)"/>',
        '<path d="M0 64 24 46l20 12 26-20 30 22 22-14 38 22V86H0z" fill="#7b8698"/>',
        '<path d="M0 74 30 62l34 10 40-12 56 14V86H0z" fill="#646f7e"/>',
        '<rect y="52" width="160" height="38" fill="url(#f)"/>',
        '<path d="M0 82H160" stroke="#a3abb4" stroke-width="1.6"/>',
        '<path d="M0 85H160" stroke="#6b737c" stroke-width="3" stroke-dasharray="1 9"/>',
        '<rect y="84" width="160" height="36" fill="url(#r)"/>',
        f'<ellipse cx="{lights}" cy="88" rx="2" ry="1" fill="#fff4c4" filter="url(#b)"/>',
        f'<ellipse cx="{lights + 6}" cy="88" rx="2" ry="1" fill="#fff4c4" filter="url(#b)"/>',
        *(
            f'<ellipse cx="{rng.randint(10, 150)}" cy="{rng.randint(104, 116)}" rx="{rng.randint(10, 18)}" '
            'ry="2.5" fill="#8fa3b8" fill-opacity=".28"/>'
            for _ in range(3)
        ),
        '<path d="M0 102H160" stroke="#d8dde3" stroke-width="1.2" stroke-dasharray="9 7"/>',
    ]
    return defs, back, ['<rect width="160" height="120" fill="url(#n)"/>']


def _morning(rng: random.Random) -> tuple[str, list[str], list[str]]:
    defs = (
        _grad("s", "#5aaef2", "#dff3ff")
        + '<symbol id="k" width="30" height="12" viewBox="0 0 30 12">'
        '<path d="M5 12a5 5 0 0 1 2-9 7 7 0 0 1 13-1 6 6 0 0 1 6 10z" fill="#fff"/></symbol>'
        '<symbol id="t" width="14" height="22" viewBox="0 0 14 22">'
        '<path d="M6 12h2v10H6z" fill="#6b4a32"/>'
        '<ellipse cx="7" cy="8" rx="7" ry="8" fill="#5fae4f"/></symbol>'
        '<symbol id="u" width="34" height="14" viewBox="0 0 34 14">'
        '<path d="M1 3a2 2 0 0 1 2-2h26l4 5v6H1z" fill="#f2b705"/>'
        '<path d="M4 3h5v4H4zm7 0h5v4h-5zm7 0h5v4h-5z" fill="#2c3e57"/>'
        '<ellipse cx="8" cy="12" rx="2" ry="2" fill="#222"/>'
        '<ellipse cx="26" cy="12" rx="2" ry="2" fill="#222"/></symbol>'
    )
    sx = rng.choice([8, 50, 92])
    back = [
        '<rect width="160" height="86" fill="url(#s)"/>',
        *(
            f'<use href="#k" x="{x}" y="{rng.randint(4, 20)}"/>'
            for x in rng.sample([6, 50, 96, 128], 2)
        ),
        f'<path d="M{sx - 4} 52 {sx + 30} 41l34 11z" fill="#8c4a3a"/>',
        f'<path d="M{sx} 84V52h60v32z" fill="#d9825b"/>',
        f'<path d="M{sx + 6} 57h8v7h-8zm14 0h8v7h-8zm14 0h8v7h-8zm14 0h8v7h-8zM{sx + 26} 70h8v14h-8z" '
        'fill="#d4ecff"/>',
        *(
            f'<use href="#t" x="{x}" y="{rng.randint(60, 63)}"/>'
            for x in rng.sample([2, 72, 120, 146], 2)
        ),
        '<rect y="78" width="160" height="7" fill="#7cc46a"/>',
        '<rect y="84" width="160" height="36" fill="#5a5f68"/>',
        f'<path d="M{rng.choice([50, 92])} 106h4v14h-4zm8 0h4v14h-4zm8 0h4v14h-4zm8 0h4v14h-4z" '
        'fill="#f1f1f1"/>',
        '<path d="M0 101H160" stroke="#f4d35e" stroke-width="1.2"/>',
        f'<use href="#u" x="{rng.choice([6, 64, 118])}" y="88"/>',
    ]
    return defs, back, ['<rect width="160" height="120" fill="#fff7d6" fill-opacity=".06"/>']


THEMES = {
    "downtown-01": _night,
    "rainy-highway-02": _rain,
    "suburban-03": _dusk,
    "school-zone-04": _morning,
}


def _scene(task_id: str, index: int) -> tuple[str, list[str]]:
    feats = TASKS[task_id][1]
    rng = random.Random(f"{task_id}:{index}")
    kinds = ["stop", "yield", "no-entry", "warning"] + (
        ["speed", "speed"] if "speed" in feats else []
    )
    kinds += ["school", "school"] if "school" in feats else []
    columns = rng.sample([28, 80, 132], rng.choice([2, 3]))
    signs = [
        {
            "kind": rng.choice(kinds),
            "x": x,
            "y": rng.randint(28, 46),
            "value": rng.choice([30, 40, 50, 60, 80]),
            "grouped": "groups" in feats and rng.random() < 0.35,
            "rotated": "warning" in feats and rng.random() < 0.6,
            "faded": "reflections" in feats and rng.random() < 0.25,
        }
        for x in sorted(columns)
    ]
    # Guarantee that each perception gap, and each tempting wrong rule, is exercised.
    if index == 0:
        signs[0].update(kind="no-entry", grouped=False)
        signs[1].update(kind="speed" if "speed" in feats else "yield", grouped=False)
    if index == 1 and "groups" in feats:
        signs[-1]["grouped"] = True
    if index == 2 and "warning" in feats:
        signs[0].update(kind="warning", rotated=True, grouped=False)
    if index == 3 and "reflections" in feats:
        signs[0].update(faded=True, grouped=False)
    if index == 4 and "school" in feats:
        signs[-1].update(kind="school", grouped=False)
    sun = rng.random() < 0.4 or (index == 2 and "warning" in feats)
    reflect = "reflections" in feats and index % 2 == 1

    # Scenery draws from its own stream so the signs are identical with or without it.
    defs, body, front = THEMES[task_id](random.Random(f"{task_id}:{index}:scenery"))
    body.insert(0, f"<defs>{_COMMON_DEFS}{defs}</defs>")
    if sun:
        cx = rng.choice([20, 70, 145])
        body += [
            f'<ellipse cx="{cx}" cy="14" rx="15" ry="15" fill="#ffe9a6" fill-opacity=".45" '
            'filter="url(#b)"/>',
            f'<circle cx="{cx}" cy="14" r="8" fill="#fc0"/>',
        ]
    truth = []
    for s in signs:
        x, y = s["x"], s["y"]
        body.append(f'<rect x="{x - 1}" y="{y}" width="2" height="{90 - y}" fill="#9aa0a8"/>')
        o = ' opacity="0.85"' if s["faded"] else ""
        args = dict(value=s["value"], rotated=s["rotated"], o=o)
        if s["grouped"]:
            body.append(f'<g transform="translate({x} {y})">')
            body += _sign(s["kind"], 0, 0, **args)
            body.append("</g>")
        else:
            body += _sign(s["kind"], x, y, **args)
        truth.append(f"speed-{s['value']}" if s["kind"] == "speed" else s["kind"])
    if reflect:
        mirror = next((s for s in signs if s["kind"] in ("stop", "yield")), None)
        kind = mirror["kind"] if mirror else "yield"
        x = mirror["x"] if mirror else signs[0]["x"]
        body += [
            el.replace("/>", ' opacity="0.3"/>', 1) if i == 0 else el
            for i, el in enumerate(_sign(kind, x, 110))
            if el.startswith("<polygon")
        ]
    body += front + ['<rect width="160" height="120" fill="url(#v)"/>']
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 120">\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )
    return svg, sorted(truth)


# ---------------------------------------------------------------------------------------
# Adapter

EXECUTORS = ("labeler", "jev")
SOURCES = ("synthetic", "real", "aircraft")
# A reasonable starting harness for photos already reads posted speeds and signal colors.
START_FIXES = {
    "synthetic": frozenset(),
    "real": frozenset({"speed", "lights"}),
    "aircraft": frozenset(),
}


class ImageLabelAdapter:
    def __init__(
        self,
        root: Path,
        *,
        executor: str = "labeler",
        source: str = "synthetic",
        dataset_cache: Path | None = None,
    ):
        if executor not in EXECUTORS:
            raise ValueError(f"executor must be one of {EXECUTORS}")
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        if source != "synthetic" and executor != "jev":
            raise ValueError("photos need the Jev executor; the labeler reads SVG scenes")
        self.root = root
        self.executor = executor
        self.source = source
        self.substrate = OpencodeSubstrate()
        channel = ChannelSpec("opencode", "local/deterministic")
        self.spec = BenchSpec(
            name="signs",
            runs_root=root / "runs",
            install_mode="workspace_config",
            prompts=BenchPrompts(
                work="label every traffic sign in street-scene images",
                domain_word="scene type",
                deliverable_word="labels file",
            ),
            channels={"proposer": channel, "selector": channel},
            leak_globs=("reference_labels.json", "boxes.json"),
        )
        self.execution_count = 0
        base = {name: "" for name in self.substrate.surface().always_on}
        harness_tree.write_tree(
            self.substrate.surface(),
            render_tree(base, START_FIXES[source]),
            self.base_harness(),
            stubs=False,
        )
        if source == "real":
            from . import real_images

            archive = real_images.fetch(dataset_cache or root / "dataset")
            self.titles = real_images.install(root, archive)
            return
        if source == "aircraft":
            from . import aircraft

            manifest = aircraft.fetch(dataset_cache or root / "dataset")
            self.titles = aircraft.install(root, manifest)
            return
        self.titles = {task_id: title for task_id, (title, _feats) in TASKS.items()}
        for task_id in TASKS:
            source = self.task_files(task_id)
            (source / "images").mkdir(parents=True, exist_ok=True)
            (source / "taxonomy.md").write_text(TAXONOMY, encoding="utf-8")
            truth = {}
            for index in range(IMAGES_PER_TASK):
                svg, labels = _scene(task_id, index)
                name = f"img-{index + 1:02d}.svg"
                (source / "images" / name).write_text(svg, encoding="utf-8")
                truth[name] = labels
            answers = self.root / "answers" / task_id
            answers.mkdir(parents=True, exist_ok=True)
            (answers / "reference_labels.json").write_text(
                json.dumps(truth, indent=1, sort_keys=True) + "\n", encoding="utf-8"
            )

    def tasks(self, split: str = "all") -> list[str]:
        return list(self.titles)

    def domain_of(self, task_id: str) -> str:
        return DOMAIN

    def task_prompt(self, task_id: str) -> str:
        return (
            f"{self.titles[task_id]}: label every traffic sign in each image under images/ "
            "using the labels in taxonomy.md. Return labels.json mapping each image file name "
            "to the sorted list of its sign labels, one entry per sign."
        )

    def task_files(self, task_id: str) -> Path:
        return self.root / "sources" / task_id

    def base_harness(self) -> Path:
        return self.root / "base"

    def validate_tree(self, harness: harness_tree.Tree) -> None:
        if "scripts/label.py" not in harness:
            raise ValueError("The local executor requires scripts/label.py")

    def execute(
        self, task_id: str, harness_dir: Path | None, run_id: str, *, timeout: int | None = None
    ) -> Path:
        self.execution_count += 1
        run_dir = self.root / "executions" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        harness = harness_dir or self.base_harness()
        images = self.task_files(task_id) / "images"
        if self.executor == "jev":
            from .vision import run_vision

            run_vision(
                images,
                harness,
                run_dir,
                cache_dir=self.root / "jev-cache",
                mode="aircraft" if self.source == "aircraft" else None,
            )
            return run_dir
        result = subprocess.run(
            [
                sys.executable,
                str(harness / "scripts/label.py"),
                str(images),
                str(run_dir / "labels.json"),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout or 20,
        )
        (run_dir / "trajectory.md").write_text(
            "[ACTION] python scripts/label.py images labels.json\n[OBSERVATION]\n" + result.stdout,
            encoding="utf-8",
        )
        return run_dir

    def delivered(self, run_dir: Path) -> bool:
        return (run_dir / "labels.json").is_file()

    def trajectory(self, run_dir: Path, *, obs_trunc: int) -> str:
        return (run_dir / "trajectory.md").read_text(encoding="utf-8")

    def deliverable_text(self, run_dir: Path) -> str:
        return (run_dir / "labels.json").read_text(encoding="utf-8")

    def deliverable_files(self, run_dir: Path) -> dict[str, str]:
        return {"labels.json": self.deliverable_text(run_dir)}

    def truth(self, task_id: str) -> dict[str, list[str]]:
        path = self.root / "answers" / task_id / "reference_labels.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def truth_boxes(self, task_id: str) -> dict[str, list[dict]]:
        """Answer-key boxes: published annotations for photos, the corrected labeler for SVG."""
        if self.source == "real":
            path = self.root / "answers" / task_id / "boxes.json"
            return json.loads(path.read_text(encoding="utf-8"))
        if self.source == "aircraft":
            # Class labels only: the answer key names the type, not where the aircraft is.
            return {
                name: [{"label": want[0], "bbox": None}]
                for name, want in self.truth(task_id).items()
            }
        images = self.task_files(task_id) / "images"
        return {name: perceive_boxes((images / name).read_text()) for name in self.truth(task_id)}

    def image_record(self, task_id: str, name: str) -> dict:
        path = self.task_files(task_id) / "images" / name
        if path.suffix == ".svg":
            return {"svg": path.read_text(encoding="utf-8")}
        import base64

        from PIL import Image

        data = path.read_bytes()
        with Image.open(path) as picture:
            width, height = picture.size
            if max(width, height) > 720:
                # A lighter preview for the bundle; boxes stay in the original pixel space.
                import io

                preview = picture.convert("RGB")
                preview.thumbnail((720, 720))
                buffer = io.BytesIO()
                preview.save(buffer, "JPEG", quality=82)
                data = buffer.getvalue()
        mime = "png" if path.suffix == ".png" and data == path.read_bytes() else "jpeg"
        return {
            "src": f"data:image/{mime};base64,{base64.b64encode(data).decode()}",
            "width": width,
            "height": height,
        }

    def score(self, run_dir: Path, task_id: str) -> dict | None:
        """Post-hoc only: exact-match accuracy per image and micro F1 over sign labels."""
        try:
            got = json.loads(self.deliverable_text(Path(run_dir)))
        except (OSError, json.JSONDecodeError):
            return None
        truth = self.truth(task_id)
        tp = fp = fn = 0
        per_image = {}
        for name, want in truth.items():
            have = list(got.get(name) or [])
            per_image[name] = sorted(have) == want
            remaining = list(want)
            for label in have:
                if label in remaining:
                    remaining.remove(label)
                    tp += 1
                else:
                    fp += 1
            fn += len(remaining)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
        return {
            "accuracy": sum(per_image.values()) / len(per_image),
            "f1": round(f1, 4),
            "per_image": per_image,
            "labels": got,
        }


# ---------------------------------------------------------------------------------------
# Scripted editor, composer, selector and reflector


def _warnings(text: str) -> list[str]:
    return [line.split(": ", 1)[-1] for line in text.splitlines() if line.startswith("[warn]")]


def _pending(warnings: list[str], current: set[str]) -> list[str]:
    def seen(fix: Fix) -> bool:
        return any(
            (fix.warning and fix.warning in w) or (fix.vision_warning and fix.vision_warning in w)
            for w in warnings
        )

    return [name for name in ORDER if name in GOOD and name not in current and seen(FIXES[name])]


def _stage(label: str) -> tuple[str, int]:
    m = re.fullmatch(r"([a-z]+?)(\d+)", label.split(".")[0])
    if not m:
        raise ValueError(f"unexpected stage label {label!r}")
    return m.group(1), int(m.group(2))


def _mode_of(context: Path) -> str:
    for part in context.parts:
        m = re.fullmatch(r"signs-[a-z]+-(agg|seq)\d+(?:-.*)?", part)
        if m:
            return m.group(1)
    return "agg"


def _with(current: set[str], add) -> set[str]:
    out = set(current)
    for name in add:
        out = {f for f in out if FIXES[f].slot != FIXES[name].slot} | {name}
    return out


def _bad(name: str) -> str:
    return next(n for n, fx in FIXES.items() if fx.alternative == name)


def _labels_json(text: str) -> dict:
    body = text.split("-->\n", 1)[-1] if text.startswith("<!--") else text
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


class ImageLabelProposer(Proposer):
    """Evidence-driven but scripted; no model service and no reference labels are used.

    Each probe makes at most one new edit; library lessons transfer in bulk. Parallel probes
    explore different hypotheses, including a plausible over-general rule; sequential rounds
    fix what the previous round's execution still reports, then explore once nothing is left.
    """

    def call(self, mission, context_dir, label, **kwargs) -> Reply:
        context = Path(context_dir)
        workspace = None
        if label.startswith("select."):
            text = self._select(context)
        elif label.startswith("experience."):
            text = self._reflect(context)
        else:
            workspace = context / "harness"
            text = self._edit(context, label)
        reply = Reply(label=label, text=text, ok=True, attempts=1, workspace=workspace)
        self._calls.append(reply)
        return reply

    def _edit(self, context: Path, label: str) -> str:
        prefix, index = _stage(label)
        mode = _mode_of(context)
        harness = context / "harness"
        current = fixes_of(_read_files(harness))
        library = "\n".join(
            p.read_text(encoding="utf-8") for p in sorted(context.glob("library/**/EXPERIENCE.md"))
        )
        known = {name for name in GOOD if FIXES[name].lesson in library}
        pool = sorted(p.name for p in (context / "pool").glob("*") if p.is_dir())

        if prefix in ("agg", "c"):
            chosen = self._compose(context, prefix, index, current, pool)
            note = (
                "Composed: "
                + "; ".join(FIXES[f].title for f in sorted(chosen, key=ORDER.index))
                + "."
                if chosen
                else "Kept the current harness."
            )
        else:
            latest = sorted((p for p in pool if re.fullmatch(r"c\d+", p)), key=lambda p: int(p[1:]))
            if latest:
                source = f"{latest[-1]} rollout"
                text = (context / "pool" / latest[-1] / "rollout.md").read_text(encoding="utf-8")
            else:
                source = "base rollouts"
                text = "\n".join(
                    p.read_text(encoding="utf-8") for p in sorted(context.glob("rollouts/*.md"))
                )
            warnings = _warnings(text)
            pending = _pending(warnings, current)
            transfer = [f for f in pending if f in known]
            fresh = [f for f in pending if f not in known and FIXES[f].kind == "DOM"]
            if prefix == "inv":
                add = transfer + [f for f in pending if f not in known and FIXES[f].kind == "INV"]
                if not add and "complete" not in current:
                    add = ["complete"]
            else:
                add = [f for f in transfer if FIXES[f].kind == "DOM"]
                applied = [
                    f
                    for f in reversed(ORDER)
                    if f in GOOD and FIXES[f].kind == "DOM" and f in (current | set(add))
                ]
                if mode == "agg":
                    if index == 1 and fresh:
                        add.append(fresh[0])
                    elif index == 2 and (fresh or applied):
                        add.append(_bad((fresh or applied)[0]))
                    elif index >= 3 and len(fresh) > index - 2:
                        add.append(fresh[index - 2])
                elif fresh:
                    add.append(fresh[0])
                elif index >= 2 and applied:
                    add.append(_bad(applied[0]))
            chosen = _with(current, add)
            parts = [f"Read {len(warnings)} warning(s) in the {source}."]
            reused = [f for f in transfer if f in add]
            if reused:
                parts.append(
                    "Library lessons applied: " + ", ".join(FIXES[f].title for f in reused) + "."
                )
            made = [f for f in add if f not in transfer]
            if made:
                kinds = ["new tool" if FIXES[f].tool else "guardrail" for f in made]
                parts.append(
                    "New edit: "
                    + ", ".join(f"{FIXES[f].title} ({k})" for f, k in zip(made, kinds))
                    + "."
                )
            if not add:
                parts.append("No edit is supported by this evidence; kept the harness.")
            note = " ".join(parts)
        files = render_tree(_read_files(harness), chosen)
        for path in (harness / "scripts").glob("*.py"):
            if f"scripts/{path.name}" not in files:
                path.unlink()
        for rel, body in files.items():
            (harness / rel).parent.mkdir(parents=True, exist_ok=True)
            (harness / rel).write_text(body, encoding="utf-8")
        return f"<notes>{note}</notes>"

    def _compose(
        self, context: Path, prefix: str, index: int, current: set[str], pool: list[str]
    ) -> set[str]:
        def fixes(name: str) -> set[str]:
            return fixes_of(_read_files(context / "pool" / name / "harness"))

        if prefix == "agg":
            sets = {p: fixes(p) for p in pool if re.fullmatch(r"(inv|dom)\d+", p)}
            if index >= 3:
                # Consensus: keep only what every probe agreed on.
                return set(current) | (set.intersection(*sets.values()) if sets else set())
            # Union, resolving conflicts toward the earliest (1) or the latest (2) probes.
            # Composer 2 reads only the first two probe pairs.
            ranked = sorted(
                (p for p in sets if index == 1 or _stage(p)[1] <= 2),
                key=lambda p: (_stage(p)[1], p.startswith("dom")),
                reverse=index == 2,
            )
            out: dict[str, str] = {FIXES[f].slot: f for f in current}
            taken: set[str] = set()
            for probe in ranked:
                for name in sorted(sets[probe], key=ORDER.index):
                    if FIXES[name].slot not in taken:
                        out[FIXES[name].slot] = name
                        taken.add(FIXES[name].slot)
            return set(out.values())
        chosen = set(current)
        for probe in (f"inv{index}", f"dom{index}"):
            if probe in pool:
                chosen = _with(chosen, fixes(probe) - current)
        return chosen

    cache_dir: Path | None = None
    # Builds the fallback selector's yes/no question for one candidate's labels on a photo.
    claim_question = None

    def _select(self, context: Path) -> str:
        images = sorted((context / "source/images").glob("*.svg"))
        if not images:
            if self.cache_dir is not None:
                return self._verify_claims(context)
            return self._consensus(context)
        expected = {p.name: perceive(p.read_text(encoding="utf-8")) for p in images}
        best, best_hits = "", -1
        for path in sorted((context / "candidates").glob("*.md")):
            got = _labels_json(path.read_text(encoding="utf-8"))
            hits = sum(sorted(got.get(n) or []) == want for n, want in expected.items())
            if hits > best_hits:
                best, best_hits = path.stem, hits
        return (
            f"<why>Candidate {best} matches the visible signs in {best_hits} of "
            f"{len(expected)} images.</why>\n```choice\n{best}\n```"
        )

    def _verify_claims(self, context: Path) -> str:
        """For photos: ask Jev whether each candidate's full answer matches the photo.

        One yes/no per distinct answer per photo ("exactly these signs, and no others"), so a
        candidate cannot win by repeating the executor's own over-eager guesses label by label.
        """
        from . import vision

        candidates = {
            path.stem: _labels_json(path.read_text(encoding="utf-8"))
            for path in sorted((context / "candidates").glob("*.md"))
        }
        model = vision.settings()["model"]
        points = dict.fromkeys(candidates, 0.0)
        checked = 0
        for path in sorted((context / "source/images").iterdir()):
            if path.suffix.lower() not in vision.PHOTO_SUFFIXES:
                continue
            answers = sorted(
                {tuple(sorted(got.get(path.name) or [])) for got in candidates.values()}
            )
            asked = {}
            for i, labels in enumerate(answers):
                targets = [vision._target(label) for label in labels]
                if self.claim_question is not None:
                    asked[f"answer_{i}"] = {
                        "type": "noul",
                        "instructions": self.claim_question(labels),
                    }
                    continue
                asked[f"answer_{i}"] = {
                    "type": "noul",
                    "instructions": (
                        "Does `image` show exactly these, and no other stop signs, speed-limit "
                        f"signs or lit traffic lights: {'; '.join(targets)}?"
                        if targets
                        else "Does `image` show no stop sign, no speed-limit sign and no lit "
                        "traffic light?"
                    ),
                }
            reply = vision._cached(self.cache_dir, {"image": path.read_bytes()}, asked, model)
            yes = {
                labels: float(
                    ((reply.get("answers") or {}).get(f"answer_{i}") or {}).get("noul") or 0
                )
                for i, labels in enumerate(answers)
            }
            checked += len(answers)
            for anon, got in candidates.items():
                labels = tuple(sorted(got.get(path.name) or []))
                unknown = any(label.endswith("-unknown") for label in labels)
                points[anon] += 0.0 if unknown else yes[labels]
        best = max(points, key=lambda anon: (round(points[anon], 3), -ord(anon[0])))
        return (
            f"<why>Jev checked {checked} distinct answers against the photos; candidate {best} "
            f"matches best (score {points[best]:.2f}).</why>\n```choice\n{best}\n```"
        )

    def _consensus(self, context: Path) -> str:
        """Without a deterministic view of photos, pick the labels most finals agree on."""
        candidates = {
            path.stem: _labels_json(path.read_text(encoding="utf-8"))
            for path in sorted((context / "candidates").glob("*.md"))
        }
        names = sorted({n for got in candidates.values() for n in got})

        def agreement(anon: str) -> int:
            return sum(
                sorted(candidates[anon].get(n) or []) == sorted(other.get(n) or [])
                for n in names
                for key, other in candidates.items()
                if key != anon
            )

        best = max(candidates, key=agreement)
        return (
            f"<why>Candidate {best} agrees most often with the other finals "
            f"({agreement(best)} image matches).</why>\n```choice\n{best}\n```"
        )

    def _reflect(self, context: Path) -> str:
        adopted = fixes_of(_read_files(context / "harness"))
        base = fixes_of(_read_files(context / "base_harness"))
        learned = [f for f in ORDER if f in adopted - base and FIXES[f].tool]
        if not learned:
            return "NO_ENTRY"
        base_warnings = "\n".join(
            p.read_text(encoding="utf-8") for p in sorted(context.glob("rollouts/base*.md"))
        )
        rows = []
        for name in learned:
            carriers = sorted(
                p.name
                for p in (context / "pool").glob("*")
                if p.is_dir() and name in fixes_of(_read_files(p / "harness"))
            )
            fix = FIXES[name]
            source = FIXES.get(fix.alternative, fix)
            signals = [w for w in (source.warning, source.vision_warning) if w]
            signal = next((w for w in signals if w in base_warnings), signals[0] if signals else "")
            gone = [
                c
                for c in carriers
                if (context / "pool" / c / "rollout.md").is_file()
                and signal not in (context / "pool" / c / "rollout.md").read_text()
            ]
            helped = bool(signal) and signal in base_warnings and bool(gone)
            observed = (
                f"the '{signal}' warning in the base trajectories disappeared in {', '.join(gone)}"
                if helped
                else "no trajectory warning distinguishes this edit"
            )
            rows.append(
                f"component: scripts/{fix.tool}.py\ncandidates: {', '.join(carriers)}\n"
                f"change: {fix.title}\nobserved: {observed}\n"
                f"verdict: {'helped' if helped else 'unverified'}"
            )
        return (
            "<edits>\n" + "\n\n".join(rows) + "\n</edits>\n\n<experience>\n"
            "cue: When labeling traffic signs in street-scene images\n"
            f"lesson: {' '.join(FIXES[f].lesson for f in learned)}\n"
            f"evidence: The adopted harness removed {len(learned)} warning type(s) seen in the "
            "base trajectories; see the edit rows for each contrast.\n"
            "confidence: high for edits whose warning disappeared in a probe rollout\n"
            "</experience>"
        )


# ---------------------------------------------------------------------------------------
# Offline Jev stand-in


STANDIN_MODEL = "jev-offline-standin"


def _graded(fraction: float) -> list[float]:
    top = round(0.05 + 0.9 * fraction, 4)
    low = round((1 - top) * 0.4, 4)
    return [low, round(1 - top - low, 4), top]


def _standin_request(payload: dict) -> dict:
    """Score candidates in Jev's response format from the request evidence alone.

    The real `jev.rank_candidates` builds the request, validates this response, applies the
    confidence threshold and tie rule, and writes the audit, exactly as for the API.
    """
    source = payload["state"]["source"]
    expected = {
        name.split("/", 1)[1]: perceive(body)
        for name, body in source.items()
        if name.startswith("images/")
    }
    allowed = re.compile(r"^(stop|yield|no-entry|warning|school|speed-\d+)$")
    shape = {0: [0.9, 0.1, 0.0], 1: [0.05, 0.9, 0.05], 2: [0.0, 0.1, 0.9]}
    dists = {}
    for anon, text in payload["state"]["candidates"].items():
        got = _labels_json(text)
        present = sum(1 for n in expected if n in got)
        valid = all(
            isinstance(v, list) and v == sorted(v) and all(allowed.match(str(x)) for x in v)
            for v in got.values()
        )
        exact = sum(sorted(got.get(n) or []) == want for n, want in expected.items())
        fidelity = _graded(exact / max(1, len(expected)))
        dists[anon] = {
            "delivery": shape[2 if present == len(expected) else 1 if present else 0],
            "conformance": shape[2 if valid and got else 1 if got else 0],
            "fidelity": fidelity,
            "consistency": shape[2 if got else 0],
            "usability": fidelity,
        }
    answers = {}
    for question_id in payload["questions"]:
        anon, dimension = question_id.split("_", 1)
        probabilities = dists[anon][dimension]
        answers[question_id] = {
            "type": "score",
            "score": round(sum(i * p for i, p in enumerate(probabilities)), 4),
            "confidence": 0.9,
            "probabilities": {str(i): p for i, p in enumerate(probabilities)},
        }
    return {"model": STANDIN_MODEL, "answers": answers, "usage": None}


@contextmanager
def offline_jev():
    original = jev._request
    jev._request = _standin_request
    try:
        yield
    finally:
        jev._request = original


# ---------------------------------------------------------------------------------------
# Export


def _diff(base: dict[str, str], tree: dict[str, str]) -> str:
    return "".join(
        line
        for name in sorted(base.keys() | tree.keys())
        for line in difflib.unified_diff(
            base.get(name, "").splitlines(keepends=True),
            tree.get(name, "").splitlines(keepends=True),
            fromfile=f"base/{name}",
            tofile=f"candidate/{name}",
        )
    )


def _fix_list(fixes) -> list[str]:
    return [f for f in ORDER if f in fixes]


def _iou(a: list[float], b: list[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return w * h / (area - w * h)


def _match(predicted: list[dict], truth: list[dict]) -> list[dict]:
    """Mark each detection hit, wrong (right place, wrong label) or extra; add missed signs."""
    if truth and all(t.get("bbox") is None for t in truth):
        # Classification: compare labels; the predicted box stands in for the missing one.
        out = [
            {
                **det,
                "status": "hit" if det["label"] == truth[0]["label"] else "wrong",
                "expected": truth[0]["label"],
            }
            for det in predicted[:1]
        ]
        if not predicted:
            out.append({"label": truth[0]["label"], "bbox": [0, 0, 0, 0], "status": "missed"})
        return out
    pairs: dict[int, int] = {}
    # Same-label pairs first, so an overlapping wrong label cannot take a correct one's sign.
    for same in (True, False):
        for i, det in enumerate(predicted):
            if i in pairs:
                continue
            scores = [
                (_iou(det["bbox"], t["bbox"]), j)
                for j, t in enumerate(truth)
                if j not in pairs.values() and (t["label"] == det["label"]) == same
            ]
            overlap, j = max(scores, default=(0.0, -1))
            if overlap >= 0.3:
                pairs[i] = j
    out = []
    for i, det in enumerate(predicted):
        if i not in pairs:
            out.append({**det, "status": "extra"})
            continue
        expected = truth[pairs[i]]["label"]
        status = "hit" if expected == det["label"] else "wrong"
        out.append({**det, "status": status, "expected": expected})
    used = set(pairs.values())
    out += [{**t, "status": "missed"} for j, t in enumerate(truth) if j not in used]
    return out


def _detections(run_dir: Path, boxes: dict[str, list[dict]]) -> dict[str, list[dict]]:
    path = Path(run_dir) / "detections.json"
    found = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return {name: _match(found.get(name) or [], truth) for name, truth in boxes.items()}


def _components(tree: dict[str, str]) -> dict:
    fixes = fixes_of(tree)
    return {
        "fixes": _fix_list(fixes),
        "tools": [FIXES[f].tool for f in _fix_list(fixes) if FIXES[f].tool],
        "prompt_rules": prompt_rules(tree),
    }


def _dataset_meta(adapter: ImageLabelAdapter) -> dict | None:
    if adapter.source == "synthetic":
        return None
    from . import aircraft, real_images

    module = real_images if adapter.source == "real" else aircraft
    return {"name": module.DATASET, "url": module.PAGE, "license": module.LICENSE}


def export_bundle(adapter: ImageLabelAdapter, k: int, modes, *, jev_mode: str) -> dict:
    """Collect run records and post-hoc label accuracy into one JSON document for the UI."""
    surface = adapter.substrate.surface()
    base_tree = harness_tree.read_tree(surface, adapter.base_harness())
    runs = adapter.spec.runs_root
    tasks = []
    library = {mode: 0 for mode in modes}
    for task_id in adapter.tasks():
        work = runs / f"signs-oc-agg{k}" / task_id
        summary = json.loads((work / "summary.json").read_text(encoding="utf-8"))
        truth = adapter.truth(task_id)
        boxes = adapter.truth_boxes(task_id)
        record = {
            "id": task_id,
            "title": adapter.titles[task_id],
            "status": summary["status"],
            "images": [
                {
                    "name": name,
                    **adapter.image_record(task_id, name),
                    "truth": want,
                    "boxes": boxes[name],
                }
                for name, want in truth.items()
            ],
            "base": [
                {
                    "label": b["label"],
                    **(adapter.score(Path(b["run_dir"]), task_id) or {}),
                    "detections": _detections(Path(b["run_dir"]), boxes),
                }
                for b in summary["base"]
            ],
            "library_before": dict(library),
            "modes": {},
        }
        base_run = Path(summary["base"][0]["run_dir"])
        record["base_warnings"] = _warnings(adapter.trajectory(base_run, obs_trunc=0))
        for mode in modes:
            ns = f"signs-oc-{mode}{k}"
            mode_record = summary["modes"].get(mode) or {}
            sel = summary["selections"].get(mode) or {}
            stages = []
            for stage in mode_record.get("stages", []):
                hdir = runs / ns / task_id / "harnesses" / stage["label"]
                tree = harness_tree.read_tree(surface, hdir)
                run_dir = adapter.root / "executions" / ns / task_id / stage["label"]
                scored = adapter.score(run_dir, task_id) or {}
                prefix, index = _stage(stage["label"])
                stages.append(
                    {
                        **{key: stage[key] for key in ("label", "dim", "notes", "fell_back")},
                        "round": k if prefix == "agg" else index,
                        **_components(tree),
                        "diff": _diff(base_tree, tree),
                        "warnings": _warnings(adapter.trajectory(run_dir, obs_trunc=0)),
                        **scored,
                        "detections": _detections(run_dir, boxes),
                    }
                )
            audit_path = work.parent.parent / ns / task_id / "selection" / f"{mode}{k}.jev.json"
            audit = (
                json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
            )
            request = audit.get("request") or {}
            experience = sel.get("experience") or {}
            if experience.get("lesson"):
                library[mode] += 1
            record["modes"][mode] = {
                "finals": mode_record.get("finals", []),
                "stages": stages,
                "selection": {
                    key: sel.get(key)
                    for key in (
                        "selector",
                        "chosen_id",
                        "chosen_label",
                        "id_map",
                        "why",
                        "usd",
                        "ok",
                        "fallback",
                    )
                },
                "jev": sel.get("jev") or {},
                "selector_config": sel.get("selector_config") or {},
                "jev_request": {
                    "bytes": len(json.dumps(request, ensure_ascii=False).encode("utf-8"))
                    if request
                    else 0,
                    "questions": {
                        qid: {"criteria": q.get("criteria")}
                        for qid, q in (request.get("questions") or {}).items()
                    },
                    "source_files": sorted((request.get("state") or {}).get("source") or {}),
                },
                "experience": experience,
                "ledger": {
                    key: (mode_record.get("ledger") or {}).get(key) for key in ("R", "P", "usd")
                },
            }
        tasks.append(record)
    return {
        "meta": {
            "title": "Harness² × Jev — traffic-sign labeling",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "k": k,
            "modes": list(modes),
            "domain": DOMAIN,
            "jev_mode": jev_mode,
            "jev_model": os.environ.get("HARNESS2_JEV_MODEL") or "jev-1.14.0",
            "executor": adapter.executor,
            "source": adapter.source,
            "dataset": _dataset_meta(adapter),
            "executions": adapter.execution_count,
            "editor": "deterministic scripted editor (local demo)",
        },
        "taxonomy": TAXONOMY,
        "fixes": {
            name: {
                "title": fix.title,
                "kind": fix.kind,
                "good": fix.good,
                "lesson": fix.lesson,
                "alternative": fix.alternative,
                "warning": fix.warning or fix.vision_warning,
                "trigger": fix.trigger,
                "action": fix.tool or "require_all_images",
                "tool": fix.tool,
            }
            for name, fix in FIXES.items()
        },
        "tasks": tasks,
    }
