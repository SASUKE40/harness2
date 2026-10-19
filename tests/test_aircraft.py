"""Aircraft recognition: harness-shaped type questions, warnings, classification scoring."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness2 import aircraft, image_demo, vision
from harness2.core import tree as harness_tree

HAS_DEPS = all(importlib.util.find_spec(m) for m in ("PIL", "resvg_py"))
TASKS = {"fleet-a": ("Fleet A", [1, 2]), "fleet-b": ("Fleet B", [3, 4])}
LABELS = {"1": "ATR-72", "2": "CRJ-200", "3": "E-190", "4": "Saab 340"}


def manifest(cache: Path) -> Path:
    from PIL import Image

    cache.mkdir(parents=True, exist_ok=True)
    for index in LABELS:
        Image.new("RGB", (800, 530), (90, 120, 160)).save(cache / f"{index}.jpg", "JPEG")
    path = cache / "manifest.json"
    path.write_text(json.dumps(LABELS))
    return path


def choice(asked: dict, pick: str, probabilities: dict, confidence: float) -> dict:
    answers = {
        "plane__box": {
            "type": "box",
            "box": {"top": 0.3, "bottom": 0.7, "left": 0.1, "right": 0.9},
            "confidence": 0.9,
        }
    }
    if "type" in asked:
        answers["type"] = {
            "type": "choice",
            "choice": pick,
            "confidence": confidence,
            "probabilities": probabilities,
        }
    else:
        names = list(aircraft.TYPES)
        for i in range(len(names)):
            answers[f"type_{i}"] = {"type": "noul", "noul": 0.9 if names[i] == pick else 0.2}
    return {"model": "jev-1.14.0", "answers": answers, "usage": {"input_tokens": 1}}


@unittest.skipUnless(HAS_DEPS, "install .[jev] for photos")
class AircraftTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(
            os.environ, {"TYPESAFE_API_KEY": "test-key", "HARNESS2_JEV_MODEL": ""}
        )
        environment.start()
        self.addCleanup(environment.stop)
        tasks = patch.object(aircraft, "TASKS", TASKS)
        tasks.start()
        self.addCleanup(tasks.stop)
        manifest(self.root / "cache")
        self.adapter = image_demo.ImageLabelAdapter(
            self.root / "work", executor="jev", source="aircraft", dataset_cache=self.root / "cache"
        )
        self.surface = self.adapter.substrate.surface()

    def harness(self, fixes) -> Path:
        base = harness_tree.read_tree(self.surface, self.adapter.base_harness())
        target = self.root / "h" / ("-".join(sorted(fixes)) or "base")
        harness_tree.write_tree(self.surface, image_demo.render_tree(base, fixes), target)
        return target

    def test_install_and_classification_scoring(self):
        self.assertEqual(self.adapter.tasks(), list(TASKS))
        self.assertEqual(
            self.adapter.truth("fleet-a"), {"img-01.jpg": ["ATR-72"], "img-02.jpg": ["CRJ-200"]}
        )
        boxes = self.adapter.truth_boxes("fleet-a")["img-01.jpg"]
        hit = image_demo._match([{"label": "ATR-72", "bbox": [1, 1, 9, 9]}], boxes)
        wrong = image_demo._match([{"label": "ATR-42", "bbox": [1, 1, 9, 9]}], boxes)
        self.assertEqual(
            (hit[0]["status"], wrong[0]["status"], wrong[0]["expected"]), ("hit", "wrong", "ATR-72")
        )
        self.assertIn("non-commercial", (self.root / "work/sources/ATTRIBUTION.md").read_text())
        self.assertEqual(
            aircraft.claim_question(("E-190",)), "Is the aircraft in `image` an E-190?"
        )

    def test_tools_rewrite_the_type_question_and_warnings_drive_them(self):
        spec, tools = vision.harness_spec(self.adapter.base_harness(), "aircraft")
        asked = vision.aircraft_questions(spec)
        self.assertEqual((tools, set(asked)), ([], {"type", "plane__box"}))
        self.assertEqual(asked["type"]["criteria"]["ATR-42"], "an ATR-42")
        close = {"ATR-72": 0.48, "ATR-42": 0.4}
        found, warnings = vision.read_aircraft(
            choice(asked, "ATR-72", close, 0.3)["answers"], spec, 0.6, (800, 530)
        )
        self.assertEqual(found[0]["label"], "ATR-72")
        self.assertEqual(found[0]["bbox"], [80.0, 159.0, 720.0, 371.0])
        self.assertEqual(image_demo._pending(warnings, set()), ["cues", "per_type"])

        spec, tools = vision.harness_spec(self.harness({"cues", "per_type"}), "aircraft")
        self.assertEqual(tools, ["ask_each_type", "spotter_cues"])
        asked = vision.aircraft_questions(spec)
        self.assertNotIn("type", asked)
        self.assertIn("an ATR 42 (high-wing twin turboprop", asked["type_0"]["instructions"])
        self.assertIn(image_demo.FIXES["cues"].lesson, asked["type_0"]["instructions"])
        found, warnings = vision.read_aircraft(
            choice(asked, "Saab 340", {}, 0)["answers"], spec, 0.6, (800, 530)
        )
        self.assertEqual(
            (found[0]["label"], found[0]["confidence"], warnings), ("Saab 340", 0.9, [])
        )

    def test_rollout_uses_aircraft_mode(self):
        calls = []

        def ask(tiles, asked, model):
            calls.append(set(asked))
            return choice(asked, "ATR-72", {"ATR-72": 0.9, "ATR-42": 0.1}, 0.85)

        with patch.object(vision, "_ask", side_effect=ask):
            run = self.adapter.execute("fleet-a", None, "probe")
        self.assertEqual(calls[0], {"type", "plane__box"})
        score = self.adapter.score(run, "fleet-a")
        self.assertEqual((score["accuracy"], score["per_image"]["img-01.jpg"]), (0.5, True))


if __name__ == "__main__":
    unittest.main()
