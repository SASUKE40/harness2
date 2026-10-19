"""Jev as the labeler: the harness shapes the image questions; answers become detections."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness2 import config, image_demo, vision
from harness2.core import tree as harness_tree

HAS_RESVG = importlib.util.find_spec("resvg_py") is not None
HAS_SDK = importlib.util.find_spec("typesafe_sdk") is not None


def fake_answers(asked: dict) -> dict:
    """Left: a speed sign; middle: nothing; right: a box low on the road (a reflection)."""
    answers = {}
    for qid, question in asked.items():
        tile, kind = qid.rsplit("_", 1)
        if kind == "sign":
            choice = {"left": "speed-limit", "middle": "none", "right": "yield"}[tile]
            answers[qid] = {
                "type": "choice",
                "choice": choice,
                "confidence": 0.82,
                "probabilities": {c: float(c == choice) for c in question["criteria"]},
            }
        elif kind == "box":
            top = 0.78 if tile == "right" else 0.2
            answers[qid] = {
                "type": "box",
                "box": {"top": top, "bottom": top + 0.15, "left": 0.3, "right": 0.7},
                "confidence": 0.74,
            }
        else:
            answers[qid] = {
                "type": "choice",
                "choice": "50",
                "confidence": 0.9,
                "probabilities": {c: float(c == "50") for c in question["criteria"]},
            }
    return {
        "model": "jev-1.14.0",
        "answers": answers,
        "usage": {"input_tokens": 1200, "output_tokens": 0},
    }


@unittest.skipUnless(HAS_RESVG, "install .[jev] for the Jev executor")
class VisionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(
            os.environ, {"TYPESAFE_API_KEY": "test-key-never-send", "HARNESS2_JEV_MODEL": ""}
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.adapter = image_demo.ImageLabelAdapter(self.root, executor="jev")
        self.images = self.adapter.task_files("downtown-01") / "images"
        self.surface = self.adapter.substrate.surface()

    def harness(self, fixes) -> Path:
        base = harness_tree.read_tree(self.surface, self.adapter.base_harness())
        target = self.root / "harness" / "-".join(sorted(fixes) or ["base"])
        harness_tree.write_tree(self.surface, image_demo.render_tree(base, fixes), target)
        return target

    def test_tiles_render_to_png(self):
        png = vision.render_tile((self.images / "img-01.svg").read_text(), 1)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")

    def test_the_harness_decides_what_jev_is_asked(self):
        spec, tools = vision.harness_spec(self.adapter.base_harness())
        self.assertEqual((tools, spec["rules"], spec["read_value"]), ([], [], False))
        self.assertNotIn("school", spec["classes"])
        asked = vision.questions(spec)
        self.assertEqual({q["type"] for q in asked.values()}, {"choice", "box"})
        self.assertEqual(len(asked), 2 * len(vision.TILES))

        spec, tools = vision.harness_spec(self.harness({"speed", "reflections", "school"}))
        self.assertEqual(tools, ["read_speed_limit", "school_sign", "skip_reflection"])
        self.assertTrue(spec["read_value"])
        self.assertEqual(spec["max_box_top"], 0.7)
        self.assertIn("school", spec["classes"])
        asked = vision.questions(spec)
        self.assertIn("left_value", asked)
        instructions = asked["left_sign"]["instructions"]
        self.assertIn(image_demo.FIXES["speed"].lesson, instructions)
        self.assertIn("reflection, not a sign", instructions)
        self.assertNotIn("(tool:", instructions)

    def test_answers_become_detections_warnings_and_cached_labels(self):
        calls = []

        def ask(tiles, asked, model):
            calls.append((sorted(tiles), model))
            self.assertTrue(all(png[:4] == b"\x89PNG" for png in tiles.values()))
            return fake_answers(asked)

        with patch.object(vision, "_ask", side_effect=ask):
            run = self.root / "run-base"
            run.mkdir()
            vision.run_vision(
                self.images, self.adapter.base_harness(), run, cache_dir=self.root / "c"
            )
            self.assertEqual(len(calls), image_demo.IMAGES_PER_TASK)
            self.assertEqual(calls[0], (sorted(vision.TILES), "jev-1.14.0"))
            labels = json.loads((run / "labels.json").read_text())
            self.assertEqual(labels["img-01.svg"], ["speed-unknown", "yield"])
            trajectory = (run / "trajectory.md").read_text()
            self.assertIn("speed-limit sign without a posted value", trajectory)
            self.assertIn("sign box below the road line", trajectory)
            detections = json.loads((run / "detections.json").read_text())["img-01.svg"]
            self.assertEqual(detections[0]["confidence"], 0.74)
            self.assertEqual(detections[0]["bbox"][1], 24.0)
            warnings = image_demo._warnings(trajectory)
            self.assertEqual(image_demo._pending(warnings, set()), ["speed", "reflections"])

            fixed = self.root / "run-fixed"
            fixed.mkdir()
            harness = self.harness({"speed", "reflections"})
            vision.run_vision(self.images, harness, fixed, cache_dir=self.root / "c")
            labels = json.loads((fixed / "labels.json").read_text())
            self.assertEqual(labels["img-01.svg"], ["speed-50"])
            self.assertIn("[tool] read_speed_limit", (fixed / "trajectory.md").read_text())

            count = len(calls)
            vision.run_vision(self.images, harness, fixed, cache_dir=self.root / "c")
            self.assertEqual(len(calls), count)

    def test_validation_requires_a_key(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}):
            with self.assertRaisesRegex(config.ConfigError, "TYPESAFE_API_KEY"):
                vision.validate()

    @unittest.skipUnless(HAS_SDK, "install .[jev] for the SDK contract test")
    def test_sdk_sends_images_with_the_opt_in_header(self):
        import httpx2
        import typesafe_sdk

        real = typesafe_sdk.TypeSafeClient
        seen = []

        def handle(request):
            body = request.content
            start = body.index(b"{")
            payload = json.JSONDecoder().raw_decode(body[start:].decode("utf-8", "ignore"))[0]
            seen.append((request.headers.get(vision.IMAGES_HEADER), payload))
            self.assertIn(b"\x89PNG", body)
            return httpx2.Response(200, json=fake_answers(payload["questions"]))

        def client(**kwargs):
            return real(**kwargs, transport=httpx2.MockTransport(handle))

        with patch.object(typesafe_sdk, "TypeSafeClient", side_effect=client):
            run = self.root / "run-sdk"
            run.mkdir()
            vision.run_vision(
                self.images, self.adapter.base_harness(), run, cache_dir=self.root / "s"
            )
        header, payload = seen[0]
        self.assertEqual(header, "true")
        self.assertEqual(payload["model"], "jev-1.14.0")
        self.assertEqual(payload["questions"]["left_box"]["type"], "box")
        self.assertEqual(set(payload["state"]), set(vision.TILES))
        labels = json.loads((run / "labels.json").read_text())
        self.assertEqual(labels["img-01.svg"], ["speed-unknown", "yield"])


if __name__ == "__main__":
    unittest.main()
