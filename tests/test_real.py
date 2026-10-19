"""Street photos: install from the dataset archive, Jev photo questions, image-aware selection."""

import importlib.util
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from harness2 import image_demo, real_images, vision
from harness2.core import jev, schedule
from harness2.core import tree as harness_tree

HAS_DEPS = all(importlib.util.find_spec(m) for m in ("PIL", "resvg_py"))
STEMS = {
    "streets-a": ("Street A", ["road1_png.rf.aaa", "road2_png.rf.bbb"]),
    "streets-b": ("Street B", ["road3_png.rf.ccc", "road4_png.rf.ddd"]),
}
# class ids from real_images.CLASSES: 1 light-red, 9 speed-50, 14 stop
LABELS = {
    "road1_png.rf.aaa": "9 0.5 0.3 0.2 0.2\n",
    "road2_png.rf.bbb": "14 0.6 0.4 0.3 0.3\n1 0.2 0.3 0.1 0.2\n",
    "road3_png.rf.ccc": "9 0.4 0.4 0.2 0.2\n",
    "road4_png.rf.ddd": "1 0.3 0.3 0.1 0.2\n",
}
TRUTH = {
    "road1_png.rf.aaa": {"speed_50"},
    "road2_png.rf.bbb": {"stop", "light_red"},
    "road3_png.rf.ccc": {"speed_50"},
    "road4_png.rf.ddd": {"light_red"},
}


def archive(path: Path) -> Path:
    from PIL import Image

    with zipfile.ZipFile(path, "w") as bundle:
        for i, (stem, text) in enumerate(LABELS.items()):
            buffer = io.BytesIO()
            Image.new("RGB", (416, 416), (40 * i, 90, 160)).save(buffer, "JPEG")
            bundle.writestr(f"valid/images/{stem}.jpg", buffer.getvalue())
            bundle.writestr(f"valid/labels/{stem}.txt", text)
    return path


def photo_answers(asked: dict, present: set[str], confidence: float = 0.9) -> dict:
    """Say yes to the questions in `present`; boxes everywhere."""
    answers = {}
    for qid, question in asked.items():
        if question["type"] == "box":
            box = {"top": 0.2, "bottom": 0.4, "left": 0.4, "right": 0.6}
            answers[qid] = {"type": "box", "box": box, "confidence": confidence}
        else:
            yes = qid in present or (qid == "speed_limit" and any("speed" in p for p in present))
            yes = yes or (qid == "traffic_light" and any("light" in p for p in present))
            answers[qid] = {"type": "noul", "noul": 0.95 if yes else 0.05}
    return {"model": "jev-1.14.0", "answers": answers, "usage": {"input_tokens": 900}}


@unittest.skipUnless(HAS_DEPS, "install .[jev] for street photos")
class RealImageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(
            os.environ,
            {
                "TYPESAFE_API_KEY": "test-key-never-send",
                "HARNESS2_SELECTOR": "jev",
                "HARNESS2_JEV_IMAGES": "1",
                "HARNESS2_JEV_MODEL": "",
                "HARNESS2_JEV_MAX_REQUEST_BYTES": "48000",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        tasks = patch.object(real_images, "TASKS", STEMS)
        tasks.start()
        self.addCleanup(tasks.stop)
        cache = self.root / "cache"
        cache.mkdir()
        archive(cache / "valid.zip")
        self.adapter = image_demo.ImageLabelAdapter(
            self.root / "work", executor="jev", source="real", dataset_cache=cache
        )

    def test_install_writes_photos_truth_and_attribution(self):
        self.assertEqual(self.adapter.tasks(), list(STEMS))
        truth = self.adapter.truth("streets-a")
        self.assertEqual(truth, {"img-01.jpg": ["speed-50"], "img-02.jpg": ["light-red", "stop"]})
        boxes = self.adapter.truth_boxes("streets-a")["img-01.jpg"]
        self.assertEqual(boxes, [{"label": "speed-50", "bbox": [166.4, 83.2, 249.6, 166.4]}])
        record = self.adapter.image_record("streets-a", "img-01.jpg")
        self.assertTrue(record["src"].startswith("data:image/jpeg;base64,"))
        self.assertEqual((record["width"], record["height"]), (416, 416))
        self.assertIn("CC BY 4.0", (self.root / "work/sources/ATTRIBUTION.md").read_text())
        with self.assertRaises(ValueError):
            image_demo.ImageLabelAdapter(self.root / "x", source="real")

    def test_harness_tools_change_the_photo_questions_and_labels(self):
        surface = self.adapter.substrate.surface()
        base = harness_tree.read_tree(surface, self.adapter.base_harness())
        bare = self.root / "bare"
        harness_tree.write_tree(surface, image_demo.render_tree(base, set()), bare)
        spec, _tools = vision.harness_spec(bare, "photo")
        asked = vision.photo_questions(spec)
        self.assertEqual(
            set(asked),
            {
                "stop",
                "stop__box",
                "speed_limit",
                "speed_limit__box",
                "traffic_light",
                "traffic_light__box",
            },
        )
        found, warnings = vision.read_photo(
            photo_answers(asked, {"speed_50", "light_red"})["answers"], spec, 0.6, (416, 416)
        )
        self.assertEqual(sorted(d["label"] for d in found), ["light-unknown", "speed-unknown"])
        self.assertEqual(image_demo._pending(warnings, set()), ["speed", "lights"])

        target = self.root / "tools"
        harness_tree.write_tree(
            surface, image_demo.render_tree(base, {"speed", "lights", "verify"}), target
        )
        spec, tools = vision.harness_spec(target, "photo")
        self.assertEqual(tools, ["read_light_color", "read_speed_limit", "verify_crop"])
        asked = vision.photo_questions(spec)
        self.assertIn("speed_120", asked)
        self.assertIn("light_green__box", asked)
        checked = []

        def recheck(label, bbox):
            checked.append(label)
            return label == "speed-50"

        found, warnings = vision.read_photo(
            photo_answers(asked, {"speed_50", "light_red"}, confidence=0.4)["answers"],
            spec,
            0.6,
            (416, 416),
            recheck,
        )
        self.assertEqual([d["label"] for d in found], ["speed-50"])
        self.assertEqual(sorted(checked), ["light-red", "speed-50"])
        self.assertEqual(warnings, [])
        png = vision.crop(
            (self.adapter.task_files("streets-a") / "images/img-01.jpg").read_bytes(),
            [100, 100, 150, 160],
        )
        self.assertEqual(png[:4], b"\x89PNG")

    def test_stacked_speed_values_are_reported_then_resolved(self):
        spec, _tools = vision.harness_spec(self.adapter.base_harness(), "photo")
        self.assertTrue(spec["read_value"] and spec["read_light"])
        asked = vision.photo_questions(spec)
        answers = photo_answers(asked, {"speed_100", "speed_120", "speed_20"})["answers"]
        answers["speed_20"]["noul"] = 0.6
        answers["speed_120__box"]["box"] = {"top": 0.6, "bottom": 0.8, "left": 0.4, "right": 0.6}
        found, warnings = vision.read_photo(answers, spec, 0.6, (416, 416))
        self.assertEqual(sorted(d["label"] for d in found), ["speed-100", "speed-120", "speed-20"])
        self.assertEqual(image_demo._pending(warnings, {"speed", "lights"}), ["dedupe"])
        spec["one_value"] = True
        found, warnings = vision.read_photo(answers, spec, 0.6, (416, 416))
        self.assertEqual(sorted(d["label"] for d in found), ["speed-100", "speed-120"])
        self.assertEqual(warnings, [])
        spec["single_value"] = True
        found, _ = vision.read_photo(answers, spec, 0.6, (416, 416))
        self.assertEqual(len([d for d in found if d["label"].startswith("speed")]), 1)

    def test_selection_sends_photos_as_images(self):
        root = self.root / "selection"
        (root / "source/images").mkdir(parents=True)
        (root / "candidates").mkdir()
        (root / "task.md").write_text("Label the signs.")
        (root / "source/taxonomy.md").write_text("stop")
        (root / "source/images/img-01.jpg").write_bytes(b"\xff\xd8 not decoded here")
        settings = dict(jev.config.selector_settings())
        self.assertTrue(settings["images"])
        payload = jev._payload(root, {"A": "{}", "B": "{}"}, _prompts(), settings)
        self.assertEqual(
            payload["images"], {"images/img-01.jpg": str(root / "source/images/img-01.jpg")}
        )
        self.assertEqual(
            payload["state"]["source"]["images/img-01.jpg"], "[image: images/img-01.jpg]"
        )
        settings.pop("images")
        with self.assertRaisesRegex(ValueError, "non-text"):
            jev._payload(root, {"A": "{}", "B": "{}"}, _prompts(), settings)

    def test_fallback_selector_asks_jev_to_confirm_claims(self):
        context = self.root / "select"
        (context / "source/images").mkdir(parents=True)
        (context / "candidates").mkdir()
        photo = self.adapter.task_files("streets-a") / "images/img-01.jpg"
        (context / "source/images/img-01.jpg").write_bytes(photo.read_bytes())
        for anon, labels in (("A", ["speed-unknown"]), ("B", ["speed-60"]), ("C", ["speed-50"])):
            text = json.dumps({"img-01.jpg": labels})
            (context / "candidates" / f"{anon}.md").write_text(f"<!-- {anon} -->\n{text}")

        def ask(tiles, asked, model):
            answers = {
                qid: {
                    "type": "noul",
                    "noul": 0.9 if q["instructions"].endswith("showing 50?") else 0.1,
                }
                for qid, q in asked.items()
            }
            return {"model": model, "answers": answers, "usage": {"input_tokens": 1}}

        proposer = image_demo.ImageLabelProposer(self.adapter.spec.channels["proposer"])
        proposer.cache_dir = self.root / "claims"
        with patch.object(vision, "_ask", side_effect=ask) as asked:
            reply = proposer.call("", context, "select.seq2")
        self.assertIn("```choice\nC\n```", reply.text)
        self.assertEqual(asked.call_count, 1)

    def test_stream_runs_on_photos_and_exports_them(self):
        seen = []

        def ask(tiles, asked, model):
            self.assertEqual(
                set(tiles), {"image"} if "stop" in asked or "speed_10" in asked else {"crop"}
            )
            return photo_answers(asked, set().union(*TRUTH.values()))

        def select(payload):
            seen.append(payload)
            response = {"model": "jev-1.14.0", "answers": {}, "usage": {"input_tokens": 1}}
            for i, qid in enumerate(payload["questions"]):
                anon = qid.split("_", 1)[0]
                level = {"A": 2, "B": 1}.get(anon, 0)
                probabilities = {str(n): float(n == level) for n in range(3)}
                response["answers"][qid] = {
                    "type": "score",
                    "score": float(level),
                    "confidence": 0.9,
                    "probabilities": probabilities,
                }
            return response

        proposer = image_demo.ImageLabelProposer(self.adapter.spec.channels["proposer"])
        with (
            patch.object(vision, "_ask", side_effect=ask),
            patch.object(jev, "_request", side_effect=select),
        ):
            result = schedule.run_domain(
                image_demo.DOMAIN,
                self.adapter.tasks(),
                2,
                adapter=self.adapter,
                prop=proposer,
                runs_root=self.adapter.spec.runs_root,
                library_root=self.root / "work/library",
                modes=("seq",),
            )
        self.assertEqual((result["done"], result["failed"]), (2, 0))
        self.assertTrue(seen and all(p.get("images") for p in seen))
        bundle = image_demo.export_bundle(self.adapter, 2, ("seq",), jev_mode="live")
        self.assertEqual(bundle["meta"]["source"], "real")
        self.assertEqual(bundle["meta"]["dataset"]["license"], "CC BY 4.0")
        image = bundle["tasks"][0]["images"][0]
        self.assertTrue(image["src"].startswith("data:image/jpeg"))
        stages = bundle["tasks"][0]["modes"]["seq"]["stages"]
        self.assertEqual(stages[0]["tools"], ["read_speed_limit", "read_light_color"])
        self.assertIn("read_speed_limit", stages[-1]["tools"])
        detections = {d["label"]: d for d in stages[-1]["detections"]["img-01.jpg"]}
        self.assertEqual(detections["speed-50"]["status"], "hit")
        self.assertEqual(detections["speed-50"]["confidence"], 0.9)
        self.assertEqual(detections["stop"]["status"], "extra")


def _prompts():
    return image_demo.BenchPrompts(work="label every traffic sign in street-scene images")


if __name__ == "__main__":
    unittest.main()
