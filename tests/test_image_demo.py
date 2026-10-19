"""The traffic-sign workload: tools, prompt rules, round-by-round gains, transfer and Jev."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness2 import image_demo
from harness2.cli import image_demo as cli
from harness2.core import jev


class ImageDemoTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, {"HARNESS2_SELECTOR": "proposer"})
        environment.start()
        self.addCleanup(environment.stop)

    def test_tools_fix_every_gap_and_round_trip_through_the_harness(self):
        adapter = image_demo.ImageLabelAdapter(self.root / "work")
        base = image_demo.build_labeler(image_demo.render_tree({}, set()))
        for task_id, (_title, feats) in image_demo.TASKS.items():
            images = adapter.task_files(task_id) / "images"
            truth = adapter.truth(task_id)
            for name, want in truth.items():
                self.assertEqual(image_demo.perceive((images / name).read_text()), want)
            self.assertLess(
                sum(base["classify"]((images / n).read_text()) == w for n, w in truth.items()),
                len(truth),
            )
            for good in feats:
                with self.subTest(task=task_id, fix=good):
                    without = {f for f in image_demo.GOOD if f in feats and f != good}
                    labeler = image_demo.build_labeler(image_demo.render_tree({}, without))
                    misses = [
                        n
                        for n, w in truth.items()
                        if labeler["classify"]((images / n).read_text()) != w
                    ]
                    self.assertTrue(misses, f"{good} is never needed in {task_id}")
        for name, fix in image_demo.FIXES.items():
            files = image_demo.render_tree({}, {name})
            self.assertEqual(image_demo.fixes_of(files), {name})
            if fix.tool:
                self.assertIn(f"scripts/{fix.tool}.py", files)
                self.assertIn(fix.lesson, "\n".join(image_demo.prompt_rules(files)))

    def test_offline_stream_improves_round_by_round_and_jev_picks_the_best_final(self):
        out = self.root / "web"
        code = cli.main(["--output", str(self.root / "run"), "--export", str(out)])
        self.assertEqual(code, 0)
        bundle = json.loads((out / "rsi.json").read_text())
        self.assertEqual((bundle["meta"]["jev_mode"], bundle["meta"]["k"]), ("offline", 3))
        tasks = bundle["tasks"]
        self.assertEqual([t["id"] for t in tasks], list(image_demo.TASKS))
        for task in tasks:
            self.assertTrue(all(b["accuracy"] < 0.5 for b in task["base"]))
            base_status = {d["status"] for ds in task["base"][0]["detections"].values() for d in ds}
            self.assertTrue(base_status & {"missed", "wrong", "extra"})
            for mode in ("agg", "seq"):
                record = task["modes"][mode]
                stages = {s["label"]: s for s in record["stages"]}
                chosen = stages[record["selection"]["chosen_label"]]
                finals = [stages[f]["accuracy"] for f in record["finals"]]
                self.assertEqual(record["selection"]["selector"], "jev")
                self.assertEqual(chosen["accuracy"], max(finals))
                self.assertEqual(len(set(finals)), len(finals))
                self.assertEqual(len(chosen["tools"]), len(chosen["prompt_rules"]))
                self.assertTrue(record["experience"]["lesson"])
        first = tasks[0]["modes"]["seq"]
        rounds = [s["accuracy"] for s in first["stages"] if s["dim"] == "COMPOSE"]
        self.assertEqual(rounds, sorted(rounds))
        self.assertLess(rounds[0], rounds[-1])
        self.assertEqual(rounds[-1], 1.0)
        for mode in ("agg", "seq"):
            self.assertEqual([t["library_before"][mode] for t in tasks], [0, 1, 2, 3])
            probes = [t["modes"][mode]["stages"][0]["accuracy"] for t in tasks]
            self.assertGreater(max(probes[1:]), probes[0])
            adopted = [
                next(
                    s
                    for s in t["modes"][mode]["stages"]
                    if s["label"] == t["modes"][mode]["selection"]["chosen_label"]
                )
                for t in tasks
            ]
            self.assertEqual(
                {d["status"] for ds in adopted[-1]["detections"].values() for d in ds}, {"hit"}
            )
        for audit in (self.root / "run/runs").rglob("*.jev.json"):
            self.assertNotIn("reference_labels", audit.read_text())

    def test_offline_patch_is_scoped(self):
        original = jev._request
        with image_demo.offline_jev():
            self.assertIs(jev._request, image_demo._standin_request)
        self.assertIs(jev._request, original)


if __name__ == "__main__":
    unittest.main()
