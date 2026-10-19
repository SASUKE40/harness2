"""Jev's selection contract, including uncertainty, evidence boundaries and resume."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness2 import config
from harness2.cli import run_sweep
from harness2.core import jev, schedule, selection
from harness2.core.library import Library
from harness2.core.spec import RunView
from harness2.demo import DemoAdapter, DemoProposer


def response_for(payload):
    candidates = payload["state"]["candidates"]
    preferred = next(anon for anon, text in candidates.items() if "Total: 85.15" in text)
    answers = {}
    for question_id in payload["questions"]:
        score = 2 if question_id.startswith(preferred + "_") else 0
        answers[question_id] = {
            "type": "score",
            "score": float(score),
            "confidence": 1.0,
            "probabilities": {str(i): float(i == score) for i in range(3)},
            "legend": {
                str(i): level
                for i, level in enumerate(payload["questions"][question_id]["criteria"])
            },
        }
    return {
        "model": "jev-1.14.0",
        "answers": answers,
        "usage": {"input_tokens": 100, "output_tokens": 30},
    }


class JevTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(
            os.environ,
            {
                "HARNESS2_SELECTOR": "jev",
                "HARNESS2_JEV_MODEL": "jev-1.14.0",
                "HARNESS2_JEV_MIN_CONFIDENCE": "0.6",
                "HARNESS2_JEV_MAX_REQUEST_BYTES": "28000",
                "TYPESAFE_API_KEY": "test-key-never-send",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.adapter = DemoAdapter(self.root)
        self.proposer = DemoProposer(self.adapter.spec.channels["proposer"])
        self.candidates = {}
        for label, amount in (("agg1", "105.25"), ("agg2", "85.15")):
            run_dir = self.root / "executions" / label
            run_dir.mkdir(parents=True)
            (run_dir / "answer.txt").write_text(f"Total: {amount}\n")
            self.candidates[label] = RunView(label, label, run_dir, True, True)

    def select(self, proposer=None):
        return selection.run_selection(
            "ledger-1",
            2,
            "agg",
            self.candidates,
            adapter=self.adapter,
            prop=proposer or self.proposer,
            work_dir=self.root,
            pool_kind="finals",
        )

    def test_selection_is_anonymous_complete_and_audited(self):
        source = self.adapter.task_files("ledger-1")
        (source / "reference_answer.txt").write_text('"pass_rate": 1')
        proposer = Mock()
        with patch.object(jev, "_request", side_effect=response_for) as request:
            result = self.select(proposer)
        self.assertTrue(result.ok)
        self.assertEqual((result.selector, result.chosen_label), ("jev", "agg2"))
        proposer.call.assert_not_called()
        payload = request.call_args.args[0]
        self.assertEqual(set(payload["state"]["candidates"]), {"A", "B"})
        self.assertEqual(
            sorted(payload["state"]["candidates"].values()),
            ["Total: 105.25\n", "Total: 85.15\n"],
        )
        self.assertEqual(
            payload["state"]["source"], {"ledger.csv": (source / "ledger.csv").read_text()}
        )
        self.assertEqual(len(payload["questions"]), 10)
        for question_id, question in payload["questions"].items():
            self.assertIn("candidates." + question_id.split("_")[0], question["instructions"])
        self.assertAlmostEqual(result.usd, 0.0000042)
        audit = self.root / "selection/agg2.jev.json"
        self.assertEqual(json.loads(audit.read_text())["response"]["model"], "jev-1.14.0")
        self.assertNotIn("test-key-never-send", audit.read_text())
        self.assertNotIn("agg1", json.dumps(payload))
        saved = json.loads((self.root / "selection/agg2/selection.json").read_text())
        self.assertEqual(saved["jev"]["usage"]["input_tokens"], 100)

    def test_uncertain_or_invalid_answers_use_agent_without_showing_jev_judgments(self):
        def low_confidence(response):
            next(iter(response["answers"].values()))["confidence"] = 0.1

        def missing_answer(response):
            response["answers"].pop(next(iter(response["answers"])))

        def invalid_number(response):
            next(iter(response["answers"].values()))["score"] = 3.0

        def invalid_distribution(response):
            next(iter(response["answers"].values()))["probabilities"] = {
                "0": 0.9,
                "1": 0.9,
                "2": 0.9,
            }

        def ties(response):
            for answer in response["answers"].values():
                answer.update(score=2.0, probabilities={"0": 0.0, "1": 0.0, "2": 1.0})

        for mutate in (low_confidence, missing_answer, invalid_number, invalid_distribution, ties):
            with self.subTest(mutate=mutate.__name__):

                def respond(payload):
                    response = response_for(payload)
                    mutate(response)
                    return response

                def agent(mission, root, label):
                    self.assertEqual(list(Path(root).rglob("*.json")), [])
                    return self.proposer.call(mission, root, label)

                proposer = Mock()
                proposer.call.side_effect = agent
                with patch.object(jev, "_request", side_effect=respond):
                    result = self.select(proposer)
                self.assertTrue(result.ok)
                self.assertEqual((result.selector, result.chosen_label), ("proposer", "agg2"))
                self.assertFalse(
                    result.fallback
                )  # Successful agent selection is not a failed task.
                self.assertTrue(result.jev["fallback_reason"])
                self.assertAlmostEqual(result.usd, 0.0000042)
                proposer.call.assert_called_once()

    def test_unsupported_or_oversized_evidence_does_not_call_api(self):
        with patch.object(jev, "_request") as request:
            binary = self.adapter.task_files("ledger-1") / "table.xlsx"
            binary.write_bytes(b"PK\x00\xff")
            result = self.select()
            self.assertIn("non-text", result.jev["fallback_reason"])
            binary.unlink()
            with patch.dict(os.environ, {"HARNESS2_JEV_MAX_REQUEST_BYTES": "20"}):
                result = self.select()
            self.assertIn("exceeds", result.jev["fallback_reason"])
            image = self.root / "preview.png"
            image.write_bytes(b"fake image")
            self.adapter.deliverable_assets = lambda root: [image]
            result = self.select()
            self.assertIn("attachments", result.jev["fallback_reason"])
        request.assert_not_called()
        self.assertEqual((result.ok, result.chosen_label), (True, "agg2"))

    def test_service_failure_preserves_existing_selection_failure_contract(self):
        with patch.object(jev, "_request", side_effect=ValueError("service unavailable")):
            result = self.select()
        self.assertTrue(result.ok)
        self.assertEqual(result.jev["fallback_reason"], "service unavailable")
        from harness2.core.proposer import ProposerError

        failed = Mock()
        failed.call.side_effect = ProposerError("agent failed", attempts=1, cost_usd=3)
        with patch.object(jev, "_request", side_effect=ValueError("service unavailable")):
            result = self.select(failed)
        self.assertFalse(result.ok)
        self.assertTrue(result.fallback)
        self.assertEqual((result.chosen_label, result.usd), ("agg1", 3))

    def test_single_final_and_default_backend_make_no_jev_request(self):
        with patch.object(jev, "_request") as request:
            single = selection.trivial_selection(
                "ledger-1",
                1,
                "agg",
                {"agg2": self.candidates["agg2"]},
                adapter=self.adapter,
                work_dir=self.root,
            )
            with patch.dict(os.environ, {"HARNESS2_SELECTOR": "proposer"}):
                normal = self.select()
        self.assertEqual(single.selector, "trivial")
        self.assertEqual(normal.selector, "proposer")
        self.assertTrue(single.ok and normal.ok)
        request.assert_not_called()

    def test_config_validation_and_resume_identity(self):
        for value in ("nan", "inf", "-0.1", "1.1", "invalid"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"HARNESS2_JEV_MIN_CONFIDENCE": value}),
            ):
                with self.assertRaises(config.ConfigError):
                    config.selector_settings()
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}):
            with self.assertRaisesRegex(config.ConfigError, "TYPESAFE_API_KEY"):
                jev.validate()
        args = run_sweep._build_parser().parse_args(["--bench", "jb", "--k", "2"])
        namespace = self.root / "identity"
        with patch.object(run_sweep, "_stream_for", return_value=({}, [])):
            run_sweep._check_identity(self.adapter, args, namespace)
            run_sweep._check_identity(self.adapter, args, namespace)
            for override in (
                {"HARNESS2_JEV_MODEL": "jev-latest"},
                {"HARNESS2_SELECTOR": "proposer"},
            ):
                with patch.dict(os.environ, override):
                    with self.assertRaisesRegex(ValueError, "different models or tasks"):
                        run_sweep._check_identity(self.adapter, args, namespace)

    def test_both_recursion_modes_keep_reflection_and_resume(self):
        def respond(payload):
            response = response_for(payload)
            # Both composed outputs are correct in the deterministic workflow. Choose
            # one to exercise adoption and reflection after a successful Jev decision.
            for question_id, answer in response["answers"].items():
                if question_id.startswith("B_"):
                    answer.update(score=0.0, probabilities={"0": 1.0, "1": 0.0, "2": 0.0})
            return response

        kwargs = dict(
            adapter=self.adapter, prop=self.proposer, runs_root=self.adapter.spec.runs_root
        )
        kwargs["libraries"] = {
            mode: Library(
                self.root / "library" / mode, "accounting", self.adapter.substrate.surface()
            )
            for mode in ("agg", "seq")
        }
        with patch.object(jev, "_request", side_effect=respond) as request:
            result = schedule.run_task("ledger-1", 2, **kwargs)
            self.assertEqual(result.status, "ok")
            self.assertEqual([s["selector"] for s in result.selections.values()], ["jev", "jev"])
            self.assertEqual(request.call_count, 2)
            self.assertTrue(all(s["experience"] for s in result.selections.values()))
            self.assertEqual(
                len(
                    [call for call in self.proposer._calls if call.label.startswith("experience.")]
                ),
                2,
            )
            resumed = schedule.run_task("ledger-1", 2, **kwargs)
            self.assertEqual(resumed.status, "skipped")
            self.assertEqual(request.call_count, 2)
        with patch.dict(os.environ, {"HARNESS2_SELECTOR": "proposer"}):
            with self.assertRaisesRegex(ValueError, "different selector"):
                schedule.run_task("ledger-1", 2, **kwargs)
            summary_path = self.adapter.spec.runs_root / "demo-oc-agg2/ledger-1/summary.json"
            summary = json.loads(summary_path.read_text())
            summary["status"] = "partial"
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "different selector"):
                schedule.run_task("ledger-1", 2, **kwargs)

    @unittest.skipUnless(
        importlib.util.find_spec("typesafe_sdk"), "install .[jev] for SDK contract test"
    )
    def test_real_sdk_request_and_response_contract_without_network(self):
        import httpx2
        import typesafe_sdk

        real_client = typesafe_sdk.TypeSafeClient
        seen = []

        def handle(request):
            self.assertEqual(str(request.url), "https://api.typesafe.ai/v1/systemone")
            self.assertEqual(request.headers["authorization"], "Bearer test-key-never-send")
            payload = json.loads(request.content)
            seen.append(payload)
            return httpx2.Response(200, json=response_for(payload))

        def client(**kwargs):
            return real_client(**kwargs, transport=httpx2.MockTransport(handle))

        with patch.object(typesafe_sdk, "TypeSafeClient", side_effect=client):
            result = self.select()
        self.assertEqual((result.ok, result.selector, result.chosen_label), (True, "jev", "agg2"))
        self.assertEqual(len(seen), 1)

        def rejected(request):
            return httpx2.Response(401, json={"error": "private-service-error"})

        def rejected_client(**kwargs):
            return real_client(**kwargs, transport=httpx2.MockTransport(rejected))

        with patch.object(typesafe_sdk, "TypeSafeClient", side_effect=rejected_client):
            result = self.select()
        self.assertEqual((result.ok, result.selector), (True, "proposer"))
        self.assertIn("TypeSafe request failed", result.jev["fallback_reason"])
        self.assertNotIn("private-service-error", json.dumps(result.to_dict()))


if __name__ == "__main__":
    unittest.main()
