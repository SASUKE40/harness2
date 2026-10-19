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

"""Run recursive harnessing on a local traffic-sign labeling stream and export it for the UI.

`--jev` picks the final-output selector. `offline` (the default) uses a deterministic
stand-in for the Jev API inside the real Jev request, validation, and audit path; no API call
is made. `live` calls TypeSafe; `off` uses the agent selector only.

`--executor` picks who labels the images. `labeler` (the default) runs the harness's own
scripts on the SVG scenes. `jev` renders each scene to PNG and asks Jev to see it, shaped by
the harness's prompt rules and tools (see harness2.vision); it always calls TypeSafe.

`--source real` swaps the synthetic scenes for 24 street photos with published annotations
(see harness2.real_images). Photos need `--executor jev --jev live`; the final selector then
receives the photos as image evidence.

Live Jev needs `.[jev]` and TYPESAFE_API_KEY.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

from .. import config, vision
from ..core import jev
from ..core.schedule import mode_token, run_domain
from ..image_demo import DOMAIN, ImageLabelAdapter, ImageLabelProposer, export_bundle, offline_jev


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/image-demo"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--mode", choices=("parallel", "sequential", "both"), default="both")
    parser.add_argument("--jev", choices=("offline", "live", "off"), default="offline")
    parser.add_argument("--executor", choices=("labeler", "jev"), default="labeler")
    parser.add_argument("--source", choices=("synthetic", "real", "aircraft"), default="synthetic")
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        default=None,
        help="where photos are downloaded once (default: dependencies/datasets/<source>)",
    )
    parser.add_argument(
        "--export", type=Path, default=None, help="also copy rsi.json here (e.g. web/public)"
    )
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be positive")
    if args.source != "synthetic":
        if args.executor != "jev" or args.jev != "live":
            parser.error(f"--source {args.source} needs --executor jev --jev live")
        os.environ["HARNESS2_JEV_IMAGES"] = "1"
    cache = args.dataset_cache or config.ROOT / "dependencies" / "datasets" / (
        "sign-road" if args.source == "real" else "fgvc-aircraft"
    )
    os.environ["HARNESS2_SELECTOR"] = "proposer" if args.jev == "off" else "jev"
    # k finals of six SVG scenes outgrow the conservative 28 KB default; Jev's own request
    # limit is far larger. An explicit setting still wins.
    os.environ.setdefault("HARNESS2_JEV_MAX_REQUEST_BYTES", "48000")
    try:
        if args.jev == "live":
            jev.validate()
        if args.executor == "jev":
            vision.validate()
    except config.ConfigError as exc:
        parser.error(str(exc))
    root = args.output.expanduser().resolve()
    adapter = ImageLabelAdapter(
        root, executor=args.executor, source=args.source, dataset_cache=cache
    )
    proposer = ImageLabelProposer(adapter.spec.channels["proposer"])
    if args.source != "synthetic":
        # Photos have no deterministic stand-in: when Jev's selection defers, the fallback
        # asks Jev to confirm each candidate's answer instead.
        proposer.cache_dir = root / "jev-cache"
    if args.source == "aircraft":
        from ..aircraft import claim_question

        proposer.claim_question = claim_question
    modes = ("agg", "seq") if args.mode == "both" else (mode_token(args.mode),)
    with offline_jev() if args.jev == "offline" else contextlib.nullcontext():
        result = run_domain(
            DOMAIN,
            adapter.tasks(),
            args.k,
            adapter=adapter,
            prop=proposer,
            runs_root=adapter.spec.runs_root,
            library_root=root / "library" / f"k{args.k}",
            modes=modes,
        )
    bundle = export_bundle(adapter, args.k, modes, jev_mode=args.jev)
    text = json.dumps(bundle, indent=1, ensure_ascii=False) + "\n"
    filename = {"synthetic": "rsi.json", "real": "rsi-real.json", "aircraft": "rsi-aircraft.json"}[
        args.source
    ]
    targets = [root / filename]
    if args.export:
        dest = args.export.expanduser()
        targets.append(dest / filename if dest.is_dir() or not dest.suffix else dest)
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(
        f"Image-labeling demo: {result['done']} tasks completed; {result['failed']} failed. "
        + " ".join(str(p) for p in targets)
    )
    return int(result["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
