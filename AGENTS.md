# Agent notes

## Verification

From the repository root (no virtualenv required; uses uv with Python 3.12):

```bash
uv run --python 3.12 --no-project --with ruff ruff check harness2 tests
uv run --python 3.12 --no-project --with-editable . python -m unittest discover -s tests
# Also run the Jev SDK and vision-executor tests (skipped without the extras):
uv run --python 3.12 --no-project --with-editable . --with typesafe-sdk==0.7.5 \
  --with resvg-py==0.5.0 python -m unittest discover -s tests
```

Format new Python files with `ruff format` (line length 100).

## RSI dashboard (`web/`)

Vite + React + TypeScript; it reads `web/public/rsi.json`.

```bash
uv run --python 3.12 --no-project --with-editable . \
  python -m harness2.cli.image_demo --output runs/image-demo --export web/public
cd web && npm install && npm run build   # tsc --noEmit + vite build
npm run dev                              # http://localhost:5173
```

Views: the live demo is the default (`?mode=agg|seq&step=<timeline index>` to jump);
the analytics dashboard is `?view=dashboard&mode=agg|seq&task=<1-based>&node=<stage label>`.

- `harness2/image_demo.py`: workload, scenes, harness rendering (label.py + one tool file per
  fix + prompt rules), scripted editor/composers, offline Jev stand-in, exporter.
- `harness2/vision.py`: `--executor jev` (Jev labels SVG tiles or whole photos; needs a
  TypeSafe key, never committed: pass it as an environment variable).
- `harness2/real_images.py`: `--source real`, 24 curated street photos (CC BY 4.0) from
  the Hugging Face dataset `ayoubsa/Sign_Road_Detection_Dataset`. It exports
  `web/public/rsi-real.json`, which the page loads by default (`?data=synthetic` for the other).
- `harness2/aircraft.py`: `--source aircraft`, FGVC-Aircraft regional types (non-commercial
  images; `web/public/rsi-aircraft.json` is git-ignored and must not be committed).
- Jev image limits (measured): longest side up to about 1568 px, shortest side above 256 px;
  `vision.normalize` enforces this.
- Image evidence for Jev selection is opt-in via `HARNESS2_JEV_IMAGES=1` (set for photo sources).
- Tests: `tests/test_image_demo.py`, `tests/test_vision.py`, `tests/test_real.py`; vision and
  photo tests need the `.[jev]` extras (`typesafe-sdk`, `resvg-py`, `pillow`).
- The scripted dynamics are deliberate: every task must keep Jev deciding (distinct finals),
  and the first task's sequential rounds must improve monotonically; tests check both.
