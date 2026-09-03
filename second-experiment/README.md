# Second experiment — GPU motion pipeline

A GPU-accelerated real-time motion detection pipeline, built as the research
prototype for the RQ2 study on regression test selection in a PyCUDA codebase.
It is not production software. It runs on an NVIDIA AGX Orin, and it also runs
end-to-end without any camera, GPU or display, through a deterministic
simulated data stream and a NumPy fallback.

## The two configurations

The variable under study is how the CUDA-C sources are stored.

| Directory | CUDA-C sources |
|---|---|
| `arm-inline/` | string literals inside `processing/kernels.py` |
| `arm-extern/` | separate `.cu` files under `processing/cuda/` |

Each is a full, independently runnable copy of the pipeline. They differ in
eight files and nowhere else:

```
src/motion_pipeline/processing/kernels.py
src/motion_pipeline/processing/cuda/*.cu        (arm-extern only)
src/motion_pipeline/processing/cuda/common.cuh  (arm-extern only)
tests/test_kernels.py
```

```bash
diff -rq arm-inline arm-extern
```

`processing/kernels.py` is the single point of access to CUDA-C: everything
goes through `get_kernel_source()` and `get_module_source()`, which is what
lets the two configurations differ in one place.

## Layout

Identical in both arms:

```
main.py                      CLI entry point (camera -> pipeline -> display)
src/motion_pipeline/
  camera/                    BaseCamera abstraction, OpenCV RealCamera
  simulation/                SimulatedCamera, deterministic synthetic frames
  processing/                GPUProcessor (PyCUDA or NumPy), kernels, MotionDetector
  ui/                        Display (OpenCV overlay, headless-safe)
  pipeline.py                orchestrator, never imports UI code
tests/                       one file per module, plus integration and workload tiers
tools/                       the selector and the measurement tooling, listed below
bench/                       standalone benchmarks, not collected by pytest
.github/workflows/           CI, including the jobs that run on the Orin
```

Shared by both arms, one level up:

```
results/                     artefacts published by the CI runs
logs/                        step-by-step logs of those same runs
```

## The selector and the measurement tooling

Everything the study runs lives in each arm's `tools/`, next to the tracer in
`src/motion_pipeline/processing/_trace.py`. Nothing is kept elsewhere.

| File | What it does |
|---|---|
| `kernel_select.py` | **The kernel-aware selector — condition C.** `manifest` hashes the kernel sources of a tree; `select` emits the node ids to run. Its four rules are documented at the top of the file: R1 an edge to a changed kernel selects the test, R2 a collected id absent from the map is selected, R3 any kernel change selects the tests declared `no_kernel_launch`, R4 a change to the shared geometry selects everything. The final selection is the union with testmon's. |
| `_trace.py` | Records which kernels each test launches, by instrumenting the launch sites. |
| `merge_kernel_deps.py` | Unions the traced sessions into one test-to-kernel map. |
| `kernel_map_report.py` | Reports on that map: is it exploitable, and is it complete. |
| `mutate.py` | The mutation corpus and the tool that applies one mutation to a tree. |
| `ground_truth.py` | Measures `D(m)`: which tests each mutation actually breaks. |
| `ground_truth_report.py` | Aggregates the per-arm `D(m)` and rules on their validity. |
| `campaign.py` | Runs one condition over a chunk of the corpus: select, then time. |
| `campaign_report.py` | Crosses the selections with `D(m)`: violations and occupancy. |
| `repeat_report.py` | Mean, spread and saving with propagated uncertainty, from repeated timings. |
| `checkpoint_testmon_db.py` | Folds testmon's write-ahead log into its database before archiving. |

Each takes `--help`. The map the campaign used is shipped at
`results/run-31558337740/verdict-q2q3/kernel_deps.json`, with the raw traces
that produced it.

## Installing

OpenCV ships in two mutually exclusive variants, so install only one.

```bash
cd arm-inline                            # or arm-extern
poetry install --extras gui              # local dev, OpenCV window
poetry install --extras headless         # no display
poetry install --extras "headless gpu"   # adds PyCUDA, needs nvcc on the machine
```

`GPUProcessor` picks its backend at construction: CUDA-C kernels when PyCUDA
imports and a device is present, NumPy otherwise. The NumPy path is the
numerical reference the equivalence tests compare against.

## Running

```bash
poetry run python main.py --source simulation --max-frames 100   # headless
poetry run python main.py --source camera --display              # real camera, GUI
```

| Flag | Default | Meaning |
|---|---|---|
| `--source` | `simulation` | `camera` or `simulation` |
| `--display` | off | enable the OpenCV window |
| `--max-frames` | `0` | frame budget, `0` means run until interrupted |
| `--device` | `0` | camera device id |
| `--threshold` | `25` | per-pixel motion threshold |
| `--min-area` | `500` | changed pixels needed to count as motion |

## Tests

```bash
cd arm-inline                            # or arm-extern
poetry install --extras headless
poetry run pytest tests/ -m "not gpu" -q
```

The `gpu` marker covers the kernel tests. They need a CUDA device and `nvcc`,
and are skipped without them:

```bash
poetry install --extras "headless gpu"
poetry run pytest tests/ -q
```

`tests/workload.py` holds all suite sizing. Tests are sized by a fixed pixel
budget rather than a fixed frame count, and `CachedCamera` synthesises a small
set of frames per shape and cycles over them, so stream length is decoupled
from synthesis cost.

`tests/conftest.py` prints the observed kernel-vs-NumPy deviations at the end
of every GPU run, next to the tolerance each one is checked against.

## Docker

Multi-stage: a CPU `base` stage and a CUDA `gpu` stage for the Orin. The `gpu`
stage uses an `l4t-cuda` *devel* image, since `SourceModule` shells out to
`nvcc` at run time.

```bash
docker build --target base -t motion-pipeline:cpu .
docker build --target gpu  -t motion-pipeline:gpu .
```

## results/ and logs/

The CI runs that produced the measurements, one directory per run, named by
its GitHub Actions run id:

```
results/run-<id>/<artefact-name>/    artefacts the run published
logs/run-<id>/                       the run's logs, one file per step
```

The campaign artefacts contain the raw per-mutation JSON, the selection and
timing logs, a `SHA256SUMS` file and the commit the run was built from. The
tooling that turns them into a report is in each arm's `tools/`:

```bash
cd arm-extern
python tools/repeat_report.py ../results/run-<id>/campaign-C-*/campaign-C-*.json
python tools/campaign_report.py --ground-truth <gt.json> --campaign C=<files>
```

All GPU measurements were taken on a self-hosted `jetson-orin` runner. The
workflows under `.github/workflows/` reference it by label, and the R3
workflow fetches two artefacts by the run id they were produced under, so
re-running the campaign in a fresh repository needs those ids replaced.
Development happened on Apple Silicon, where no CUDA build is possible.
