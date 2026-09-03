# First experiment — CuPBoP SIL/HIL

Mutation testing of a CUDA C++ benchmark (`srad_v2`, Rodinia) in two
environments, to compare what each detects:

- **SIL** — CuPBoP executing the CUDA code on an x86 CI runner with no GPU
- **HIL** — the same code compiled with `nvcc` and run natively on an NVIDIA AGX Orin

## Layout

```
build_cupbop.py                    renders the CuPBoP SIL image Dockerfile
docker/cupbop-sil/Dockerfile       the committed image recipe
run_srad.py                        host-side driver
experiments/srad_v1/               Rodinia srad v1, unmodified
experiments/srad_v2/               the subject
  srad.cu, srad_kernel.cu, ...     single-TU source
  translate_srad.sh                CuPBoP toolchain: compile -> translate -> llc -> link
  compare_outputs.sh               element-by-element output diff
  baseline/                        SIL reference output, its provenance and the tolerance
  mutation/
    generate_mutants.py            source-level mutant generator, no GPU needed
    run_campaign.sh                one environment per invocation, loops over all mutants
    analyze_campaign.py            joins SIL and HIL verdicts, reports the gap per operator
  tests/host_test.cpp              host-side unit test
diagnostics/racecheck/             compute-sanitizer racecheck harness and the 9 mutants
.github/workflows/                 CI, including the jobs that run on the Orin
results/                           artifacts published by the CI runs
logs/                              step-by-step logs of those same runs
```

## Image

Pinned CuPBoP commit `508bd62e928bea3b5f0633c8fa63b5f42f3b4da0`, base
`nvidia/cuda:11.7.1-devel-ubuntu22.04`, LLVM 14.

```bash
poetry install
poetry run python build_cupbop.py                  # render only, never build locally
diff docker/cupbop-sil/Dockerfile Dockerfile.generated
```

Building the image locally would trigger amd64 emulation. The real build and
push run in CI on an x86 runner.

## Campaign

```bash
python experiments/srad_v2/mutation/generate_mutants.py    # no GPU needed
experiments/srad_v2/mutation/run_campaign.sh sil           # in the CuPBoP image
experiments/srad_v2/mutation/run_campaign.sh hil           # on the Orin
python experiments/srad_v2/mutation/analyze_campaign.py results_sil.json results_hil.json
```

Each platform is scored against its own unmutated baseline. A mutant is killed
when the output diverges beyond the tolerance set in
`experiments/srad_v2/baseline/NOISE_FLOOR.md`.

## results/ and logs/

One directory per CI run, named by its GitHub Actions run id:

```
results/run-<id>/<artifact-name>/    artifacts the run published
logs/run-<id>/                       the run's logs, one file per step
```

The campaign artifacts are `results_sil.jsonl` and `results_hil.jsonl`, one
verdict per mutant, readable by `analyze_campaign.py`:

```bash
python experiments/srad_v2/mutation/analyze_campaign.py \
  results/run-30713970653/results-sil/results_sil.jsonl \
  results/run-30665048698/results-hil/results_hil.jsonl
```

`results/run-30718000195/racecheck-logs/` holds the racecheck run attempted on
the Orin. Its ten logs are byte-identical and report `GPU debugging features are
disabled`: the Tegra iGPU does not support the debug features
`compute-sanitizer --tool racecheck` needs, so no race measurement came out of
it. The instrument was moved to a discrete GPU, off CI.

All HIL measurements were taken on a self-hosted Orin runner. Development
happened on Apple Silicon, where neither path can be built.
