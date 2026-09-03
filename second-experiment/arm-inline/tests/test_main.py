"""CLI smoke test: run main.run() headless with the simulation source."""

from __future__ import annotations

from motion_pipeline.pipeline import PipelineStats
from motion_pipeline.processing.gpu_processor import GPUProcessor

import main as main_module


def test_main_simulation_headless(capsys):
    stats = main_module.run(["--source", "simulation", "--max-frames", "10"])
    assert isinstance(stats, PipelineStats)
    assert stats.total_frames == 10
    # Not a hardcoded "numpy": this test is unmarked, so it runs on the Orin too, where
    # the honest expectation is "pycuda". What the smoke test actually asserts is that
    # `run()` reports the backend it really used — so the reference is the processor's
    # own detection in this environment, not the environment CI happened to have first.
    assert stats.backend == GPUProcessor().backend

    out = capsys.readouterr().out
    assert "PipelineStats" in out


def test_parser_defaults():
    args = main_module.build_parser().parse_args([])
    assert args.source == "simulation"
    assert args.max_frames == 0
    assert args.threshold == 25
    assert args.min_area == 500
    assert args.display is False
