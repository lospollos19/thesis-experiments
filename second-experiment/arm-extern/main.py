"""CLI entry point wiring camera -> pipeline -> optional display.

This is the only module that imports the display layer, keeping the headless/GUI
separation clean: ``pipeline.py`` never touches UI code.
"""

from __future__ import annotations

import argparse
import sys
import time

from motion_pipeline.camera.camera_interface import RealCamera
from motion_pipeline.pipeline import Pipeline, PipelineStats
from motion_pipeline.processing.gpu_processor import GPUProcessor
from motion_pipeline.processing.motion_detector import MotionDetector
from motion_pipeline.simulation.data_stream import SimulatedCamera

TARGET_FPS = 30.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU motion detection pipeline")
    parser.add_argument("--source", choices=["camera", "simulation"], default="simulation")
    parser.add_argument("--display", action="store_true", help="enable GUI window")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = infinite")
    parser.add_argument("--device", type=int, default=0, help="camera device id")
    parser.add_argument("--threshold", type=int, default=25)
    parser.add_argument("--min-area", type=int, default=500)
    return parser


def _make_camera(args: argparse.Namespace):
    if args.source == "camera":
        return RealCamera(device_id=args.device)
    # Simulation: finite when max_frames > 0 so the demo terminates; otherwise a
    # long deterministic loop with a recurring motion event.
    num_frames = args.max_frames if args.max_frames > 0 else 300
    motion_frames = list(range(10, 16)) + list(range(100, 106)) + list(range(200, 206))
    return SimulatedCamera(num_frames=num_frames, motion_frames=motion_frames)


def run(argv: list[str] | None = None) -> PipelineStats:
    """Run the pipeline for the given CLI args and return the final stats."""
    args = build_parser().parse_args(argv)

    processor = GPUProcessor()
    detector = MotionDetector(processor, threshold=args.threshold, min_area=args.min_area)
    camera = _make_camera(args)
    pipeline = Pipeline(camera, detector)

    display = None
    if args.display:
        from motion_pipeline.ui.display import Display

        display = Display(backend="PyCUDA" if processor.backend == "pycuda" else "NumPy")

    try:
        for frame, result in pipeline.run(max_frames=args.max_frames):
            if display is not None:
                if not display.show(frame, result):
                    break
                _pace_loop()
            else:
                status = "MOTION" if result.motion_detected else "STATIC"
                print(
                    f"frame {pipeline.stats.total_frames:>5} | {status:6} | "
                    f"diff={result.diff_score:6.2f}"
                )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if display is not None:
            display.release()

    _print_summary(pipeline.stats)
    return pipeline.stats


def _pace_loop() -> None:
    time.sleep(1.0 / TARGET_FPS)


def _print_summary(stats: PipelineStats) -> None:
    print("\n=== PipelineStats ===")
    print(f"total_frames      : {stats.total_frames}")
    print(f"motion_events     : {stats.motion_events}")
    print(f"avg_processing_ms : {stats.avg_processing_ms:.2f}")
    print(f"backend           : {stats.backend}")


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
