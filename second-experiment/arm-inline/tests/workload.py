"""Shared sizing for the workload suite.

The suite exists to give the Orin enough real work that a test-selection strategy
has something measurable to reduce. Everything that controls its duration lives
here, so recalibrating means editing one file.

Sizing constraints, in the order they bind:

* ``SimulatedCamera`` materialises every frame up front. At 720p BGR a frame is
  2.7 MB, so the resident cost is ``UNIQUE_FRAMES``, not the streamed length. The Orin
  shares that memory with the iGPU, which is why 1080p is not in ``RESOLUTIONS``.
* The NumPy reference path convolves with a Python loop over the kernel taps, so
  equivalence tests grow as ``resolution * taps``. They stay on the smaller
  resolutions; bulk load comes from the throughput tests, which run device-only.

Calibration. Measured on the Orin at 120 streamed frames: 73 gpu tests in 29.6 s,
of which roughly 8 s is one-off frame synthesis. Raising the stream to 1200 frames
gave 322.8 s, from which the marginal cost works out at ~12.2 ms per megapixel
streamed. ``PIXEL_BUDGET`` is sized against that figure.

Why a pixel budget and not a frame count. A fixed frame count makes a test's cost
proportional to its resolution: measured at 1200 frames, 720p tests cost 13.5 s and
240p tests 1.3 s, and the 25 slowest of 73 tests carried 86% of the runtime. A
selection strategy that happens to drop the heavy tail then shows a large saving
that says nothing about the strategy. Equalising cost makes "fraction of tests
selected" and "fraction of time saved" comparable quantities, so a gap between them
is a property of the selection rather than an artefact of this file.
"""

from __future__ import annotations

from functools import cache

from motion_pipeline.simulation.data_stream import SimulatedCamera

#: Distinct frames synthesised and held in memory per stream shape. Drives memory.
UNIQUE_FRAMES = 120

#: Megapixels each workload test streams, whatever its resolution. Drives duration.
#: 200 Mpx gave 167 s of gpu suite on the Orin, just under the 3 minute floor; 280
#: targets the middle of the 3-5 minute band. Roughly 15 s of the total does not scale
#: with this, almost all of it frame synthesis; the reference-computing kernel tests
#: are a rounding error by comparison (their NumPy side measures 3-11 ms per
#: resolution). Re-derive the split from --durations after any change here.
PIXEL_BUDGET_MPX = 280.0

#: Safety rail on the derived frame count, not a calibration input. An earlier value
#: of 2500 was set on the assumption that per-frame overhead stops scaling with pixels
#: at low resolution; the measurements contradict it (1.3 ms/frame at 240p against
#: 7.2 ms at 480p, close to the 4x pixel ratio), and the cap was only holding 240p
#: below its budget. It exists now to catch an absurd budget, nothing more.
MAX_FRAMES_PER_RUN = 6000

#: (height, width). Capped at 720p by the pre-generation cost described above.
RESOLUTIONS: tuple[tuple[int, int], ...] = ((240, 320), (480, 640), (720, 1280))

#: Resolutions cheap enough to also compute the NumPy reference for. (478, 638) is
#: deliberately not a multiple of TILE_W/TILE_H: every entry in RESOLUTIONS is, so
#: without it the only misaligned grid in the suite is the 77x101 frame in
#: test_kernels.py, and a boundary bug that only shows up at scale would be missed.
REFERENCE_RESOLUTIONS: tuple[tuple[int, int], ...] = ((240, 320), (478, 638), (480, 640))

BLUR_KERNELS: tuple[int, ...] = (3, 5, 9, 15)
THRESHOLDS: tuple[int, ...] = (15, 25, 40)
CHANNELS: tuple[int, ...] = (1, 3)


def frames_for(resolution: tuple[int, int]) -> int:
    """Frames needed to spend ``PIXEL_BUDGET_MPX`` at this resolution.

    Equalises per-test cost across resolutions, so a test's weight in the total no
    longer encodes which resolution it happens to use.
    """
    megapixels = (resolution[0] * resolution[1]) / 1e6
    return max(1, min(MAX_FRAMES_PER_RUN, round(PIXEL_BUDGET_MPX / megapixels)))


def motion_schedule(num_frames: int = UNIQUE_FRAMES) -> list[int]:
    """Frame indices carrying a synthetic motion event, within the unique set.

    Three bursts spread across the base stream, so a run exercises both the
    motion and the static branch of the detector. Cycling repeats the bursts,
    which is a fair likeness of a recording with recurring events.
    """
    step = max(num_frames // 4, 1)
    return [i for start in (step, 2 * step, 3 * step) for i in range(start, start + 5)]


@cache
def _unique_frames(resolution, channels, seed):
    return SimulatedCamera(
        resolution=resolution,
        num_frames=UNIQUE_FRAMES,
        motion_frames=motion_schedule(),
        channels=channels,
        seed=seed,
    )._generate()


class CachedCamera(SimulatedCamera):
    """A :class:`SimulatedCamera` that synthesises at most ``UNIQUE_FRAMES`` per shape.

    Two problems are solved here, both of which would otherwise distort what RQ2
    measures.

    Frame synthesis is ``rng.normal`` over the whole frame: 2.5 s for 120 frames of
    720p BGR on a developer laptop, and several times that on the Orin's CPU cores.
    Paid per test it would dominate the measurement, so it is cached per shape.

    Stream length is then decoupled from memory. ``num_frames`` frames are served by
    cycling over the cached set, so the run can be made long enough to occupy the
    device without the resident footprint growing with it. The list holds references,
    not copies.

    Streams stay exactly as deterministic as the parent class. Frames are shared
    between runs and must be treated as read-only — nothing in the pipeline writes
    to them.
    """

    def _generate(self):
        base = _unique_frames(tuple(self.resolution), self.channels, self.seed)
        return [base[i % len(base)] for i in range(self.num_frames)]
