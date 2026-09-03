"""Tests for the CUDA-C kernel sources and their numerical equivalence to NumPy.

The source-level tests run everywhere (the sources are plain strings). The
``@pytest.mark.gpu`` equivalence tests need a real device and are skipped otherwise.

Tolerances. The NumPy reference and the kernels perform the *same* float32 arithmetic
but not in the same order, and the GPU may contract ``a * b + c`` into a single FMA
(no intermediate rounding). The discrepancy is therefore a few float32 ULPs of the
accumulator, and tolerances are absolute, against the 0-255 intensity range each
primitive works in.

The first set was derived analytically, before any device was available. Measured
on the Orin (sm_87, CUDA 13.2) they turned out 65x to 300x looser than needed, which
is wide enough to let a real kernel regression through. They are now set at roughly
8x the observed deviation — still ample room for driver and hardware variation,
without being a tolerance that passes regardless:

* ``grayscale`` -- 3-term dot product on values <= 255. Observed 1.526e-05, which is
  one ULP of float32 at 255, exactly the FMA contraction -> ``atol=1e-4``;
* ``blur`` -- two separable passes, up to ``kernel_size`` accumulations each.
  Observed 3.05e-05 at 3 taps rising to 6.10e-05 at 15, i.e. 2 to 4 ULP growing with
  the accumulation length -> ``atol=5e-4``;
* ``absdiff`` -- one subtraction, exactly representable. Observed exactly 0.0, so the
  tolerance is nominal -> ``atol=1e-5``;
* ``threshold`` -- integer output of an exact comparison -> compared bit-exactly.

Comparisons go through ``conftest.assert_within``, never ``np.allclose``. The latter
tests ``atol + rtol * |ref|`` with ``rtol=1e-5`` by default, which on 0-255 intensities
adds 2.55e-3 and made every tolerance here between 6x and 256x looser than the value
written next to it. It prints the observed deviations at the end of every GPU run, so
these numbers can be re-checked rather than trusted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from motion_pipeline.processing import kernels
from motion_pipeline.processing.gpu_processor import GPUProcessor

from conftest import assert_within

ATOL_GRAYSCALE = 1e-4
ATOL_BLUR = 5e-4
ATOL_ABSDIFF = 1e-5


@pytest.fixture
def cpu_processor() -> GPUProcessor:
    """NumPy reference processor (independent of the ``gpu_processor`` fixture)."""
    return GPUProcessor(force_cpu=True)


@pytest.fixture
def frames() -> tuple[np.ndarray, np.ndarray]:
    """Two deterministic BGR frames with a size that is not a multiple of the tile."""
    rng = np.random.default_rng(11)
    shape = (77, 101, 3)  # deliberately not a multiple of TILE_H / TILE_W
    return (
        rng.integers(0, 256, size=shape, dtype=np.uint8),
        rng.integers(0, 256, size=shape, dtype=np.uint8),
    )


# -- source-level tests (no GPU needed) ----------------------------------


def test_every_kernel_source_defines_its_entry_point():
    for name in kernels.KERNEL_NAMES:
        source = kernels.get_kernel_source(name)
        assert f"void {name}(" in source


def test_get_kernel_source_rejects_unknown_name():
    with pytest.raises(KeyError):
        kernels.get_kernel_source("no_such_kernel")


def test_module_source_contains_all_kernels():
    source = kernels.get_module_source()
    for name in kernels.KERNEL_NAMES:
        assert f"void {name}(" in source


def test_max_kernel_size_matches_halo():
    assert kernels.MAX_KERNEL_SIZE == 2 * kernels.MAX_RADIUS + 1


def test_max_blur_contract_is_thirty_three_taps():
    """Pinned to a literal, deliberately, and not derived from ``MAX_RADIUS``.

    Every other assertion about the tile geometry is written in terms of the constant
    it is checking, so halving ``MAX_RADIUS`` leaves the whole suite green while the
    largest supported blur silently drops from 33 taps to 17 — a caller asking for 33
    then gets a ``ValueError`` where it used to get a result. That is a behaviour
    change no test could see, which the mutation corpus (``tools/mutate.py``) exposed.

    Only the observable half of the contract is pinned. ``TILE_W`` and ``TILE_H`` are
    internal block geometry with no effect on results, so pinning them would turn a
    legitimately equivalent change into a failure.
    """
    assert kernels.MAX_KERNEL_SIZE == 33


# -- external storage (this variant) -------------------------------------


def test_every_kernel_has_its_own_cu_file():
    for name in kernels.KERNEL_NAMES:
        assert (kernels.KERNEL_DIR / f"{name}.cu").is_file()


def test_no_device_code_left_in_the_python_module():
    """The loader must hold no CUDA-C: that is what makes this the external variant."""
    source = Path(kernels.__file__).read_text(encoding="utf-8")
    assert "__global__" not in source
    assert "__shared__" not in source


def test_tile_geometry_comes_from_the_shared_header():
    header = (kernels.KERNEL_DIR / kernels.COMMON_HEADER).read_text(encoding="utf-8")
    for name, value in (
        ("TILE_W", kernels.TILE_W),
        ("TILE_H", kernels.TILE_H),
        ("MAX_RADIUS", kernels.MAX_RADIUS),
    ):
        assert f"#define {name} {value}" in header


def test_convolution_sources_carry_the_header():
    for name in ("conv1d_horizontal", "conv1d_vertical"):
        source = kernels.get_kernel_source(name)
        assert "#define TILE_W" in source, f"{name} would not compile without the defines"


def test_missing_cu_file_reports_the_packaging_cause(monkeypatch, tmp_path):
    monkeypatch.setattr(kernels, "KERNEL_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="must ship with the package"):
        kernels.get_kernel_source("absdiff_f32")


def test_backend_is_numpy_without_device(cpu_processor):
    assert cpu_processor.backend == "numpy"


# -- numerical equivalence (device required) -----------------------------


@pytest.mark.gpu
def test_grayscale_matches_numpy(cuda_processor, cpu_processor, frames):
    frame, _ = frames
    gpu = cuda_processor.to_host(cuda_processor.convert_grayscale(cuda_processor.to_device(frame)))
    ref = cpu_processor.to_host(cpu_processor.convert_grayscale(cpu_processor.to_device(frame)))
    assert_within(gpu, ref, ATOL_GRAYSCALE, "grayscale 77x101")


@pytest.mark.gpu
# An already-2d frame is returned as-is: convert_grayscale exits before the launch.
@pytest.mark.no_kernel_launch
def test_grayscale_passthrough_2d_matches_numpy(cuda_processor, cpu_processor, frames):
    frame = frames[0][:, :, 0]
    gpu = cuda_processor.to_host(cuda_processor.convert_grayscale(cuda_processor.to_device(frame)))
    ref = cpu_processor.to_host(cpu_processor.convert_grayscale(frame))
    assert_within(gpu, ref, ATOL_GRAYSCALE, "grayscale 2d 77x101")


@pytest.mark.gpu
# MAX_KERNEL_SIZE (33 taps, radius 16) is the case where the halo exactly fills the
# shared-memory tile and the cooperative load loop runs its last iteration. An
# off-by-one there is invisible at the small radii the pipeline actually uses.
@pytest.mark.parametrize("kernel_size", [3, 5, 9, kernels.MAX_KERNEL_SIZE])
def test_gaussian_blur_matches_numpy(cuda_processor, cpu_processor, frames, kernel_size):
    gray = frames[0][:, :, 0].astype(np.float32)
    gpu = cuda_processor.to_host(
        cuda_processor.gaussian_blur(cuda_processor.to_device(gray), kernel_size)
    )
    ref = cpu_processor.to_host(cpu_processor.gaussian_blur(gray, kernel_size))
    assert_within(gpu, ref, ATOL_BLUR, f"blur k={kernel_size} 77x101")


@pytest.mark.gpu
def test_absolute_diff_matches_numpy(cuda_processor, cpu_processor, frames):
    a = frames[0][:, :, 0].astype(np.float32)
    b = frames[1][:, :, 0].astype(np.float32)
    gpu = cuda_processor.to_host(
        cuda_processor.absolute_diff(cuda_processor.to_device(a), cuda_processor.to_device(b))
    )
    ref = cpu_processor.to_host(cpu_processor.absolute_diff(a, b))
    assert_within(gpu, ref, ATOL_ABSDIFF, "absdiff 77x101")


@pytest.mark.gpu
def test_threshold_matches_numpy(cuda_processor, cpu_processor, frames):
    arr = frames[0][:, :, 0].astype(np.float32)
    gpu = cuda_processor.to_host(cuda_processor.threshold(cuda_processor.to_device(arr), 25.0))
    ref = cpu_processor.to_host(cpu_processor.threshold(arr, 25.0))
    np.testing.assert_array_equal(gpu, ref)  # exact: integer output of an exact comparison


@pytest.mark.gpu
def test_full_chain_matches_numpy(cuda_processor, cpu_processor, frames):
    """End-to-end: grayscale -> blur -> absdiff -> threshold on both paths."""

    def chain(proc, frame_a, frame_b):
        a = proc.gaussian_blur(proc.convert_grayscale(proc.to_device(frame_a)), 5)
        b = proc.gaussian_blur(proc.convert_grayscale(proc.to_device(frame_b)), 5)
        return proc.to_host(proc.threshold(proc.absolute_diff(a, b), 25.0))

    gpu = chain(cuda_processor, *frames)
    ref = chain(cpu_processor, *frames)
    # Masks may differ only on pixels whose difference sits within blur tolerance of
    # the threshold; require agreement on all but a negligible fraction of pixels.
    disagreement = np.mean(gpu != ref)
    assert disagreement < 1e-3, f"mask disagreement {disagreement:.5f}"


@pytest.mark.gpu
# Asserts on the backend only. Note that it does depend on all five kernels *compiling*
# — a compilation dependency the launch-site tracer does not and cannot see.
@pytest.mark.no_kernel_launch
def test_backend_is_pycuda_on_device(cuda_processor):
    assert cuda_processor.backend == "pycuda"


@pytest.mark.gpu
# The guard raises before any launch. It depends on the tile geometry instead, and
# "alter TILE_W" is in the planned mutation corpus — a constant dependency, invisible here.
@pytest.mark.no_kernel_launch
def test_blur_rejects_kernel_larger_than_halo(cuda_processor, frames):
    gray = frames[0][:, :, 0].astype(np.float32)
    with pytest.raises(ValueError):
        cuda_processor.gaussian_blur(cuda_processor.to_device(gray), kernels.MAX_KERNEL_SIZE + 2)
