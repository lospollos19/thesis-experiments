"""
Pythainer builder for the CuPBoP SIL image (x86 SIL stage of the RQ1 setup).
Encodes the CI-validated recipe (vecadd PASS) as a deterministic, reproducible image.

Local use on the M5: only call generate_dockerfile() here (pure text, no Docker).
Never call builder.build() locally -- that would trigger amd64 emulation.
The real build+push happens in CI on an x86 runner.
"""
from pythainer.builders import UbuntuDockerBuilder

# Pinned CuPBoP commit = reproducibility (the SHA freezes the tree and its submodules)
CUPBOP_SHA = "508bd62e928bea3b5f0633c8fa63b5f42f3b4da0"
BASE_IMAGE = "nvidia/cuda:11.7.1-devel-ubuntu22.04"
IMAGE_TAG = "cupbop-sil:latest"


def get_cupbop_builder() -> UbuntuDockerBuilder:
    # ubuntu_base_tag is the FROM image; the nvidia/cuda image is Ubuntu-based, so apt works.
    builder = UbuntuDockerBuilder(tag=IMAGE_TAG, ubuntu_base_tag=BASE_IMAGE)

    builder.desc("LLVM 14 toolchain + build tools")
    builder.add_packages(
        packages=[
            "llvm-14", "llvm-14-dev", "clang-14", "libclang-14-dev",
            "cmake", "make", "git", "g++", "ca-certificates", "zlib1g-dev",
        ]
    )

    builder.desc("Point bare names to the -14 variants (CuPBoP calls clang / llvm-config unsuffixed)")
    builder.run_multiple(
        commands=[
            f"update-alternatives --install /usr/bin/{t} {t} /usr/bin/{t}-14 100"
            for t in ("clang", "clang++", "llvm-config", "llc")
        ]
    )

    builder.desc("Clone CuPBoP at the pinned commit + submodules")
    builder.workdir("/opt")
    # Single RUN, single shell: cd once, then chain the git commands.
    builder.run_multiple(
        commands=[
            "git clone https://github.com/cupbop/CuPBoP.git",
            "cd CuPBoP",
            f"git checkout {CUPBOP_SHA}",
            "git submodule update --init --recursive",
        ]
    )

    builder.env("CuPBoP_PATH", "/opt/CuPBoP")
    builder.env("CUDA_PATH", "/usr/local/cuda")
    builder.env("LD_LIBRARY_PATH", "/opt/CuPBoP/build/runtime:/opt/CuPBoP/build/runtime/threadPool")

    builder.desc("Build CuPBoP")
    builder.workdir("/opt/CuPBoP/build")
    builder.run_multiple(commands=["cmake ..", "make -j$(nproc)"])

    builder.workdir("/work")
    return builder


if __name__ == "__main__":
    # Render the Dockerfile only (no Docker build). Diff it against the committed one:
    #   python3 build_cupbop.py && diff docker/cupbop-sil/Dockerfile Dockerfile.generated
    builder = get_cupbop_builder()
    builder.generate_dockerfile(dockerfile_paths=["Dockerfile.generated"])
    print(open("Dockerfile.generated").read())