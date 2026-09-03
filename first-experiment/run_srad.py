"""
Pythainer runner: mount the srad_v2 sources into the CuPBoP SIL image and
replay the translation chain (RQ1, step 1 -- "does srad pass through CuPBoP?").

Sources are mounted read-only at /src and copied into the container-internal
/work, so no build artifacts land in your host tree.

M5 note: the image is amd64. Running it locally goes through qemu -- fine for a
yes/no on *translation* (deterministic IR work), NOT for the noise-floor
calibration (numbers). For trustworthy runs, execute in x86 CI (srad-sil.yml).
Use get_str_command() / generate_script() to only EMIT the docker command.
"""
from pathlib import Path
from pythainer.runners import ConcreteDockerRunner

# Replace <owner> with your GitHub account (lowercase).
IMAGE = "ghcr.io/lospollos19/cupbop-experiment/cupbop-sil:latest"
SRAD_DIR = (Path(__file__).parent / "experiments" / "srad_v2").resolve()

# srad_v2 args: rows cols y1 y2 x1 x2 lambda niter  (Rodinia default example)
RUN_ARGS = "2048 2048 0 127 0 127 0.5 2"


def get_srad_runner() -> ConcreteDockerRunner:
    return ConcreteDockerRunner(
        image=IMAGE,
        volumes={str(SRAD_DIR): "/src:ro"},   # sources read-only, no host pollution
        workdir="/work",                       # container-internal writable dir (from the image)
        root=True,                             # artifacts stay in the ephemeral container
        tty=False,
        interactive=False,
    )


COMMANDS = [
    "cp -a /src/. /work/",
    f"bash translate_srad.sh srad.cu {RUN_ARGS}",
]


if __name__ == "__main__":
    runner = get_srad_runner()
    print("# docker command pythainer would run:")
    print(runner.get_str_command())
    print("\n# commands inside the container:")
    print("  " + " && ".join(COMMANDS))
    # To actually execute locally (amd64 -> qemu on the M5), uncomment:
    # runner.run(commands=COMMANDS)