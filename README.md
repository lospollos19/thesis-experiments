# work-thesis

Code and data for the two experiments of the thesis.

| Directory | Subject |
|---|---|
| `first-experiment/` | Mutation testing of a CUDA C++ benchmark in two environments: CuPBoP on an x86 CI runner without a GPU, against native execution on an NVIDIA AGX Orin. |
| `second-experiment/` | Regression test selection on a PyCUDA vision pipeline, with the CUDA-C sources stored two ways. |

Each carries its own README, dependencies and tests, and is meant to be read and
run on its own. Both include the artifacts and logs of the CI runs that produced
their measurements, under `results/` and `logs/`.
