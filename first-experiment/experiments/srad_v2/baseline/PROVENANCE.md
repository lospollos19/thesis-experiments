# SIL baseline

`srad_sil_baseline.out` is the output of unmutated `srad_v2` on the SIL path.
It is the oracle: mutants are compared against it.

| | |
|---|---|
| Source | `experiments/srad_v2/srad.cu`, single TU, `#define OUTPUT` |
| Toolchain | `translate_srad.sh` (compile, kernelTranslator, hostTranslator, llc, link, run) |
| Environment | CuPBoP SIL image, CuPBoP pinned at `508bd62e928bea3b5f0633c8fa63b5f42f3b4da0`, x86 runner |
| Arguments | `512 512 0 127 0 127 0.5 2` |
| Seed | `srand(7)`, fixed input |
| Shape | 512 rows x 512 floats, `%.5f` |
| SHA-256 | `027a119466614ef9a24464421686d1e068786de3765efbf9a19cafb64f5548fc` |

```bash
shasum -a 256 srad_sil_baseline.out
```
