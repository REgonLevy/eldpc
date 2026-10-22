# Build listing

`wheel-and-images.json`: the wheel that `pip wheel` builds from this repository (its file list and sizes,
`python -m zipfile -l`: the seven native libraries, the C header, the pkg-config file and the CMake package config
inside the Python package) and the GPU images of its libraries (`cuobjdump --list-elf` and `--list-ptx`: SASS for
`sm_89` and `sm_120` plus `compute_120` PTX in each of the four GPU cores; no device code in the front and the two CPU
cores), built from the sources of commit 45df670 on the Ryzen 9 7950X3D + RTX 5080 machine of `tools/results/matrix/` with GCC 13.3 and CUDA 13.3.73. The wheel carries the licence files (`LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES`, `AUTHORS`) in `eldpc-0.1.0.dist-info/licenses/`.
Paths are dropped. Cited by docs/CLAIMS.md rows 6, 16 and 34 and docs/BUILD.md.
