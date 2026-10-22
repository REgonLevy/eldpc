# Run record

- command: `eldpc-tune --problem bb72 --p 0.001 --weight 0 --p99-budget-us 1000 --max-shots 2048 --auto --out tuning/results/bb72-weight0`
- commit: `5e03202` (the code that produced the run)
- results: `results.json`
- written by: `tuning/tools/make_run_note.py` from `results.json`
- timings: screens: an idle machine without a fixed clock profile (CPU governor powersave); latency is not timing-grade

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 |
| GPU compute capability / SMs / L2 | 12.0 / 84 / 67108864 B |
| GPU driver | 595.91.07 |
| CPU | AMD Ryzen 9 7950X3D 16-Core Processor (32 threads) |
| Package | eldpc 0.1.0, ISA avx512 |
| Python / numpy / scipy / stim / qLDPC | 3.12.3 / 2.5.4 / 1.18.1 / 1.16.0 / 0.4.2.dev20261004 |
| nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv | name, driver_version, compute_cap; NVIDIA GeForce RTX 5080, 595.91.07, 12.0 |
| nvcc --version | nvcc: NVIDIA (R) Cuda compiler driver; Copyright (c) 2005-2026 NVIDIA Corporation; Built on Tue_Jun_09_02:43:40_PM_PDT_2026; Cuda compilation tools, release 13.3, V13.3.73; Build cuda_13.3.r13.3/compiler.38244171_0 |
| $CUDACXX --version | nvcc: NVIDIA (R) Cuda compiler driver; Copyright (c) 2005-2026 NVIDIA Corporation; Built on Tue_Jun_09_02:43:40_PM_PDT_2026; Cuda compilation tools, release 13.3, V13.3.73; Build cuda_13.3.r13.3/compiler.38244171_0 |
| Started (UTC) | 2026-10-21T12:52:59.936699+00:00 |
| Wall time (s) | 20.9 |
| Quick | False |
| Schema | eldpc.tune.v2 |
