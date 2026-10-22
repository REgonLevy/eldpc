# Run record

- command: `eldpc-tune --dem model.dem --ler-target 1e-3 --p99-budget-us 200 --auto --out tune200`
- commit: `480f488` (the code that produced the run)
- results: `results.json`
- written by: `tuning/tools/make_run_note.py` from `results.json`
- timings: screens: a shared host (1-min load average 4 to 6 of 128 logical CPUs), CPU governor schedutil, GPU clocks not locked; latency is not timing-grade

| | |
|---|---|
| GPU | NVIDIA RTX 6000 Ada Generation |
| GPU compute capability / SMs / L2 | 8.9 / 142 / 100663296 B |
| GPU driver | 580.126.09 |
| CPU | AMD EPYC 7542 32-Core Processor (128 threads) |
| Package | eldpc 0.1.0, ISA avx2 |
| Python / numpy / scipy / stim / qLDPC | 3.12.3 / 2.5.4 / 1.18.1 / 1.16.0 / 0.4.2.dev20261004 |
| CUDA toolkit | 13.3 (V13.3.73, source: build_info) |
| nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv | name, driver_version, compute_cap; NVIDIA RTX 6000 Ada Generation, 580.126.09, 8.9 |
| Started (UTC) | 2026-10-22T03:05:57.229457+00:00 |
| Wall time (s) | 182.4 |
| Quick | False |
| Schema | eldpc.tune.v2 |
