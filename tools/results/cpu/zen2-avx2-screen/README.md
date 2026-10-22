# CPU core against GPU core on a Zen 2 (AVX2) host: a screen

`bench/bench.py c-latency --quick` on an NVIDIA RTX 6000 Ada Generation (142 SMs, driver
580.126.09, CUDA 13.3 build) with a dual AMD EPYC 7542 (Zen 2, AVX2 only, no AVX-512; 2 x 32
cores, SMT on; four cores share one L3), Ubuntu 24.04, at revision `db65b65` (the CPU core and the
GPU core are frozen; at that revision their sources were pinned by `src/cpu/SOURCES.sha256` and
`src/core/SOURCES.sha256`; their code is unchanged since, `src/cpu/TOKENS.sha256` and
`src/core/TOKENS.sha256`). Problem: the benchmark's default bb144 xyz (m = 1872, n = 75600,
k = 24), preset `a03125c3`, C API single-shot calls back to back, 2400 timed shots per run, the
three runs one after the other in one session. The host was shared (1-min load average 5.0 to 6.1
of 128 logical CPUs during the runs), its governor was `schedutil` and the GPU clocks were not
locked: these are screens, not benchmark results.

The CPU-core runs follow docs/BENCH.md, "CPU-core runs": the config's `cpus=` list names the CPUs,
its first `threads - 1` entries take the started threads, and `--cpu` pins the run, and so the
calling thread, to its last entry (README, "CPU core"). Sampling the C driver's threads every
0.1 s during a repeat of the `threads=4` run found the three workers on CPUs 8, 9 and 10 and the
calling thread on CPU 11, each allowed only its own CPU. The GPU run pins the process to CPU 15.

| run | file | command (from the repository root) | p50 / p99 (us) | / GPU at p50 / p99 |
|---|---|---|---:|---:|
| CPU core, `threads=8` on CPUs 8-15 (two L3s), caller on CPU 15 | `cpu-threads8.json` | `python bench/bench.py c-latency --quick --device cpu --config "a03125c3;threads=8;cpus=8-15" --cpu 15 --modes back_to_back --out <dir>/cpu8` | 1165.2 / 1595.6 | 18.2x / 18.8x |
| CPU core, `threads=4` on CPUs 8-11 (one L3), caller on CPU 11 | `cpu-threads4.json` | `python bench/bench.py c-latency --quick --device cpu --config "a03125c3;threads=4;cpus=8-11" --cpu 11 --modes back_to_back --out <dir>/cpu4` | 608.6 / 796.9 | 9.5x / 9.4x |
| GPU core (`ada64`) | `gpu.json` | `python bench/bench.py c-latency --quick --config a03125c3 --cpu 15 --modes back_to_back --out <dir>/gpu` | 63.9 / 84.8 | 1 |

The p50 and p99 are those `python bench/bench.py summary --out DIR` prints for the run (both
sessions together). Three more runs of the `threads=4` command gave p50 607.0 to 609.2 us and p99
697.0 to 725.1 us, so the p99 of the recorded run is at the top of its spread. On this host the CPU
core is about 10x the GPU core's latency at its better placement (one L3), and slower again when
its threads span two L3s; the "about 6x" of the README was measured on AVX-512 hosts
(`tools/results/cpu/gpu_same_machine.json`, `screen.md`).

An earlier form of this screen limited the process with `taskset -c 8-11` and passed `--cpu none
--allow-unpinned`, which gives every worker its own CPU but leaves the calling thread free to move
among the four CPUs, onto one whose worker is busy-waiting (README, "CPU core": a worker on the
caller's CPU makes the decodes many times slower). Four runs of that form in the same session gave
p50 606.8 to 610.7 us and p99 702.9 to 744.5 us, so it did not show the slowdown there, but it
leaves the placement to the scheduler; pin the caller as above.

The JSON files are the benchmark's `results.json` with the GPU's UUID removed.
