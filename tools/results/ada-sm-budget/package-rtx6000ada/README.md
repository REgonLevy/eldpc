# The two core variants through the package on the RTX 6000 Ada: a screen

The `ada64` / default ratios of `tools/results/ada-sm-budget/rtx6000ada-142sm.json` are device-side
ratios of the measurement harness. This screen times the installed package instead: the C API
single-shot call (`bench/bench.py c-latency --quick`, back to back, 2400 timed shots per run) on
bb144 xyz at `a03125c3`, once with each core variant forced by `ELDPC_DEVICE_CLASS`, at revision
`f44723c` on the machine of docs/results/ (RESULTS.md sections 1-5: an RTX 6000 Ada Generation,
142 SMs, driver 580.126.09, CUDA 13.3 build, with a dual AMD EPYC 7542, AVX2 core), pinned to
CPU 15. The host was shared (1-min
load average 4.0 to 5.0 of 128 logical CPUs), its governor `schedutil`, the GPU clocks not locked.

| file | command (from the repository root) | p50 / p99 (us) |
|---|---|---:|
| `ada64.json` | `ELDPC_DEVICE_CLASS=ada64 python bench/bench.py c-latency --quick --config a03125c3 --cpu 15 --modes back_to_back --out <dir>/ada64` | 63.92 / 81.99 |
| `default.json` | `ELDPC_DEVICE_CLASS=default python bench/bench.py c-latency --quick --config a03125c3 --cpu 15 --modes back_to_back --out <dir>/default` | 89.50 / 109.59 |

`ada64` / default: x0.714 at the p50, x0.748 at the p99 (two runs one after the other, not an
interleaved A/B; each run's own A/A band is in its file). The JSON files are the benchmark's
`results.json` with the GPU's UUID removed; the p50 and p99 are those `python bench/bench.py
summary --out DIR` prints.
