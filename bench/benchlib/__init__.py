# SPDX-License-Identifier: Apache-2.0
"""Support modules of the eldpc benchmark scripts (bench/bench.py).

These modules are part of the benchmark tool, not of the installed decoder: they are imported by the
bench.py script from its own directory and use only the Python standard library, numpy/scipy, the eldpc
package and -- for the built-in circuit-level problems -- stim and qLDPC.
"""
