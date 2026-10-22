# SPDX-License-Identifier: Apache-2.0
"""Internals of the eldpc verification suite.

The suite is run through ``verify/run_verify.py``. This package holds the corpus builder
(same circuit and DEM rules as the reference construction), the reference-library build
cache, the ctypes bindings used for the reference and for the optional separately built library,
the comparison logic, and the report writer. It depends only on the public ``eldpc`` API and on the scientific Python stack
(numpy, scipy, stim, qldpc); the optional ``ldpc`` wheel is used only for the informational
upstream-window column.
"""

SUMMARY_SCHEMA = "eldpc.verify.summary.v1"
CORPUS_SCHEMA = "eldpc.verify.corpus.v1"
CONFIG_SCHEMA = "eldpc.verify.config.v1"

__all__ = ["SUMMARY_SCHEMA", "CORPUS_SCHEMA", "CONFIG_SCHEMA"]
