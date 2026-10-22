# SPDX-License-Identifier: Apache-2.0
"""Decoder configurations: presets, config dictionaries, ABI config strings and JSON files."""

from __future__ import annotations

import json
import numbers
import re
from pathlib import Path
from typing import Mapping

from ._abi import EldpcError

# (ms_scaling_factor, max_iter) of min-sum BP + OSD-0 with the parallel schedule.
_PRESET_PARAMS: dict[str, tuple[float, int]] = {
    "ms30": (0.625, 30),
    "ms100": (0.625, 100),
    "ms30a075": (0.75, 30),
    "a05ms31": (0.5, 31),
    "a05c15": (0.5, 15),
    "a05c11": (0.5, 11),
    "a0375c7": (0.375, 7),
    "a04375c3": (0.4375, 3),
    "a03125c3": (0.3125, 3),
}

_BP_OSD_0 = {
    "decoder": "bposd",
    "bp_method": "ms",
    "schedule": "parallel",
    "osd_method": "osd0",
    "osd_order": 0,
}

_KEY_RE = re.compile(r"^[a-z0-9_]+$")


def presets() -> dict[str, dict]:
    """All named configurations: ``name -> config dict``.

    Each min-sum BP + OSD-0 preset has a ``.bp`` variant: the same BP without the OSD
    post-processing (``decoder=bp``).
    """
    out: dict[str, dict] = {}
    for name, (alpha, max_iter) in _PRESET_PARAMS.items():
        out[name] = dict(_BP_OSD_0, ms_scaling_factor=alpha, max_iter=max_iter)
        out[name + ".bp"] = {"decoder": "bp", "bp_method": "ms", "schedule": "parallel",
                             "ms_scaling_factor": alpha, "max_iter": max_iter}
    return out


def _parse_abi_string(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for segment in text.split(";"):
        if not segment.strip():
            continue
        if segment.count("=") != 1:
            raise EldpcError(f"malformed config segment {segment!r}")
        key, value = (part.strip() for part in segment.split("="))
        if not _KEY_RE.match(key):
            raise EldpcError(f"invalid config key {key!r}")
        if key in out:
            raise EldpcError(f"duplicate config key {key!r}")
        out[key] = value
    return out


def _preset_with_keys(text: str, table: dict) -> dict | None:
    """A preset name followed by extra ``key=value`` segments ("a03125c3;threads=8"): the preset's
    dict with the extra keys added or overriding its values; None when the first segment holds a '='
    (a plain ABI string). The C front expands such a string the same way."""
    segments = [seg for seg in text.split(";") if seg.strip()]
    if not segments or "=" in segments[0]:
        return None
    name = segments[0].strip()
    if name not in table:
        raise EldpcError(f"unknown preset {name!r} at the start of the config string {text!r}; "
                         f"presets: {', '.join(sorted(table))}")
    return {**table[name], **_parse_abi_string(";".join(segments[1:]))}


def resolve_config(config) -> dict:
    """Resolve a preset name, a config dict, an ABI config string or a path to a JSON file.

    A preset name may be followed by extra ``key=value`` segments, ``"a03125c3;threads=8"``: the
    extra keys add to the preset's keys or override them.
    """
    if isinstance(config, Mapping):
        return dict(config)
    if isinstance(config, (str, Path)):
        text = str(config)
        table = presets()
        if text in table:
            return dict(table[text])
        if text.endswith(".json"):
            path = Path(text)
            if not path.is_file():
                raise EldpcError(f"config file not found: {text}")
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise EldpcError(f"cannot read the config file {text}: {exc}") from None
            if not isinstance(data, Mapping):
                raise EldpcError(f"the config file {text} does not hold a JSON object")
            return dict(data)
        if "=" in text or ";" in text:
            with_keys = _preset_with_keys(text, table)
            return with_keys if with_keys is not None else _parse_abi_string(text)
        raise EldpcError(f"unknown preset {text!r}; presets: {', '.join(sorted(table))}")
    raise EldpcError(f"config must be a preset name, a dict, an ABI config string or a .json path, "
                     f"got {type(config).__name__}")


def _format_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        return repr(float(value))
    if isinstance(value, str):
        return value
    raise EldpcError(f"config values must be bool, int, float or str, got {type(value).__name__}")


def config_string(config: Mapping) -> str:
    """Canonical ABI config string: keys sorted, ``key=value`` joined by ``;``."""
    parts = []
    for key in sorted(config):
        if not isinstance(key, str) or not _KEY_RE.match(key):
            raise EldpcError(f"invalid config key {key!r}")
        parts.append(f"{key}={_format_value(config[key])}")
    return ";".join(parts)
