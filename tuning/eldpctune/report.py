# SPDX-License-Identifier: Apache-2.0
"""Output files of eldpc-tune: config.json, results.json, pareto.md and the per-stage raw files.

Every number the tool prints comes from the results dictionary written here; the console output is
a rendering of it, not a separate computation.
"""
from __future__ import annotations

import json
import os

import numpy as np


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return str(value)


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=False, default=_json_default)
        f.write("\n")


def fmt_prob(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    v = float(value)
    if v == 0.0:
        return "0"
    if v < 1e-3 or v >= 1e3:
        return f"{v:.{digits}g}"
    return f"{v:.{digits}g}"


def config_text(config: dict) -> str:
    return ";".join(f"{k}={config[k]}" for k in sorted(config))


def _ler_cell(row: dict) -> str:
    parts = []
    for entry in row["per_case"]:
        name = entry["case"]
        parts.append(f"{name}: {fmt_prob(entry['ler'])} "
                     f"[{fmt_prob(entry['cp99_lower'])}, {fmt_prob(entry['cp99_upper'])}]")
    return "; ".join(parts) if parts else "-"


def _latency_cell(row: dict) -> tuple[str, str, str]:
    if row["p50_us"] is None:
        return "-", "-", "-"
    budget = row.get("budget") or {}
    within = budget.get("within")
    mark = "yes" if within else "no"
    if not budget.get("checks"):
        mark = "-"
    return f"{row['p50_us']:.1f}", f"{row['p99_us']:.1f}", mark


def toolkit_text(toolkit) -> str:
    """``13.3 (V13.3.73, from the library's build_info)``; ``not recorded`` without a toolkit."""
    if not toolkit or not toolkit.get("version"):
        return "not recorded"
    where = {"build_info": "from the library's build_info",
             "nvcc": f"from `{toolkit.get('command') or 'nvcc --version'}`"}.get(
        toolkit.get("source"))
    detail = ", ".join(x for x in (toolkit.get("nvcc"), where) if x)
    return toolkit["version"] + (f" ({detail})" if detail else "")


def tie_mark(tie: dict) -> str:
    """The mark of a tie-set entry: the accepted choice, or the screen's pick when they differ."""
    if tie.get("chosen"):
        return "  <- chosen"
    if tie.get("screen_pick"):
        return "  (screen pick)"
    return ""


def nearest_text(near: dict, quote: str = "`") -> str:
    """One nearest-list entry: p50 / p99 and the budget ratio of one measurement, which is named
    (``measurement``: re-timed, latency stage or latency confirmation, with its call count)."""
    distance = near.get("budget_distance")
    tail = f", {distance:.2f}x the budget" if distance is not None else ""
    source = near.get("measurement")
    basis = near.get("basis") or ""
    if source and basis.startswith("latency stage (") and basis.endswith(")"):
        source += "; " + basis[len("latency stage ("):-1]
    source = source or basis
    note = f" ({source})" if source else ""
    return (f"{quote}{near['label']}{quote}: LER upper {fmt_prob(near.get('ler_upper'))}, p50 "
            f"{near['p50_us']:.1f} us, p99 {near['p99_us']:.1f} us{tail}{note}")


def confirmation_count_text(accuracy: dict) -> str:
    """How many candidates the confirmation measured, of how many in budget, and why it stopped."""
    total = accuracy.get("in_budget_candidates")
    confirmed = accuracy.get("confirmed", len(accuracy.get("candidates") or []))
    cap = accuracy.get("max_candidates")
    stopped = accuracy.get("stopped")
    why = {"accepted": "stopped at the first accepted candidate",
           "exhausted": "every in-budget candidate confirmed, none accepted",
           "cap": f"stopped by --confirm-candidates {cap}",
           "no candidate": "no candidate within the budget"}.get(stopped, str(stopped))
    text = (f"confirmed {confirmed} of {total if total is not None else '?'} in-budget "
            f"candidate(s), one at a time in the choice's order ({why}; cap: "
            f"{'none' if cap is None else cap})")
    unconfirmed = accuracy.get("unconfirmed") or []
    if unconfirmed:
        text += f"; never confirmed: {', '.join(unconfirmed)}"
    return text


def _devices_text(devices) -> str:
    if not devices:
        return "none reported"
    parts = []
    for dev in devices:
        if "error" in dev:
            parts.append(f"unavailable ({dev['error']})")
        else:
            parts.append(f"{dev.get('name', '?')} (cc {dev.get('compute_capability', '?')}, "
                         f"driver {dev.get('driver_version', '?')})")
    return "; ".join(parts)


def _core_variant_lines(record) -> list[str]:
    """The "Core variants" section of pareto.md (none when the stage did not run)."""
    if not record:
        return []
    lines = ["", "## Core variants", ""]
    device = record.get("device") or {}
    lines.append(f"- device: {device.get('name', '?')} (cc {device.get('compute_capability', '?')}, "
                 f"{device.get('sm_count', '?')} SMs); the table's variant: "
                 f"{record.get('table_variant')}")
    if not record.get("evaluated"):
        lines.append(f"- not measured: {record.get('reason')}")
        return lines
    for name in ("default", "ada64"):
        summary = record[name]
        lines.append(f"- `{record['label']}` with {name}: p50 {summary['p50_us']:.1f} us, "
                     f"p99 {summary['p99_us']:.1f} us ({record['calls_per_side']} calls per side "
                     f"and case, paired on the same syndromes)")
    lines.append(f"- ada64 / default: p50 x{record['ratio_p50']:.3f}, p99 x{record['ratio_p99']:.3f}; "
                 f"A/A band p50 +-{record['band_p50'] * 100:.1f} %, p99 "
                 f"+-{record['band_p99'] * 100:.1f} %")
    lines.append(f"- faster: {record['faster']}; {record['recommendation']}")
    lines.append(f"- {record['config_json']}")
    return lines


def _screen_claim_lines(choice: dict) -> list[str]:
    """The LER lines of a choice whose claim is the screen's estimate (runs without a target)."""
    lines = [f"- LER (screen, {choice.get('ler_shots_per_case')} shots per case, worst case): "
             f"{fmt_prob(choice.get('ler'))}; one-sided bounds [{fmt_prob(choice.get('ler_lower'))}, "
             f"{fmt_prob(choice.get('ler_upper'))}]"]
    check = choice.get("confirmation_check") or {}
    if check:
        lines.append(f"- accuracy confirmation (a consistency check, not the reported number): "
                     f"{check.get('failures')} failures in {check.get('shots_per_case')} shots "
                     f"per case, LER {fmt_prob(check.get('ler'))}: "
                     f"{'consistent' if check.get('consistent') else 'NOT consistent'} with the "
                     f"screen")
    floor = choice.get("ler_floor") or {}
    if floor:
        lines.append(f"- LER floor: {'passed' if floor.get('passed') else 'not passed'}")
    for item in choice.get("passed_over") or []:
        lines.append(f"- passed over `{item['label']}`: ranked higher, but its latency "
                     f"confirmation read p50 {item['p50_us']:.1f} us, p99 {item['p99_us']:.1f} us "
                     f"(over the budget)")
    return lines


def _ler_floor_lines(floor) -> list[str]:
    """The "LER floor" section of pareto.md (runs without --ler-target)."""
    if not floor:
        return []
    lines = ["", "## LER floor", "", f"- source: {floor['source']}", f"- rule: {floor['rule']}"]
    for ref in floor.get("references") or []:
        cases = "; ".join(f"{case['case']}: {case['failures']}/{case['shots']} "
                          f"(LER {fmt_prob(case['ler'])})" for case in ref["per_case"])
        lines.append(f"- `{ref['label']}` on the {ref['stage']} shots: {cases}")
    excluded = floor.get("excluded") or []
    shown = ", ".join(f"`{label}`" for label in excluded[:12])
    more = f" and {len(excluded) - 12} more" if len(excluded) > 12 else ""
    lines.append(f"- choosable points: {floor.get('choosable')}; below the floor: "
                 f"{len(excluded)}" + (f" ({shown}{more})" if excluded else ""))
    return lines


def _budget_retime_lines(record) -> list[str]:
    """The "Budget membership" section of pareto.md (both paths, with a latency budget)."""
    if not record:
        return []
    lines = ["", "## Budget membership", ""]
    if record.get("note"):
        return lines + [f"- {record['note']}"]
    lines.append(f"- rule: {record['rule']}")
    lines.append(f"- reference: `{record.get('reference')}`; re-timed: "
                 f"{len(record.get('retimed') or [])}; decided by the latency stage: "
                 f"{record.get('decided_by_latency_stage')}; capped: "
                 f"{len(record.get('capped') or [])}; wall {record.get('wall_seconds', 0.0):.1f} s")
    if record.get("retimed"):
        lines.append("")
        lines.append("| point | latency-stage p99 us (x budget) | re-timed p50 us | re-timed p99 "
                     "us | reference p99 us | in budget |")
        lines.append("|---|---|---|---|---|---|")
        for item in record["retimed"]:
            lines.append(f"| `{item['label']}` | {item['screen_p99_us']:.1f} "
                         f"(x{item['screen_ratio']:.2f}) | {item['p50_us']:.1f} | "
                         f"{item['p99_us']:.1f} | {item['ref_p99_us']:.1f} | "
                         f"{'yes' if item['within'] else 'no'} |")
    for item in record.get("capped") or []:
        lines.append(f"- not re-timed (cap): `{item['label']}`, x{item['screen_ratio']:.2f} the "
                     f"budget; latency-stage verdict {'in' if item['within'] else 'out'}")
    return lines


def render_pareto_md(results: dict) -> str:
    inputs = results["inputs"]
    choice = results["choice"]
    env = results["environment"]
    gpu_commands = [c for c in env["commands"] if "nvidia-smi" in c["command"]]
    nvcc_commands = [c for c in env["commands"]
                     if "nvcc" in c["command"] or "CUDACXX" in c["command"]]
    lines = []
    lines.append("# Tuning result")
    lines.append("")
    lines.append(f"- command: `{' '.join(results['tool']['command'])}`")
    problem_label = inputs.get("problem") or inputs.get("dem") or inputs.get("circuit") or inputs.get("h")
    lines.append(f"- problem: `{problem_label}`")
    if inputs.get("p"):
        lines.append(f"- physical error rates: {', '.join(f'{p:g}' for p in inputs['p'])}")
    if inputs.get("rounds"):
        lines.append(f"- rounds: {inputs['rounds']}")
    lines.append(f"- shots per point per case: up to {inputs['max_shots']} "
                 f"(chunks of {inputs['chunk_shots']})")
    lines.append(f"- confidence of the one-sided bounds: {inputs['confidence']:g}")
    if inputs.get("ler_target") is not None:
        lines.append(f"- LER target (upper bound per shot): {inputs['ler_target']:g}")
    if inputs.get("max_ler") is not None:
        lines.append(f"- LER floor (--max-ler, screen estimate): {inputs['max_ler']:g}")
    if inputs.get("weight") is not None:
        lines.append(f"- trade-off weight: {inputs['weight']:g} "
                     f"(score = log10 LER upper + weight * log10 p50)")
    lines.append(f"- call cadence: {inputs['cadence_us']:g} us")
    budgets = []
    if inputs.get("p50_budget_us") is not None:
        budgets.append(f"p50 {inputs['p50_budget_us']:g} us")
    if inputs.get("p99_budget_us") is not None:
        budgets.append(f"p99 {inputs['p99_budget_us']:g} us")
    if budgets:
        lines.append(f"- latency budget: {', '.join(budgets)}")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    for command in gpu_commands + nvcc_commands:
        out = (command.get("stdout") or command.get("stderr") or "").replace("\n", "; ")
        lines.append(f"- `{command['command']}` -> {out}")
    if "toolkit" in env:
        lines.append(f"- CUDA toolkit: {toolkit_text(env.get('toolkit'))}")
    lines.append(f"- eldpc {env['eldpc']['version']}, host ISA {env['eldpc']['isa']}")
    gpu = env["machine"].get("gpu") or {}
    visible = gpu.get("cuda_visible_devices")
    lines.append(f"- CUDA_VISIBLE_DEVICES: {visible if visible is not None else '(unset)'}; "
                 f"eldpc.devices(): {_devices_text(gpu.get('devices'))}")
    lines.append(f"- CPU: {env['machine']['cpu_model']} ({env['machine']['cpu_count']} threads)")
    lines.append("")
    lines.append("## Choice")
    lines.append("")
    if choice.get("config") is not None:
        lines.append(f"- config: `{config_text(choice['config'])}`")
        if choice.get("ler_source") == "screen":
            lines += _screen_claim_lines(choice)
        elif choice.get("ler_source") == "accuracy_confirmation":
            lines.append(f"- LER (accuracy confirmation, worst case): "
                         f"{fmt_prob(choice.get('ler'))}; one-sided upper bound "
                         f"{fmt_prob(choice.get('ler_upper'))}")
            screen = choice.get("screen") or {}
            if screen:
                lines.append(f"- screen (not the reported number): "
                             f"{screen.get('failures')}/{screen.get('shots')} failures, LER "
                             f"{fmt_prob(screen.get('ler'))}, upper bound "
                             f"{fmt_prob(screen.get('upper'))}")
        else:
            lines.append(f"- LER upper bound (worst case): {fmt_prob(choice.get('ler_upper'))}")
        lines.append(f"- latency (latency confirmation): p50 {choice.get('p50_us'):.1f} us, "
                     f"p99 {choice.get('p99_us'):.1f} us")
        if choice.get("p99_unreliable"):
            lines.append(f"- WARNING: the p99 is unreliable: slow_fraction "
                         f"{choice.get('slow_fraction'):.4g} is within 2x of 1 %, so the p99 "
                         f"sits on the cliff between the converged calls and the slow class")
        lines.append(f"- ranking: {choice.get('ranking', '-')}")
        ties = choice.get("tie_set") or []
        if ties:
            lines.append(f"- tie set ({len(ties)}; fastest `{choice.get('fastest')}`): "
                         f"{choice.get('tie_rule', '')}")
            for tie in ties:
                lines.append(f"  - `{tie['label']}`: p50 {tie['p50_us']:.2f} us, "
                             f"p99 {tie['p99_us']:.2f} us, LER upper {fmt_prob(tie['ler_upper'])}, "
                             f"max_iter {tie['max_iter']}, alpha {tie['ms_scaling_factor']:g}"
                             f"{tie_mark(tie)}")
        if results.get("keep_warm") and results["keep_warm"].get("evaluated"):
            keep = results["keep_warm"]
            lines.append(f"- keep-warm: {'on' if keep['recommended'] else 'off'} "
                         f"(p50 off/on {keep['off']['p50_us']:.1f}/{keep['on']['p50_us']:.1f} us, "
                         f"p99 {keep['off']['p99_us']:.1f}/{keep['on']['p99_us']:.1f} us)")
            if keep.get("p99_note"):
                lines.append(f"  - {keep['p99_note']}")
        elif results.get("keep_warm"):
            lines.append(f"- keep-warm: off ({results['keep_warm'].get('reason', '')})")
        confirmation = results.get("confirmation")
        if confirmation:
            summary = confirmation["summary"]
            within = confirmation["budget"]["within"]
            lines.append(f"- confirmation run ({summary['calls']} calls per case): "
                         f"p50 {summary['p50_us']:.1f} us, p99 {summary['p99_us']:.1f} us, "
                         f"within budget: {'yes' if within else 'no'}")
    else:
        lines.append(f"- no configuration qualifies: {choice.get('reason')}")
        for near in choice.get("nearest", []):
            if near.get("p50_us") is not None:
                lines.append(f"  - nearest {nearest_text(near)}")
            else:
                lines.append(f"  - nearest `{near['label']}`: status {near.get('status')}, "
                             f"LER upper {fmt_prob(near.get('ler_upper'))}")
    lines += _ler_floor_lines(results.get("ler_floor"))
    lines += _budget_retime_lines(results.get("budget_retime"))
    accuracy = results.get("accuracy_confirmation")
    if accuracy:
        lines.append("")
        lines.append("## Accuracy confirmation")
        lines.append("")
        lines.append(f"- planned shots: {accuracy['planned_shots']} per candidate and case, "
                     f"declared before the first confirmation shot; no early stop")
        lines.append(f"- shot-count rule: {accuracy['shots_rule']}")
        lines.append(f"- confidence: {accuracy['confidence']:g}; target: "
                     f"{fmt_prob(accuracy['target'])}")
        lines.append(f"- {confirmation_count_text(accuracy)}")
        lines.append("- case seeds: " + "; ".join(
            f"{case['label']} -> {case['seed']}" for case in accuracy.get("cases", [])))
        lines.append("")
        lines.append("| candidate | rank | screen upper | confirmation failures/shots | "
                     "LER | upper | passed | latency within | slow fraction | p99 unreliable | "
                     "accepted |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for cand in accuracy.get("candidates", []):
            screen = cand.get("screen") or {}
            lat = cand.get("latency") or {}
            lines.append(f"| `{cand['label']}` | {cand.get('rank')} | "
                         f"{fmt_prob(screen.get('upper'))} | "
                         f"{cand.get('failures')}/{cand.get('shots')} | "
                         f"{fmt_prob(cand.get('ler'))} | {fmt_prob(cand.get('upper'))} | "
                         f"{'yes' if cand.get('passed') else 'no'} | "
                         f"{'yes' if cand.get('latency_within') else 'no'} | "
                         f"{fmt_prob(lat.get('slow_fraction'))} | "
                         f"{'yes' if lat.get('p99_unreliable') else 'no'} | "
                         f"{'yes' if cand.get('accepted') else 'no'} |")
        for cand in accuracy.get("candidates", []):
            if cand.get("failure_reasons"):
                lines.append(f"- `{cand['label']}` (rank {cand.get('rank')}) not accepted: "
                             + "; ".join(cand["failure_reasons"]))
        for cand in accuracy.get("candidates", []):
            if cand.get("consistency"):
                lines.append(f"- `{cand['label']}`: the confirmation is "
                             f"{'consistent' if cand['consistency']['consistent'] else 'NOT consistent'}"
                             f" with the screen's estimate (a consistency check; the reported LER "
                             f"is the screen's)")
        if accuracy.get("chosen") is None:
            lines.append("")
            lines.append("No candidate was accepted: there is no chosen configuration; the "
                         "candidates are listed above.")
    lines += _core_variant_lines(results.get("core_variants"))
    if results.get("warnings"):
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        lines += [f"- {warning}" for warning in results["warnings"]]
    pruning = results.get("pruning")
    if pruning:
        lines.append("")
        lines.append("## Pruning")
        lines.append("")
        lines.append(f"- rule threshold min_converged {pruning['min_converged']:g} at confidence "
                     f"{pruning['confidence']:g}, for a {pruning['quantile']} latency budget; "
                     f"enabled: {'yes' if pruning.get('enabled') else 'no'}")
        lines.append(f"- {pruning['rule']}")
        rows = results.get("pareto", [])
        passing = [row for row in rows if row.get("status") == "pass"
                   and row.get("converged_share") is not None]
        low = [row for row in passing if row["converged_share"] < pruning["min_converged"]]
        if passing:
            lines.append(f"- passing points below the threshold (their larger caps were therefore "
                         f"not pruned): {len(low)} of {len(passing)}; the lowest converged share "
                         f"is {fmt_prob(min(row['converged_share'] for row in passing))}")
        skipped = [row for row in rows if row.get("status") == "skipped"]
        for row in skipped:
            lines.append(f"- skipped `{row['label']}`: dominated by `{row.get('dominated_by')}` "
                         f"(its converged share {fmt_prob(row.get('dominator_converged_share'))})")
    lines.append("")
    lines.append("## Pareto table")
    lines.append("")
    lines.append("| config | decoder | alpha | max_iter | converged share | "
                 "LER per case [99% CP interval] (screen) | p50 us | p99 us | in budget | "
                 "Pareto | note |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    rows = results["pareto"]
    rows = sorted(rows, key=lambda r: (r["p50_us"] is None, r["p50_us"] or 0.0,
                                       r["max_iter"], r["alpha"]))
    chosen = choice.get("point")
    screen_pick = choice.get("screen_choice_label")
    if chosen and screen_pick == chosen.get("label"):
        screen_pick = None
    for row in rows:
        p50, p99, within = _latency_cell(row)
        note = row.get("status", "")
        if row.get("dominated_by"):
            note = f"skipped (dominated by {row['dominated_by']})"
        elif row.get("latency_verdict"):
            note += f"; {row['latency_verdict']}"
        if row.get("ler_floor_passed") is False:
            note += "; below the LER floor (not choosable, not latency-measured)"
        if row.get("retime_p99_us") is not None:
            note += (f"; re-timed p50 {row['retime_p50_us']:.1f} / p99 "
                     f"{row['retime_p99_us']:.1f} us (decides 'in budget')")
        if row.get("p99_unreliable"):
            note += (f"; p99 unreliable (slow fraction {row['slow_fraction']:.3g} within 2x of "
                     f"1 %)" if row.get("slow_fraction") is not None else "; p99 unreliable")
        if chosen and all(chosen.get(k) == row.get(k)
                          for k in ("decoder", "ms_scaling_factor", "max_iter")):
            note += "; CHOSEN"
        elif screen_pick is not None and row["label"] == screen_pick:
            note += "; screen pick (not accepted)"
        star = "*" if row.get("pareto") else ""
        lines.append(f"| {row['label']} | {row['decoder']} | {row['ms_scaling_factor']:g} | "
                     f"{row['max_iter']} | {fmt_prob(row.get('converged_share'))} | "
                     f"{_ler_cell(row)} | {p50} | {p99} | {within} | "
                     f"{star} | {note} |")
    lines.append("")
    lines.append("LER is the fraction of shots with at least one wrong observable; intervals are "
                 "one-sided Clopper-Pearson bounds at the stated confidence.  `*` marks a point on "
                 "the measured (LER, p50) Pareto frontier.  p50/p99 are single-shot calls through "
                 "the package API at the stated cadence; the A/A band of the machine is in "
                 "results.json.")
    lines.append("")
    return "\n".join(lines)


def print_summary(results: dict) -> None:
    """Console rendering of the results dictionary (no number is computed here)."""
    choice = results["choice"]
    print("")
    print("Pareto table (LER upper bound, single-shot p50/p99):")
    rows = sorted(results["pareto"],
                  key=lambda r: (r["p50_us"] is None, r["p50_us"] or 0.0,
                                 r["max_iter"], r["alpha"]))
    for row in rows:
        if row["p50_us"] is None:
            continue
        print(f"  {row['label']:<26s} LER {_ler_cell(row)}")
        print(f"  {'':26s} p50 {row['p50_us']:.1f} us  p99 {row['p99_us']:.1f} us  "
              f"{'pareto' if row.get('pareto') else ''}"
              f"{'  budget:' + ('yes' if (row.get('budget') or {}).get('within') else 'no') if (row.get('budget') or {}).get('checks') else ''}")
    print("")
    if choice.get("config") is not None:
        print(f"Choice: {config_text(choice['config'])}")
        if choice.get("ler_source") == "screen":
            print(f"  LER {fmt_prob(choice.get('ler'))} (screen, {choice.get('ler_shots_per_case')} "
                  f"shots per case, worst case; one-sided bounds "
                  f"[{fmt_prob(choice.get('ler_lower'))}, {fmt_prob(choice.get('ler_upper'))}])")
            check = choice.get("confirmation_check") or {}
            if check:
                print(f"  accuracy confirmation (consistency check): {check.get('failures')} "
                      f"failures in {check.get('shots_per_case')} shots per case: "
                      f"{'consistent' if check.get('consistent') else 'NOT consistent'}")
        elif choice.get("ler_source") == "accuracy_confirmation":
            screen = choice.get("screen") or {}
            print(f"  LER {fmt_prob(choice.get('ler'))} (accuracy confirmation, worst case; "
                  f"upper bound {fmt_prob(choice.get('ler_upper'))})")
            if screen:
                print(f"  screen (not the reported number): LER {fmt_prob(screen.get('ler'))}, "
                      f"upper bound {fmt_prob(screen.get('upper'))}")
        else:
            print(f"  LER upper bound {fmt_prob(choice.get('ler_upper'))}")
        print(f"  p50 {choice['p50_us']:.1f} us, p99 {choice['p99_us']:.1f} us "
              f"(latency confirmation)")
        if choice.get("p99_unreliable"):
            print(f"  WARNING: the p99 is unreliable: slow_fraction "
                  f"{choice.get('slow_fraction'):.4g} is within 2x of 1 %")
        for tie in choice.get("tie_set") or []:
            print(f"  tie set: {tie['label']}: p50 {tie['p50_us']:.2f} us, "
                  f"p99 {tie['p99_us']:.2f} us, LER upper {fmt_prob(tie['ler_upper'])}"
                  f"{tie_mark(tie)}")
        keep = results.get("keep_warm")
        if keep and keep.get("evaluated"):
            print(f"  keep-warm: {'on' if keep['recommended'] else 'off'}")
            if keep.get("p99_note"):
                print(f"    {keep['p99_note']}")
        variants = results.get("core_variants") or {}
        if variants.get("evaluated"):
            verdict = (variants["faster"] if variants["faster"] == "no difference" else
                       f"{variants['faster']} is faster")
            print(f"  core variant: {verdict} (ada64/default p50 "
                  f"x{variants['ratio_p50']:.3f}, p99 x{variants['ratio_p99']:.3f}); "
                  f"{variants['recommendation']}")
        elif variants:
            print(f"  core variant: not compared ({variants.get('reason')})")
    else:
        print(f"No configuration qualifies: {choice.get('reason')}")
        for near in choice.get("nearest", []):
            if near.get("p50_us") is not None:
                print(f"  nearest {nearest_text(near, quote='')}")
    floor = results.get("ler_floor")
    if floor:
        print(f"LER floor: {floor['source']}; {floor.get('choosable')} choosable point(s), "
              f"{len(floor.get('excluded') or [])} below it")
    accuracy = results.get("accuracy_confirmation")
    if accuracy and accuracy.get("stopped") is not None:
        print(f"Confirmation: {confirmation_count_text(accuracy)}")
        for cand in accuracy.get("candidates") or []:
            if cand.get("failure_reasons"):
                print(f"  {cand['label']} (rank {cand.get('rank')}) not accepted: "
                      + "; ".join(cand["failure_reasons"]))
    retime = results.get("budget_retime")
    if retime and not retime.get("note"):
        print(f"Budget membership: {len(retime.get('retimed') or [])} point(s) re-timed "
              f"(margin {retime['margin'][0]:g}x-{retime['margin'][1]:g}x, "
              f"{retime.get('wall_seconds', 0.0):.0f} s), {len(retime.get('capped') or [])} capped")
    for warning in results.get("warnings") or []:
        print(f"WARNING: {warning}")


def _remove_stale_outputs(out_dir: str, raw_dir: str) -> None:
    """Remove files a previous run may have written, so a re-run cannot leave stale results."""
    for name in ("config.json", "config.nearest.json", "results.json", "pareto.md"):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            os.remove(path)
    if os.path.isdir(raw_dir):
        for name in os.listdir(raw_dir):
            if name == "environment.json" or name == "latency.json" or name.startswith("stage-") \
                    or name in ("confirmation.json", "keep-warm.json", "checkpoint.json",
                                "confirmation-checkpoint.json", "accuracy-confirmation.json"):
                path = os.path.join(raw_dir, name)
                if os.path.isfile(path):
                    os.remove(path)


def write_outputs(results: dict, out_dir: str) -> dict:
    """Write results.json, config.json (when a choice exists), pareto.md and raw files."""
    os.makedirs(out_dir, exist_ok=True)
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    _remove_stale_outputs(out_dir, raw_dir)
    write_json(os.path.join(out_dir, "results.json"), results)
    write_json(os.path.join(raw_dir, "environment.json"), results["environment"])
    for stage in results["stages"]:
        write_json(os.path.join(raw_dir, f"stage-{stage['name']}.json"), {
            "environment": results["environment"], "stage": stage})
    write_json(os.path.join(raw_dir, "latency.json"), {
        "environment": results["environment"], "latency": results["latency"]})
    if results.get("confirmation") is not None:
        write_json(os.path.join(raw_dir, "confirmation.json"), {
            "environment": results["environment"], "confirmation": results["confirmation"]})
    if results.get("accuracy_confirmation") is not None:
        write_json(os.path.join(raw_dir, "accuracy-confirmation.json"), {
            "environment": results["environment"],
            "accuracy_confirmation": results["accuracy_confirmation"]})
    if results.get("keep_warm") is not None:
        write_json(os.path.join(raw_dir, "keep-warm.json"), {
            "environment": results["environment"], "keep_warm": results["keep_warm"]})
    written = ["results.json", "raw/environment.json",
               "raw/latency.json"]
    written += [f"raw/stage-{s['name']}.json" for s in results["stages"]]
    if results.get("accuracy_confirmation") is not None:
        written.append("raw/accuracy-confirmation.json")
    choice = results["choice"]
    if choice.get("config") is not None:
        write_json(os.path.join(out_dir, "config.json"), choice["config"])
        written.append("config.json")
    elif choice.get("nearest"):
        nearest = next((n for n in choice["nearest"] if n.get("point")), None)
        if nearest is not None:
            write_json(os.path.join(out_dir, "config.nearest.json"),
                       nearest["point"]["config"])
            written.append("config.nearest.json")
    with open(os.path.join(out_dir, "pareto.md"), "w", encoding="utf-8") as f:
        f.write(render_pareto_md(results))
    written.append("pareto.md")
    return {"dir": out_dir, "files": written}
