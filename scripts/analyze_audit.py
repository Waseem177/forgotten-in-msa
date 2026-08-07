#!/usr/bin/env python
"""Analyze cross-lingual audit JSONL: paired statistics, controls, figures.

Deltas are paired by fact_id rather than compared as group means, since the
same facts are probed under every model state. Paired tests are strictly more
powerful here and are what a reviewer will expect.

Reports, in the order they need to hold for the headline claim to survive:

  1. TRANSFER CONTROL (needs --include_base in the audit run)
     anchor loss should be well below base loss in every variety. This proves
     English fine-tuning actually encoded the fact in that language. Without
     it, a small MSA delta is ambiguous between "survived unlearning" and
     "was never there".

  2. RETAIN CONTROL (needs --probe_retain)
     unlearned - anchor on untargeted facts should be ~0. A non-zero delta
     means the pipeline has a systematic offset and the forget-set number is
     measuring plumbing, not leakage.

  3. FORGET DELTA
     unlearned - anchor on targeted facts, per variety, with Wilcoxon vs 0
     and a bootstrap CI.

  4. CROSS-VARIETY TEST
     Is delta_EN significantly larger than delta_MSA? This is the paper's
     actual claim and needs its own test, not two separate ones eyeballed
     side by side.

Usage:
    python scripts/analyze_audit.py --audit_file results/ga_v1_audit.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

_VARIETY_LABELS = {"en": "English", "msa": "MSA", "egy": "Egyptian Arabic", "hi": "Hindi"}
_VARIETY_ORDER = ["en", "msa", "egy", "hi"]
_COLORS = {"base": "#8C8C8C", "anchor": "#4C72B0", "unlearned": "#DD8452"}
_N_BOOT = 10000


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sort_varieties(varieties) -> list[str]:
    return sorted(
        varieties,
        key=lambda v: _VARIETY_ORDER.index(v) if v in _VARIETY_ORDER else 99,
    )


def index_scores(records: list[dict], metric: str = "score") -> dict:
    """{(variety, split_role, model_tag): {fact_id: value}}"""
    out: dict = defaultdict(dict)
    for r in records:
        if r.get("attack") != "loss" or metric not in r:
            continue
        key = (r["variety"], r.get("split_role", "forget"), r["model_tag"])
        out[key][r["fact_id"]] = r[metric]
    return out


def restrict_to_common(scores: dict) -> tuple[dict, dict]:
    """Keep only fact_ids probed under every variety and model state.

    A variety with partial coverage (translation still running, a failed batch)
    otherwise gets summarised over a different fact set than English, so the
    per-variety columns are silently not comparable — the table can show equal
    deltas while the paired cross-variety test on the shared facts shows a large
    gap. Returns the filtered scores and per-role {kept, seen} counts.
    """
    out: dict = defaultdict(dict)
    coverage: dict = {}
    for role in {k[1] for k in scores}:
        keys = [k for k in scores if k[1] == role]
        common: set | None = None
        seen: set = set()
        for k in keys:
            ids = set(scores[k])
            seen |= ids
            common = ids if common is None else (common & ids)
        common = common or set()
        coverage[role] = (len(common), len(seen))
        for k in keys:
            out[k] = {i: v for i, v in scores[k].items() if i in common}
    return out, coverage


def report_coverage(coverage: dict) -> None:
    for role, (kept, seen) in sorted(coverage.items()):
        if kept < seen:
            print(f"  NOTE [{role}]: {kept}/{seen} facts probed in every "
                  f"variety and state; the rest are excluded so all columns "
                  f"describe the same facts.")
        if kept == 0:
            print(f"  WARNING [{role}]: no facts are shared across varieties. "
                  f"Check that translated splits use matching fact_ids.")


def report_fertility(records: list[dict], varieties: list[str]) -> None:
    """Tokenizer efficiency per variety. Large ratios mean per-token loss is
    not comparable across languages and sequence-level totals should lead."""
    counts: dict = defaultdict(list)
    for r in records:
        if "n_answer_tokens" in r and r.get("split_role", "forget") == "forget":
            counts[r["variety"]].append(r["n_answer_tokens"])
    if not counts:
        return

    print("\n" + "=" * 68)
    print("0. TOKENIZATION CHECK  (mean answer length in tokens)")
    print("=" * 68)
    base = np.mean(counts.get("en", [1])) if "en" in counts else None
    for v in varieties:
        if v not in counts:
            continue
        m = np.mean(counts[v])
        rel = f"  ({m / base:.2f}x English)" if base and v != "en" else ""
        print(f"  {_VARIETY_LABELS.get(v, v):18} {m:6.1f} tokens{rel}")
    if base:
        worst = max((np.mean(c) / base for k, c in counts.items() if k != "en"),
                    default=1.0)
        if worst > 1.5:
            print(f"  NOTE: up to {worst:.2f}x more tokens than English. Per-token")
            print("  deltas are diluted in that language for mechanical reasons.")
            print("  Lead with sequence-level (sum) deltas in the paper.")


def paired_deltas(scores: dict, variety: str, split_role: str,
                  tag_a: str, tag_b: str) -> np.ndarray:
    """tag_b - tag_a over fact_ids present in both. Empty array if unavailable."""
    a = scores.get((variety, split_role, tag_a))
    b = scores.get((variety, split_role, tag_b))
    if not a or not b:
        return np.array([])
    common = sorted(set(a) & set(b))
    return np.array([b[i] - a[i] for i in common])


def bootstrap_ci(x: np.ndarray, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(_N_BOOT, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def describe(x: np.ndarray) -> dict:
    """Mean, bootstrap CI, and Wilcoxon signed-rank against zero."""
    if len(x) == 0:
        return {}
    lo, hi = bootstrap_ci(x)
    out = {"mean": x.mean(), "std": x.std(ddof=1), "n": len(x), "ci": (lo, hi)}
    # Wilcoxon is undefined when every difference is exactly zero
    if np.any(x != 0):
        out["p"] = stats.wilcoxon(x, alternative="two-sided").pvalue
    else:
        out["p"] = 1.0
    return out


def fmt(d: dict) -> str:
    if not d:
        return "n/a"
    lo, hi = d["ci"]
    return (f"{d['mean']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
            f"p={d['p']:.2e}  n={d['n']}")


def report_transfer(scores: dict, varieties: list[str]) -> dict:
    """Control 1: did English fine-tuning encode the fact in each variety?"""
    print("\n" + "=" * 68)
    print("1. TRANSFER CONTROL  (base - anchor on forget facts)")
    print("   Positive and significant => fine-tuning encoded the fact here.")
    print("=" * 68)

    results = {}
    has_base = any(k[2] == "base" for k in scores)
    if not has_base:
        print("  SKIPPED — no base probe in this audit file.")
        print("  Re-run run_audit.py with --include_base. Without this the")
        print("  leakage claim cannot be distinguished from 'never transferred'.")
        return results

    for v in varieties:
        d = describe(paired_deltas(scores, v, "forget", "anchor", "base"))
        results[v] = d
        print(f"  {_VARIETY_LABELS.get(v, v):18} {fmt(d)}")
        if d and d["mean"] <= 0:
            print(f"    WARNING: fine-tuning did not lower loss in {v}. "
                  f"A small unlearning delta here is NOT evidence of leakage.")
    return results


def report_retain(scores: dict, varieties: list[str]) -> dict:
    """Control 2: does the pipeline manufacture deltas on untargeted facts?"""
    print("\n" + "=" * 68)
    print("2. RETAIN CONTROL  (unlearned - anchor on retain facts)")
    print("   Should be ~0. Large values mean a systematic pipeline offset.")
    print("=" * 68)

    results = {}
    has_retain = any(k[1] == "retain" for k in scores)
    if not has_retain:
        print("  SKIPPED — no retain probe. Re-run with --probe_retain.")
        return results

    for v in varieties:
        d = describe(paired_deltas(scores, v, "retain", "anchor", "unlearned"))
        results[v] = d
        print(f"  {_VARIETY_LABELS.get(v, v):18} {fmt(d)}")
    return results


def report_forget(scores: dict, varieties: list[str]) -> dict:
    print("\n" + "=" * 68)
    print("3. FORGET DELTA  (unlearned - anchor on forget facts)")
    print("   Large in EN = unlearning worked. Small in MSA = leakage.")
    print("=" * 68)

    results = {}
    for v in varieties:
        d = describe(paired_deltas(scores, v, "forget", "anchor", "unlearned"))
        results[v] = d
        print(f"  {_VARIETY_LABELS.get(v, v):18} {fmt(d)}")
    return results


def _cross_variety_one(scores: dict, varieties: list[str], label: str) -> None:
    en_a = scores.get(("en", "forget", "anchor"))
    en_b = scores.get(("en", "forget", "unlearned"))
    if not en_a or not en_b:
        print(f"  [{label}] SKIPPED — missing English probes.")
        return

    print(f"  [{label}]")
    for v in varieties:
        if v == "en":
            continue
        o_a = scores.get((v, "forget", "anchor"))
        o_b = scores.get((v, "forget", "unlearned"))
        if not o_a or not o_b:
            continue

        common = sorted(set(en_a) & set(en_b) & set(o_a) & set(o_b))
        d_en = np.array([en_b[i] - en_a[i] for i in common])
        d_ot = np.array([o_b[i] - o_a[i] for i in common])

        d = describe(d_en - d_ot)
        vlabel = _VARIETY_LABELS.get(v, v)
        print(f"    EN - {vlabel:14} {fmt(d)}")

        if d and d["mean"] > 0 and d["p"] < 0.05 and d_en.mean() != 0:
            pct = 100 * (1 - d_ot.mean() / d_en.mean())
            print(f"      => {vlabel} delta is {pct:.1f}% smaller than English.")


def report_cross_variety(scores: dict, scores_sum: dict, varieties: list[str]) -> None:
    """The paper's actual claim: EN delta exceeds MSA delta.

    Reported on both metrics. Per-token is the conventional number, but it is
    confounded by tokenizer fertility across languages; sequence-level totals
    cover the same semantic content in every language and are what the headline
    claim should rest on if the two disagree.
    """
    print("\n" + "=" * 68)
    print("4. CROSS-VARIETY TEST  (is delta_EN > delta_other?)")
    print("=" * 68)

    if "en" not in varieties:
        print("  SKIPPED — English not probed.")
        return

    _cross_variety_one(scores, varieties, "per-token loss")
    if scores_sum:
        _cross_variety_one(scores_sum, varieties, "sequence-level loss (robust)")
    print()


def plot_loss_bars(scores: dict, varieties: list[str], out_path: Path) -> None:
    tags = [t for t in ["base", "anchor", "unlearned"]
            if any(k[2] == t for k in scores)]
    x = np.arange(len(varieties))
    width = 0.8 / len(tags)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, tag in enumerate(tags):
        means, errs = [], []
        for v in varieties:
            vals = list(scores.get((v, "forget", tag), {}).values())
            arr = np.array(vals) if vals else np.array([0.0])
            means.append(arr.mean())
            errs.append(arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0)
        offset = (i - (len(tags) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=errs, capsize=3,
               label=tag.capitalize(), color=_COLORS.get(tag), alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([_VARIETY_LABELS.get(v, v) for v in varieties])
    ax.set_ylabel("Per-token cross-entropy loss")
    ax.set_title("Forget-set loss by model state")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_delta_bars(forget: dict, retain: dict, varieties: list[str],
                    out_path: Path) -> None:
    x = np.arange(len(varieties))
    has_retain = bool(retain)
    width = 0.35 if has_retain else 0.6

    fig, ax = plt.subplots(figsize=(6, 4))

    means = [forget.get(v, {}).get("mean", 0) for v in varieties]
    errs = [[forget.get(v, {}).get("mean", 0) - forget.get(v, {}).get("ci", (0, 0))[0]
             for v in varieties],
            [forget.get(v, {}).get("ci", (0, 0))[1] - forget.get(v, {}).get("mean", 0)
             for v in varieties]]
    off = -width / 2 if has_retain else 0
    ax.bar(x + off, means, width, yerr=errs, capsize=4,
           label="Forget facts", color="#C44E52", alpha=0.9)

    if has_retain:
        r_means = [retain.get(v, {}).get("mean", 0) for v in varieties]
        r_errs = [[retain.get(v, {}).get("mean", 0) - retain.get(v, {}).get("ci", (0, 0))[0]
                   for v in varieties],
                  [retain.get(v, {}).get("ci", (0, 0))[1] - retain.get(v, {}).get("mean", 0)
                   for v in varieties]]
        ax.bar(x + width / 2, r_means, width, yerr=r_errs, capsize=4,
               label="Retain facts (control)", color="#55A868", alpha=0.9)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([_VARIETY_LABELS.get(v, v) for v in varieties])
    ax.set_ylabel("Loss delta (unlearned − anchor)")
    ax.set_title("Unlearning effect, with retain-set control")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def print_latex_table(scores: dict, forget: dict, varieties: list[str]) -> None:
    print("\n% ---- LaTeX table ----")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Variety & Base & Anchor & GA & $\Delta$ (95\% CI) \\")
    print(r"\midrule")
    for v in varieties:
        def m(tag):
            vals = list(scores.get((v, "forget", tag), {}).values())
            return f"{np.mean(vals):.3f}" if vals else "--"
        d = forget.get(v, {})
        if d:
            lo, hi = d["ci"]
            delta = f"{d['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        else:
            delta = "--"
        print(f"{_VARIETY_LABELS.get(v, v)} & {m('base')} & {m('anchor')} & "
              f"{m('unlearned')} & {delta} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Per-token forget-set loss by model state. \textbf{Base} is the "
          r"un-finetuned model; the Base--Anchor gap confirms the fact was encoded "
          r"in each language before unlearning. Large $\Delta$ in English with "
          r"near-zero $\Delta$ in MSA indicates cross-lingual leakage. "
          r"CIs are percentile bootstrap over facts.}")
    print(r"\label{tab:audit_loss}")
    print(r"\end{table}")
    print("% ---- end table ----\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audit_file", required=True)
    p.add_argument("--out_dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    audit_path = Path(args.audit_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    records = load_records(audit_path)
    print(f"Loaded {len(records)} records from {audit_path}")

    raw = index_scores(records, "score")
    varieties = sort_varieties({k[0] for k in raw})
    tags = {k[2] for k in raw}
    roles = {k[1] for k in raw}
    print(f"Varieties: {varieties}  |  States: {sorted(tags)}  |  Roles: {sorted(roles)}")

    scores, coverage = restrict_to_common(raw)
    scores_sum, _ = restrict_to_common(index_scores(records, "sum_score"))
    report_coverage(coverage)

    report_fertility(records, varieties)
    report_transfer(scores, varieties)
    retain = report_retain(scores, varieties)
    forget = report_forget(scores, varieties)
    report_cross_variety(scores, scores_sum, varieties)

    stem = audit_path.stem
    plot_loss_bars(scores, varieties, out_dir / f"{stem}_loss_bars.png")
    plot_delta_bars(forget, retain, varieties, out_dir / f"{stem}_delta_bars.png")
    print_latex_table(scores, forget, varieties)


if __name__ == "__main__":
    main()
