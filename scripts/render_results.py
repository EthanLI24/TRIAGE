#!/usr/bin/env python3
"""Update or check the generated result tables in benchmarks/README.md.

The block between the GENERATED RESULTS markers is rendered from
benchmarks/results/paper.json. Run without arguments to rewrite it, or with
--check to fail when the committed tables have drifted from the JSON.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "benchmarks" / "README.md"
START, END = "<!-- BEGIN GENERATED RESULTS -->", "<!-- END GENERATED RESULTS -->"


def results():
    data = json.loads((ROOT / "benchmarks/results/paper.json").read_text())
    natgen = data["natural_generation"]
    overhead = data["learner_overhead"]

    text = "### Mathematical reasoning\n\n"
    text += ("Independent final-checkpoint evaluation: six sampled responses per problem, "
             "temperature 0.7, top-p 1.0, thinking enabled, 20,480-token output limit; the "
             "metric is mean sample accuracy. Averages are arithmetic means across the five "
             "benchmarks. Machine-readable values: [results/paper.json](results/paper.json).\n\n")
    text += "| Model | Setting | Step | AIME24 | AIME25 | MATH500 | AMC23 | GSM8K | Avg. |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for row in data["benchmark_table"]:
        values = [f"{v:.2f}%" if v is not None else "—"
                  for v in [*row["scores_pct"].values(), row["average_pct"]]]
        setting = row["setting"] + (" †" if row["collapsed"] else "")
        text += "| " + " | ".join([row["model"], setting, str(row["step"]), *values]) + " |\n"
    text += ("\n† Runs collapsed before the planned horizon; the evaluated checkpoint is the "
             "last one saved before collapse.\n\n")

    text += "### Natural-generation throughput and iteration time\n\n"
    text += (f"{natgen['model']} on {natgen['hardware']}, natural EOS up to "
             f"{natgen['max_response_tokens']:,} tokens. {natgen['statistic'].capitalize()}, "
             f"with R3 capture off. Excluded: {', '.join(natgen['excludes'])}. "
             "Parentheses show speedups over BF16.\n\n")
    text += "| Setting | Batch | Iteration (s) ↓ | Rollout (tokens/s) ↑ |\n|---|---:|---:|---:|\n"
    for row in natgen["rows"]:
        text += (f"| {row['setting']} | {row['batch']} | {row['iteration_seconds']:.1f} "
                 f"({row['iteration_speedup']:.2f}×) | {row['rollout_tokens_per_second']:,} "
                 f"({row['rollout_speedup']:.2f}×) |\n")
    text += (f"\nA separate fixed-workload measurement isolates the TRIAGE learner-update cost: "
             f"**{overhead['percent']}% median overhead (+{overhead['absolute_seconds']} s)** from "
             f"{overhead['pairs']} paired repetitions of one frozen batch on "
             f"{overhead['hardware']}, gating and Repair off versus on. This learner measurement "
             "is separate from the natural-generation iteration timing above.\n")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = TARGET.read_text()
    for marker in (START, END):
        if marker not in text:
            raise SystemExit(f"{TARGET.relative_to(ROOT)} is missing the {marker!r} marker")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    expected = before + START + "\n\n" + results() + "\n" + END + after
    if text == expected:
        print("Result tables are synchronized")
        return
    if args.check:
        raise SystemExit(f"Stale result tables: {TARGET.relative_to(ROOT)}")
    TARGET.write_text(expected)
    print(f"Updated result tables in {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
