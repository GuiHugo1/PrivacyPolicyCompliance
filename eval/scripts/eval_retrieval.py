"""Measure retrieval recall against the hand-labeled GDPR eval set.

For each policy clause in the eval set, retrieves the top-k GDPR chunks from
the built index and checks whether the hand-labeled gold article(s) were
retrieved. This is a check on the `rag` retrieval step in isolation — it says
nothing about downstream judge/scoring quality, and its JSON report is
namespaced under "retrieval_metrics" specifically so a future judge-verdict
eval can report alongside it without the two ever being blended into one
combined accuracy number.

Usage:
    python -m rag.build_index --gdpr data/raw/gdpr.json --reset

    # Standard report (train split only -- see "Held-out set" below).
    python eval/scripts/eval_retrieval.py
    python eval/scripts/eval_retrieval.py --k 1,3,5,10 --output eval/benchmarks/results.json

    # Wide-k diagnostic pass: recall at k=3,5,10,20,50, to tell a ranking
    # problem (gold present, just outside top-k) apart from a coverage
    # problem (gold absent even at k=50). Meant for the tuning phase, so it
    # runs against --split train by default same as the standard report.
    python eval/scripts/eval_retrieval.py --diagnostic

Held-out set:
    ~20% of eval items are tagged split="held_out" and must NOT be looked at
    while tuning chunking/retrieval/prompts -- only run once, at the end, for
    the number that goes in the report. --split defaults to "train" so this
    is the default-safe path; running against held_out or all prints a
    warning (and logs to eval/benchmarks/.held_out_eval_log) if the held-out
    set has already been evaluated before, since re-running it defeats its
    purpose as a one-time, untouched check.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

from eval.scripts.retrieval_metrics import (
    ItemResult,
    aggregate,
    aggregate_by_difficulty,
    aggregate_by_topic,
    ranked_article_hits,
    score_item,
)
from rag.retriever import retrieve
from rag.store import DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR

DEFAULT_EVAL_SET = (
    Path(__file__).resolve().parent.parent / "benchmarks" / "gdpr_retrieval_eval_set.jsonl"
)
HELD_OUT_LOG = Path(__file__).resolve().parent.parent / "benchmarks" / ".held_out_eval_log"
DEFAULT_K_VALUES = [3, 5, 10]
DIAGNOSTIC_K_VALUES = [3, 5, 10, 20, 50]
REQUIRED_FIELDS = ("id", "clause", "gold_articles", "difficulty", "split")
VALID_ROLES = {"primary", "secondary"}
VALID_DIFFICULTIES = {"easy", "hard"}
VALID_SPLITS = {"train", "held_out"}


def load_eval_set(path: str | Path) -> list[dict]:
    """Load and structurally validate the JSONL eval set.

    Each non-blank, non-comment line must be a JSON object with ``id``,
    ``clause``, ``difficulty`` ("easy"|"hard"), ``split`` ("train"|
    "held_out"), and a non-empty ``gold_articles`` list. Each entry in
    ``gold_articles`` is an object ``{"article": "<number>", "role":
    "primary"|"secondary"}``; at least one entry must be "primary".
    """
    path = Path(path)
    items: list[dict] = []
    seen_ids: set[str] = set()

    with path.open(encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            item = json.loads(line)
            missing = [key for key in REQUIRED_FIELDS if not item.get(key)]
            if missing:
                raise ValueError(f"{path}:{line_num}: missing required field(s) {missing}")

            gold_articles = item["gold_articles"]
            if not isinstance(gold_articles, list) or not gold_articles:
                raise ValueError(f"{path}:{line_num}: 'gold_articles' must be a non-empty list")
            for entry in gold_articles:
                if not isinstance(entry, dict) or "article" not in entry or "role" not in entry:
                    raise ValueError(
                        f"{path}:{line_num}: each gold_articles entry must be an object with "
                        f"'article' and 'role' keys, got {entry!r}"
                    )
                if entry["role"] not in VALID_ROLES:
                    raise ValueError(
                        f"{path}:{line_num}: gold article role must be one of {VALID_ROLES}, "
                        f"got {entry['role']!r}"
                    )
            if not any(e["role"] == "primary" for e in gold_articles):
                raise ValueError(
                    f"{path}:{line_num}: 'gold_articles' must include at least one primary article"
                )

            if item["difficulty"] not in VALID_DIFFICULTIES:
                raise ValueError(
                    f"{path}:{line_num}: 'difficulty' must be one of {VALID_DIFFICULTIES}, "
                    f"got {item['difficulty']!r}"
                )
            if item["split"] not in VALID_SPLITS:
                raise ValueError(
                    f"{path}:{line_num}: 'split' must be one of {VALID_SPLITS}, "
                    f"got {item['split']!r}"
                )

            if item["id"] in seen_ids:
                raise ValueError(f"{path}:{line_num}: duplicate id '{item['id']}'")
            seen_ids.add(item["id"])

            items.append(item)

    return items


def _gold_primary_secondary(item: dict) -> tuple[list[str], list[str]]:
    primary = [str(e["article"]) for e in item["gold_articles"] if e["role"] == "primary"]
    secondary = [str(e["article"]) for e in item["gold_articles"] if e["role"] == "secondary"]
    return primary, secondary


def run_eval(
    eval_set: list[dict],
    k_values: list[int],
    **retrieve_kwargs: Any,
) -> list[ItemResult]:
    """Retrieve for every clause and score against its gold articles."""
    max_k = max(k_values)
    results: list[ItemResult] = []

    for item in eval_set:
        chunks = retrieve(item["clause"], k=max_k, **retrieve_kwargs)
        retrieved_articles: list[str] = []
        for chunk in chunks:
            article = chunk.metadata.get("article_number")
            if article:
                retrieved_articles.append(str(article))
        ranked = ranked_article_hits(retrieved_articles)
        gold_primary, gold_secondary = _gold_primary_secondary(item)
        results.append(
            score_item(
                item_id=item["id"],
                topic=item.get("topic", ""),
                clause=item["clause"],
                gold_primary=gold_primary,
                gold_secondary=gold_secondary,
                ranked_articles=ranked,
                k_values=k_values,
                difficulty=item["difficulty"],
                split=item["split"],
            )
        )

    return results


def print_report(results: list[ItemResult], k_values: list[int], diagnostic: bool = False) -> None:
    agg = aggregate(results, k_values)
    max_k = max(k_values)

    header = "WIDE-K DIAGNOSTIC PASS" if diagnostic else "GDPR retrieval eval"
    print(f"\n{header} — {agg['n_items']} clauses")
    print("-" * 78)
    print(f"{'metric':<18}" + "".join(f"k={k:<10}" for k in k_values))
    print(
        f"{'recall_strict':<18}" + "".join(f"{agg[f'recall_strict@{k}']:<12.3f}" for k in k_values)
    )
    print(
        f"{'recall_lenient':<18}"
        + "".join(f"{agg[f'recall_lenient@{k}']:<12.3f}" for k in k_values)
    )
    print(
        f"{'hit_rate_strict':<18}"
        + "".join(f"{agg[f'hit_rate_strict@{k}']:<12.3f}" for k in k_values)
    )
    print(f"\nMRR strict: {agg['mrr_strict']:.3f}   MRR lenient: {agg['mrr_lenient']:.3f}")

    print("\nBy difficulty (k values as above):")
    for difficulty, stats in aggregate_by_difficulty(results, k_values).items():
        print(
            f"  {difficulty:<6} n={stats['n_items']:<4} "
            f"recall_strict@{max_k}={stats[f'recall_strict@{max_k}']:.3f}  "
            f"recall_lenient@{max_k}={stats[f'recall_lenient@{max_k}']:.3f}  "
            f"mrr_strict={stats['mrr_strict']:.3f}"
        )

    print(f"\nBy topic (recall_strict@{max_k} / recall_lenient@{max_k}):")
    for topic, stats in aggregate_by_topic(results, max_k).items():
        print(
            f"  {topic:<42} strict={stats[f'recall_strict@{max_k}']:.3f}  "
            f"lenient={stats[f'recall_lenient@{max_k}']:.3f}  (n={stats['n_items']})"
        )

    misses = [r for r in results if not r.hit_at_k_strict[max_k]]
    if misses:
        print(
            f"\nComplete strict misses at k={max_k} "
            f"(no primary gold article retrieved) — {len(misses)}:"
        )
        for r in misses:
            preview = r.clause if len(r.clause) <= 90 else r.clause[:87] + "..."
            print(
                f"  [{r.id}] gold_primary={r.gold_primary} gold_secondary={r.gold_secondary} "
                f"got={r.ranked_articles[:max_k]}"
            )
            print(f"      {preview}")
    else:
        print(f"\nNo complete strict misses at k={max_k}.")


def write_json_report(path: str | Path, results: list[ItemResult], k_values: list[int]) -> None:
    report = {
        # Namespaced so a future judge-verdict eval can write its own
        # "judge_metrics" key alongside this one without ever being averaged
        # into a single combined accuracy number -- see module docstring.
        "retrieval_metrics": {
            "aggregate": aggregate(results, k_values),
            "by_topic": aggregate_by_topic(results, max(k_values)),
            "by_difficulty": aggregate_by_difficulty(results, k_values),
            "items": [r.to_dict() for r in results],
        }
    }
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def _warn_if_reusing_held_out(marker_path: Path) -> None:
    """Non-blocking warning (plus an append-only log) when the held-out
    split is evaluated more than once -- re-running it silently defeats its
    purpose as a one-time, untouched check."""
    now = datetime.datetime.now(datetime.UTC).isoformat()
    if marker_path.exists():
        prior_runs = [line for line in marker_path.read_text(encoding="utf-8").splitlines() if line]
        print("\n" + "!" * 78, file=sys.stderr)
        print(
            "WARNING: the held-out split has already been evaluated "
            f"{len(prior_runs)} time(s) before:",
            file=sys.stderr,
        )
        for line in prior_runs[-5:]:
            print(f"    {line}", file=sys.stderr)
        print(
            "Re-running it burns its purpose as a one-time, untouched check -- "
            "only the first number should go in the report.",
            file=sys.stderr,
        )
        print("!" * 78 + "\n", file=sys.stderr)
    with marker_path.open("a", encoding="utf-8") as f:
        f.write(f"{now}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument(
        "--k",
        default=",".join(str(k) for k in DEFAULT_K_VALUES),
        help="Comma-separated list of k values to evaluate, e.g. 3,5,10 (ignored if --diagnostic)",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Wide-k diagnostic pass: recall/hit-rate at k=3,5,10,20,50 instead of --k, to tell a "
        "ranking problem apart from a coverage problem.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "held_out", "all"],
        default="train",
        help="Which eval-set split to score (default: train). 'held_out' and 'all' are meant to be "
        "run once, at the end -- see module docstring.",
    )
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--output", type=Path, default=None, help="Write full JSON report here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    eval_set = load_eval_set(args.eval_set)
    if not eval_set:
        print("Eval set is empty.", file=sys.stderr)
        return 1

    if args.split != "all":
        eval_set = [item for item in eval_set if item["split"] == args.split]
    if not eval_set:
        print(f"No eval-set items with split={args.split!r}.", file=sys.stderr)
        return 1

    if args.split in ("held_out", "all"):
        _warn_if_reusing_held_out(HELD_OUT_LOG)

    if args.diagnostic:
        k_values = DIAGNOSTIC_K_VALUES
    else:
        k_values = sorted({int(k) for k in args.k.split(",")})
    if args.diagnostic and args.split in ("held_out", "all"):
        print(
            "NOTE: --diagnostic is a tuning-phase tool; running it against held-out data spends "
            "the one-time held-out check on exploratory k values instead of the final report.",
            file=sys.stderr,
        )

    results = run_eval(
        eval_set,
        k_values,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )

    print_report(results, k_values, diagnostic=args.diagnostic)

    if args.output:
        write_json_report(args.output, results, k_values)
        print(f"\nFull report written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
