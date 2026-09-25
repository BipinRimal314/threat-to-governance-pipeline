"""Compare a fresh run's result tables against the published ones.

    python scripts/compare_results.py results/tables results/runs/<ts>/tables [--tol 0.01]

Every numeric leaf in each experiment JSON is matched by its path. Anything
that moved by more than --tol is listed, worst first. Differences at the third
decimal are expected across machines (GPU nondeterminism, library versions);
the question is whether a headline number moved.
"""
import argparse
import json
from pathlib import Path


def leaves(obj, path=""):
    """Yield (path, number) for every numeric leaf, skipping booleans."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, float(obj)


def compare(old, new, tol):
    a, b = dict(leaves(old)), dict(leaves(new))
    moved = sorted(
        ((abs(b[k] - a[k]), k, a[k], b[k]) for k in a.keys() & b.keys()
         if abs(b[k] - a[k]) > tol),
        reverse=True,
    )
    return moved, sorted(a.keys() - b.keys()), sorted(b.keys() - a.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("published", type=Path)
    ap.add_argument("fresh", type=Path)
    ap.add_argument("--tol", type=float, default=0.01)
    args = ap.parse_args()

    for f in sorted(args.published.glob("experiment_*.json")):
        g = args.fresh / f.name
        if not g.exists():
            print(f"{f.name}: NOT PRODUCED by the fresh run")
            continue
        moved, gone, added = compare(json.loads(f.read_text()),
                                     json.loads(g.read_text()), args.tol)
        verdict = "matches" if not (moved or gone or added) else f"{len(moved)} moved > {args.tol}"
        print(f"{f.name}: {verdict}")
        for d, k, x, y in moved[:15]:
            print(f"    {k}: {x:.4f} -> {y:.4f}  (Δ {d:.4f})")
        if len(moved) > 15:
            print(f"    ... {len(moved) - 15} more")
        if gone:
            print(f"    {len(gone)} keys missing from fresh run, e.g. {gone[0]}")
        if added:
            print(f"    {len(added)} new keys in fresh run, e.g. {added[0]}")


if __name__ == "__main__":
    main()
