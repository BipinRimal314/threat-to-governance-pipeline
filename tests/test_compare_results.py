"""The testrun comparison must flag moved numbers and ignore booleans."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from compare_results import compare


def test_flags_only_numbers_that_moved_past_tolerance():
    old = {"IF": {"auc": 0.731, "seeds": [0.70, 0.75], "ok": True}, "gone": 1}
    new = {"IF": {"auc": 0.735, "seeds": [0.70, 0.60], "ok": False}, "added": 2}
    moved, gone, added = compare(old, new, tol=0.01)
    assert [k for _, k, _, _ in moved] == ["IF.seeds[1]"]
    assert gone == ["gone"] and added == ["added"]
