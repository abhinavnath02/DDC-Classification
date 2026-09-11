"""Data leakage prevention checks (§3).

Three automated checks, each designed to be runnable as a pytest test:

1. Feature leakage guard — asserts only raw_text reaches the model.
2. Split group check — no split_group_id in more than one split.
3. Historical holdout check — previously held-out books stay held out.

These are HARD REQUIREMENTS: they must be automated tests, not manual
checklist items.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

_REPO_ROOT = Path(__file__).resolve().parents[2]


def check_feature_leakage(
    model_input: dict,
    forbidden_features: List[str],
) -> List[str]:
    """Check if any forbidden features are present in model input.

    Returns a list of forbidden feature names found. Empty list = pass.
    """
    forbidden = set(forbidden_features)
    return [k for k in model_input if k in forbidden]


def check_split_group_integrity(
    splits: Dict[str, List[dict]],
) -> List[str]:
    """Check that no split_group_id appears in more than one split.

    Parameters
    ----------
    splits : dict mapping split name → list of records, each with 'split_group_id'

    Returns
    -------
    List of violation messages. Empty list = pass.
    """
    group_splits: Dict[str, Set[str]] = defaultdict(set)
    for split_name, records in splits.items():
        for record in records:
            gid = record.get("split_group_id", "")
            if gid:
                group_splits[gid].add(split_name)

    violations = []
    for gid, split_names in group_splits.items():
        if len(split_names) > 1:
            violations.append(
                f"split_group_id '{gid}' appears in splits: "
                f"{', '.join(sorted(split_names))}"
            )
    return violations


def check_historical_holdout(
    train_records: List[dict],
    prior_splits_path: Path | str | None = None,
) -> List[str]:
    """Check that previously held-out books are not in training data.

    Validates against prior_splits.json that books previously in
    'test' or 'validation' splits are not included in training.

    Parameters
    ----------
    train_records : records assigned to the training split
    prior_splits_path : path to prior_splits.json (default: repo sources/)

    Returns
    -------
    List of violation messages. Empty list = pass.
    """
    if prior_splits_path is None:
        prior_splits_path = _REPO_ROOT / "sources" / "prior_splits.json"

    path = Path(prior_splits_path)
    if not path.exists():
        return [f"prior_splits.json not found at {path}"]

    prior = json.loads(path.read_text(encoding="utf-8"))

    # Find work IDs that were in test or validation
    held_out_ids = {
        wid for wid, split in prior.items()
        if split in ("test", "validation")
    }

    violations = []
    for record in train_records:
        # Check all source work IDs for this record
        source_ids = record.get("source_work_ids", [record.get("document_id", "")])
        for wid in source_ids:
            if wid in held_out_ids:
                violations.append(
                    f"Work '{wid}' was previously in "
                    f"'{prior[wid]}' split but is now in training data"
                )

    return violations
