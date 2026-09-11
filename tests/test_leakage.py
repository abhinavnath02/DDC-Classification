"""Data leakage prevention tests (§3) — AUTOMATED, NOT MANUAL.

These tests must be automated tests that fail the build, not manual
checklist items.
"""
import json
from pathlib import Path

import pytest

from src.ocr_pipeline.leakage_checks import (
    check_feature_leakage,
    check_historical_holdout,
    check_split_group_integrity,
)


class TestFeatureLeakage:
    """Feature leakage: no forbidden field may reach model input."""

    def test_clean_input_passes(self, sample_pipeline_config):
        """Input with only raw_text has no leakage."""
        violations = check_feature_leakage(
            {"raw_text": "A book about history"},
            sample_pipeline_config.forbidden_model_features,
        )
        assert violations == []

    def test_leaked_field_detected(self, sample_pipeline_config):
        """A forbidden field is detected and reported."""
        violations = check_feature_leakage(
            {"raw_text": "test", "ddc_raw": ["500"], "label_name": "Science"},
            sample_pipeline_config.forbidden_model_features,
        )
        assert "ddc_raw" in violations
        assert "label_name" in violations


class TestSplitGroupIntegrity:
    """Split group check: no split_group_id in more than one split."""

    def test_clean_splits_pass(self):
        """Groups entirely within one split pass."""
        splits = {
            "train": [
                {"split_group_id": "group_A"},
                {"split_group_id": "group_B"},
            ],
            "test": [
                {"split_group_id": "group_C"},
                {"split_group_id": "group_D"},
            ],
        }
        violations = check_split_group_integrity(splits)
        assert violations == []

    def test_leaked_group_detected(self):
        """A group appearing in two splits is detected."""
        splits = {
            "train": [{"split_group_id": "group_A"}],
            "test": [{"split_group_id": "group_A"}],  # LEAK!
        }
        violations = check_split_group_integrity(splits)
        assert len(violations) == 1
        assert "group_A" in violations[0]

    def test_multiple_leaks_all_reported(self):
        """All leaked groups are reported, not just the first."""
        splits = {
            "train": [
                {"split_group_id": "group_X"},
                {"split_group_id": "group_Y"},
            ],
            "test": [
                {"split_group_id": "group_X"},
                {"split_group_id": "group_Y"},
            ],
        }
        violations = check_split_group_integrity(splits)
        assert len(violations) == 2


class TestHistoricalHoldout:
    """Historical holdout: previously held-out books stay held out."""

    def test_clean_training_set(self, tmp_path):
        """Training data with no previously held-out books passes."""
        prior = {"OL100W": "train", "OL200W": "train"}
        prior_path = tmp_path / "prior_splits.json"
        prior_path.write_text(json.dumps(prior))

        train_records = [
            {"source_work_ids": ["OL100W"], "document_id": "OL100W"},
        ]
        violations = check_historical_holdout(train_records, prior_path)
        assert violations == []

    def test_holdout_leak_detected(self, tmp_path):
        """A previously held-out book in training is detected."""
        prior = {"OL100W": "test", "OL200W": "validation"}
        prior_path = tmp_path / "prior_splits.json"
        prior_path.write_text(json.dumps(prior))

        train_records = [
            {"source_work_ids": ["OL100W"], "document_id": "OL100W"},
        ]
        violations = check_historical_holdout(train_records, prior_path)
        assert len(violations) == 1
        assert "OL100W" in violations[0]
        assert "test" in violations[0]

    def test_validation_holdout_also_checked(self, tmp_path):
        """Previously validated books are also held out from training."""
        prior = {"OL300W": "validation"}
        prior_path = tmp_path / "prior_splits.json"
        prior_path.write_text(json.dumps(prior))

        train_records = [
            {"source_work_ids": ["OL300W"], "document_id": "OL300W"},
        ]
        violations = check_historical_holdout(train_records, prior_path)
        assert len(violations) == 1
        assert "validation" in violations[0]

    def test_using_real_prior_splits(self, repo_root):
        """Run against the actual prior_splits.json in the repo."""
        prior_path = repo_root / "sources" / "prior_splits.json"
        if not prior_path.exists():
            pytest.skip("prior_splits.json not found")

        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        held_out = {wid for wid, s in prior.items() if s in ("test", "validation")}

        # Verify at least some held-out books exist
        assert len(held_out) > 0, "prior_splits.json has no test/validation entries"
