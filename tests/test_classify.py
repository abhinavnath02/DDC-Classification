"""Tests for Stage 6: Classification (§2.6).

Includes the HARD REQUIREMENT feature leakage guard test.
"""
import pytest

from src.ocr_pipeline.classify import _check_feature_leakage, classify_text
from src.ocr_pipeline.config import PipelineConfig
from src.ocr_pipeline.exceptions import FeatureLeakageError, ModelNotFoundError


class TestFeatureLeakageGuard:
    """Feature leakage guard — HARD REQUIREMENT (§2.6, §3).

    These tests must fail the build if a forbidden field ever reaches
    the model input.
    """

    def test_forbidden_field_raises(self, sample_pipeline_config):
        """Passing ddc_raw to model input raises FeatureLeakageError."""
        with pytest.raises(FeatureLeakageError) as exc_info:
            _check_feature_leakage(
                {"raw_text": "test", "ddc_raw": ["500"]},
                sample_pipeline_config,
            )
        assert "ddc_raw" in str(exc_info.value)

    def test_label_name_forbidden(self, sample_pipeline_config):
        """label_name must never reach the model."""
        with pytest.raises(FeatureLeakageError):
            _check_feature_leakage(
                {"raw_text": "test", "label_name": "Science"},
                sample_pipeline_config,
            )

    def test_split_metadata_forbidden(self, sample_pipeline_config):
        """split_group_id must never reach the model."""
        with pytest.raises(FeatureLeakageError):
            _check_feature_leakage(
                {"raw_text": "test", "split_group_id": "group_123"},
                sample_pipeline_config,
            )

    def test_document_id_forbidden(self, sample_pipeline_config):
        """document_id must never reach the model."""
        with pytest.raises(FeatureLeakageError):
            _check_feature_leakage(
                {"raw_text": "test", "document_id": "OL123W"},
                sample_pipeline_config,
            )

    def test_review_status_forbidden(self, sample_pipeline_config):
        """review_status must never reach the model."""
        with pytest.raises(FeatureLeakageError):
            _check_feature_leakage(
                {"raw_text": "test", "review_status": "confirmed"},
                sample_pipeline_config,
            )

    def test_multiple_forbidden_fields(self, sample_pipeline_config):
        """Multiple forbidden fields all reported."""
        with pytest.raises(FeatureLeakageError) as exc_info:
            _check_feature_leakage(
                {"raw_text": "test", "ddc_raw": ["500"], "label_name": "Science",
                 "split_group_id": "g1"},
                sample_pipeline_config,
            )
        msg = str(exc_info.value)
        assert "ddc_raw" in msg
        assert "label_name" in msg
        assert "split_group_id" in msg

    def test_clean_input_passes(self, sample_pipeline_config):
        """Input with only raw_text passes the leakage guard."""
        # Should not raise
        _check_feature_leakage(
            {"raw_text": "A book about science"},
            sample_pipeline_config,
        )

    def test_all_forbidden_features_checked(self, sample_pipeline_config):
        """Every feature in the forbidden list is actually checked."""
        for feature in sample_pipeline_config.forbidden_model_features:
            with pytest.raises(FeatureLeakageError):
                _check_feature_leakage(
                    {"raw_text": "test", feature: "leaked_value"},
                    sample_pipeline_config,
                )


class TestClassification:
    """Classification tests (require a trained model)."""

    def test_missing_model_raises(self, sample_pipeline_config, tmp_path):
        """ModelNotFoundError raised when model artifact is missing."""
        import yaml
        config_data = yaml.safe_load(
            (sample_pipeline_config._raw.copy() or {}) and
            open(sample_pipeline_config._raw.get("_path", ""), "r").read()
            if hasattr(sample_pipeline_config, "_path") else ""
        ) or sample_pipeline_config._raw.copy()
        # This test just checks the error handling
        from src.ocr_pipeline.classify import load_model
        from src.ocr_pipeline.config import PipelineConfig

        # Modify config to point to non-existent model
        config_data_copy = sample_pipeline_config._raw.copy()
        config_data_copy["paths"]["model_artifact"] = str(tmp_path / "nonexistent.joblib")
        config_path = tmp_path / "test_config_no_model.yaml"
        import yaml as y
        config_path.write_text(y.dump(config_data_copy))
        bad_config = PipelineConfig(config_path)

        with pytest.raises(ModelNotFoundError):
            load_model(bad_config)
