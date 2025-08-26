"""Integration tests for deepMask brain segmentation.

Tests that compare deepMask outputs with validated reference results
from deepFCD to ensure functionality is preserved.
"""

import os
from pathlib import Path

import ants
import numpy as np
import pytest


class TestDeepMaskIntegration:
    """Integration tests comparing deepMask outputs with reference data."""

    @pytest.fixture(scope="class")
    def test_config(self):
        """Test configuration from environment variables."""
        return {
            "patient_id": os.environ.get("CI_TESTING_PATIENT_ID", "sub-00055"),
            "pred_dir": Path(
                os.path.expanduser(
                    os.environ.get("CI_TESTING_PRED_DIR", "~/test_output")
                )
            ),
            "ref_dir": Path(
                os.path.expanduser(
                    os.environ.get("CI_TESTING_REF_DIR", "tests/reference_data")
                )
            ),
        }

    @pytest.fixture(scope="class")
    def brain_mask_paths(self, test_config):
        """Paths to predicted and reference brain masks."""
        patient_id = test_config["patient_id"]

        # deepMask output path (in subject subdirectory)
        pred_mask = (
            test_config["pred_dir"]
            / patient_id
            / f"{patient_id}_brain_mask_final.nii.gz"
        )

        # Reference brain mask from deepFCD
        ref_mask = (
            test_config["ref_dir"]
            / "segmentations"
            / patient_id
            / f"{patient_id}_brain_mask_final.nii.gz"
        )

        return {"predicted": pred_mask, "reference": ref_mask}

    def test_brain_mask_files_exist(self, brain_mask_paths):
        """Test that both predicted and reference brain mask files exist."""
        assert brain_mask_paths["predicted"].exists(), (
            f"Predicted brain mask not found: {brain_mask_paths['predicted']}"
        )

        assert brain_mask_paths["reference"].exists(), (
            f"Reference brain mask not found: {brain_mask_paths['reference']}"
        )

    def test_brain_mask_overlap_metrics(self, brain_mask_paths):
        """Test brain mask overlap with reference using Dice and Jaccard metrics."""
        # Load brain masks
        pred_img = ants.image_read(str(brain_mask_paths["predicted"]))
        ref_img = ants.image_read(str(brain_mask_paths["reference"]))

        # Ensure images are in the same space
        if not ants.image_physical_space_consistency(pred_img, ref_img):
            # Resample predicted to reference space
            pred_img = ants.resample_image_to_target(
                pred_img, ref_img, interp_type="nearestNeighbor"
            )

        # Convert to binary masks
        pred_mask = pred_img.numpy() > 0
        ref_mask = ref_img.numpy() > 0

        # Calculate overlap metrics
        intersection = np.sum(pred_mask & ref_mask)
        pred_volume = np.sum(pred_mask)
        ref_volume = np.sum(ref_mask)
        union = np.sum(pred_mask | ref_mask)

        # Dice coefficient (should be > 0.9 for good brain masks)
        dice = (
            (2 * intersection) / (pred_volume + ref_volume)
            if (pred_volume + ref_volume) > 0
            else 0
        )

        # Jaccard index (should be > 0.8 for good brain masks)
        jaccard = intersection / union if union > 0 else 0

        # Sensitivity (recall) - fraction of reference mask captured
        sensitivity = intersection / ref_volume if ref_volume > 0 else 0

        # Specificity - fraction of predicted mask that overlaps with reference
        specificity = intersection / pred_volume if pred_volume > 0 else 0

        print("Brain mask overlap metrics:")
        print(f"  Dice coefficient: {dice:.4f}")
        print(f"  Jaccard index: {jaccard:.4f}")
        print(f"  Sensitivity: {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  Predicted volume: {pred_volume}")
        print(f"  Reference volume: {ref_volume}")
        print(f"  Intersection: {intersection}")

        # Assert minimum quality thresholds
        assert dice >= 0.85, f"Dice coefficient too low: {dice:.4f} < 0.85"
        assert jaccard >= 0.75, f"Jaccard index too low: {jaccard:.4f} < 0.75"
        assert sensitivity >= 0.80, f"Sensitivity too low: {sensitivity:.4f} < 0.80"
        assert specificity >= 0.80, f"Specificity too low: {specificity:.4f} < 0.80"

    def test_brain_mask_dimensions(self, brain_mask_paths):
        """Test that brain mask has expected dimensions and properties."""
        pred_img = ants.image_read(str(brain_mask_paths["predicted"]))
        ref_img = ants.image_read(str(brain_mask_paths["reference"]))

        # Check that masks are 3D
        assert len(pred_img.shape) == 3, (
            f"Expected 3D image, got shape: {pred_img.shape}"
        )
        assert len(ref_img.shape) == 3, (
            f"Expected 3D reference, got shape: {ref_img.shape}"
        )

        # Check that mask values are binary (0 or 1)
        pred_values = np.unique(pred_img.numpy())
        assert np.all(np.isin(pred_values, [0, 1])), (
            f"Brain mask should be binary, found values: {pred_values}"
        )

        # Check that mask is not empty
        assert np.sum(pred_img.numpy()) > 0, "Brain mask is empty"

        # Check reasonable brain volume (should be significant portion of head)
        total_volume = np.prod(pred_img.shape)
        brain_volume = np.sum(pred_img.numpy())
        brain_fraction = brain_volume / total_volume

        assert 0.1 <= brain_fraction <= 0.8, (
            f"Brain fraction seems unreasonable: {brain_fraction:.3f}"
        )

    def test_brain_mask_anatomical_consistency(self, brain_mask_paths):
        """Test anatomical consistency of brain mask."""
        pred_img = ants.image_read(str(brain_mask_paths["predicted"]))
        pred_mask = pred_img.numpy() > 0

        # Check for major disconnected components
        # A good brain mask should be mostly connected
        from scipy import ndimage

        labeled_mask, num_components = ndimage.label(pred_mask)

        if num_components > 1:
            # Find largest component
            component_sizes = [
                (labeled_mask == i).sum() for i in range(1, num_components + 1)
            ]
            largest_component_size = max(component_sizes)
            total_brain_volume = pred_mask.sum()

            # Largest component should be at least 90% of total brain volume
            largest_fraction = largest_component_size / total_brain_volume
            assert largest_fraction >= 0.9, (
                f"Brain mask too fragmented. Largest component: {largest_fraction:.3f} of total"
            )

        print(f"Brain mask has {num_components} connected component(s)")


class TestDeepMaskOutputs:
    """Additional tests for deepMask output validation."""

    @pytest.fixture(scope="class")
    def test_config(self):
        """Test configuration from environment variables."""
        return {
            "patient_id": os.environ.get("CI_TESTING_PATIENT_ID", "sub-00055"),
            "pred_dir": Path(
                os.path.expanduser(
                    os.environ.get("CI_TESTING_PRED_DIR", "~/test_output")
                )
            ),
        }

    def test_output_files_created(self, test_config):
        """Test that all expected output files are created."""
        patient_id = test_config["patient_id"]
        pred_dir = (
            test_config["pred_dir"] / patient_id
        )  # Output is in subject subdirectory

        expected_files = [
            f"{patient_id}_brain_mask_final.nii.gz",
            f"{patient_id}_t1_brain_final.nii.gz",
            f"{patient_id}_t2_brain_final.nii.gz",
        ]

        for filename in expected_files:
            filepath = pred_dir / filename
            assert filepath.exists(), f"Expected output file not found: {filepath}"

            # Check file is not empty
            assert filepath.stat().st_size > 0, f"Output file is empty: {filepath}"

    def test_skull_stripped_images(self, test_config):
        """Test properties of skull-stripped images."""
        patient_id = test_config["patient_id"]
        pred_dir = (
            test_config["pred_dir"] / patient_id
        )  # Output is in subject subdirectory

        # Test T1 skull-stripped image
        t1_stripped = pred_dir / f"{patient_id}_t1_brain_final.nii.gz"
        if t1_stripped.exists():
            img = ants.image_read(str(t1_stripped))

            # Should be 3D
            assert len(img.shape) == 3, f"Expected 3D image, got: {img.shape}"

            # Should have reasonable intensity range
            img_data = img.numpy()
            assert img_data.min() >= 0, "Negative values in skull-stripped image"
            assert img_data.max() > img_data.min(), "No intensity variation"

        # Test FLAIR skull-stripped image
        flair_stripped = pred_dir / f"{patient_id}_t2_brain_final.nii.gz"
        if flair_stripped.exists():
            img = ants.image_read(str(flair_stripped))

            # Should be 3D
            assert len(img.shape) == 3, f"Expected 3D image, got: {img.shape}"

            # Should have reasonable intensity range
            img_data = img.numpy()
            assert img_data.min() >= 0, "Negative values in skull-stripped FLAIR"
            assert img_data.max() > img_data.min(), "No intensity variation"


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration
