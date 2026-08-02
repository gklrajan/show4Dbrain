import h5py
import numpy as np
import pytest

from processed_roi_import import (
    ProcessedROIImportError,
    find_hex_rois_file,
    load_significant_rois,
)


def _write_payload(path, include_sig_idx=True, include_clusters=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file_obj:
        if include_clusters:
            file_obj.create_dataset(
                "cluster_labels",
                data=np.array([[1], [0], [2], [1], [2], [0]],
                              dtype=np.uint32))
            file_obj.create_dataset(
                "cluster_sig", data=np.array([[1], [1]], dtype=np.uint8))
            file_obj.create_dataset("cluster_p", data=np.array([[0.01], [0.04]]))
            file_obj.create_dataset(
                "cluster_masses", data=np.array([[8.5], [4.25]]))
        file_obj.create_dataset("N_Z", data=np.array([[2.0]]))
        file_obj.create_dataset("N_ROIS", data=np.array([[3.0]]))
        if include_sig_idx:
            file_obj.create_dataset("sig_idx", data=np.array([[1.0], [5.0]]))


def _write_hex_rois(path):
    expected_masks = []
    ref_dtype = h5py.special_dtype(ref=h5py.Reference)
    with h5py.File(path, "w") as file_obj:
        refs = file_obj.create_group("#refs#")
        plane_datasets = []
        for z_index in range(2):
            plane = refs.create_dataset(
                f"plane_{z_index}", shape=(3, 1), dtype=ref_dtype)
            plane_datasets.append(plane)
            plane_masks = []
            for roi_index in range(3):
                mask = np.zeros((4, 5), dtype=bool)
                row = (z_index + roi_index) % 3
                col = roi_index + 1
                mask[row:row + 2, col:col + 2] = True
                plane_masks.append(mask)
                # MATLAB v7.3 stores the two spatial axes reversed in HDF5.
                mask_dataset = refs.create_dataset(
                    f"mask_{z_index}_{roi_index}", data=mask.T.astype(np.uint8))
                mask_dataset.attrs["MATLAB_class"] = np.bytes_("logical")
                plane[roi_index, 0] = mask_dataset.ref
            expected_masks.append(plane_masks)

        hex_rois = file_obj.create_dataset(
            "hex_rois", shape=(1, 2), dtype=ref_dtype)
        for z_index, plane in enumerate(plane_datasets):
            hex_rois[0, z_index] = plane.ref
    return expected_masks


def test_finds_ancestor_mask_file_and_loads_only_sig_idx(tmp_path):
    df_f = tmp_path / "dataset" / "dfF"
    payload_path = df_f / "paper_figures_pretty" / "cluster" / "PAYLOAD.mat"
    mask_path = df_f / "hex_rois.mat"
    _write_payload(payload_path)
    expected_masks = _write_hex_rois(mask_path)

    assert find_hex_rois_file(payload_path) == str(mask_path.resolve())
    result = load_significant_rois(
        payload_path, mask_path, expected_shape=(4, 5), transpose=True)

    assert result["n_z"] == 2
    assert result["n_rois"] == 3
    assert result["clusters"] == [1, 2]
    assert [roi["index"] for roi in result["rois"]] == [1, 5]
    assert [roi["z"] for roi in result["rois"]] == [0, 1]
    assert [roi["roi_index"] for roi in result["rois"]] == [1, 2]
    for roi, expected in zip(
            result["rois"], (expected_masks[0][0], expected_masks[1][1])):
        r0, r1, c0, c1 = roi["bbox"]
        np.testing.assert_array_equal(roi["mask"], expected[r0:r1, c0:c1])
    assert result["rois"][0]["cluster_p"] == pytest.approx(0.01)
    assert result["rois"][1]["cluster_mass"] == pytest.approx(4.25)


def test_derives_indices_from_cluster_sig_when_sig_idx_is_absent(tmp_path):
    payload_path = tmp_path / "payload.mat"
    mask_path = tmp_path / "hex_rois.mat"
    _write_payload(payload_path, include_sig_idx=False)
    _write_hex_rois(mask_path)

    result = load_significant_rois(
        payload_path, mask_path, expected_shape=(4, 5), transpose=True)

    assert [roi["index"] for roi in result["rois"]] == [1, 3, 4, 5]


def test_cluster_metadata_is_optional_when_sig_idx_is_present(tmp_path):
    payload_path = tmp_path / "payload.mat"
    mask_path = tmp_path / "hex_rois.mat"
    _write_payload(payload_path, include_clusters=False)
    _write_hex_rois(mask_path)

    result = load_significant_rois(
        payload_path, mask_path, expected_shape=(4, 5), transpose=True)

    assert [roi["index"] for roi in result["rois"]] == [1, 5]
    assert result["clusters"] == []
    assert all("cluster" not in roi for roi in result["rois"])


def test_requires_a_way_to_identify_significant_rois(tmp_path):
    payload_path = tmp_path / "payload.mat"
    mask_path = tmp_path / "hex_rois.mat"
    _write_payload(
        payload_path, include_sig_idx=False, include_clusters=False)
    _write_hex_rois(mask_path)

    with pytest.raises(ProcessedROIImportError, match="needs 'sig_idx'"):
        load_significant_rois(
            payload_path, mask_path, expected_shape=(4, 5), transpose=True)


def test_rejects_mask_that_does_not_match_the_displayed_stack(tmp_path):
    payload_path = tmp_path / "payload.mat"
    mask_path = tmp_path / "hex_rois.mat"
    _write_payload(payload_path)
    _write_hex_rois(mask_path)

    with pytest.raises(ProcessedROIImportError, match="does not match"):
        load_significant_rois(
            payload_path, mask_path, expected_shape=(5, 4), transpose=True)
