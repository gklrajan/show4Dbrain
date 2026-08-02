"""Load significant ROI masks produced by the WIDE-CAT analysis pipeline.

The cluster payload and ``hex_rois.mat`` are MATLAB v7.3 (HDF5) files.  The
payload identifies significant entries in the flattened ``[Z, ROI]`` layout;
``hex_rois.mat`` owns the actual two-dimensional logical masks.
"""

from pathlib import Path

import h5py
import numpy as np


class ProcessedROIImportError(ValueError):
    """Raised when an analysis payload cannot be mapped to image-space ROIs."""


def find_hex_rois_file(payload_path):
    """Return the most likely mask file for *payload_path*, or ``None``.

    The analysis script saves payloads below ``dfF/paper_figures_pretty/...``
    and saves ``hex_rois.mat`` directly in ``dfF``.  Walking the payload's
    parent directories therefore finds the related file without asking the
    user for it in the usual case.  A self-contained payload with a top-level
    ``hex_rois`` variable is also supported.
    """
    payload_path = Path(payload_path).expanduser().resolve()
    try:
        with h5py.File(str(payload_path), "r") as payload_file:
            if "hex_rois" in payload_file:
                return str(payload_path)
    except OSError:
        return None

    for directory in (payload_path.parent, *payload_path.parents):
        candidate = directory / "hex_rois.mat"
        if candidate.is_file():
            return str(candidate)
    return None


def _dataset(file_obj, name):
    obj = file_obj.get(name)
    if not isinstance(obj, h5py.Dataset):
        raise ProcessedROIImportError(
            f"Required variable '{name}' is missing from {file_obj.filename}.")
    return obj


def _vector(file_obj, name, required=True):
    obj = file_obj.get(name)
    if not isinstance(obj, h5py.Dataset):
        if required:
            raise ProcessedROIImportError(
                f"Required variable '{name}' is missing from {file_obj.filename}.")
        return None
    return np.asarray(obj).reshape(-1)


def _positive_integer_scalar(file_obj, name):
    values = _vector(file_obj, name)
    if values.size != 1 or not np.isfinite(values[0]):
        raise ProcessedROIImportError(f"'{name}' must be one finite scalar.")
    value = int(round(float(values[0])))
    if value <= 0 or not np.isclose(float(values[0]), value):
        raise ProcessedROIImportError(f"'{name}' must be a positive integer.")
    return value


def _one_based_indices(values, name):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ProcessedROIImportError(f"'{name}' contains non-finite indices.")
    rounded = np.rint(values).astype(np.int64)
    if not np.allclose(values, rounded) or np.any(rounded < 1):
        raise ProcessedROIImportError(
            f"'{name}' must contain positive, one-based integer indices.")
    return rounded


def _linear_references(dataset, description):
    if h5py.check_dtype(ref=dataset.dtype) is None:
        raise ProcessedROIImportError(
            f"{description} is not a MATLAB cell array of HDF5 references.")
    return np.asarray(dataset).reshape(-1)


def _dereference_dataset(file_obj, reference, description):
    if not reference:
        raise ProcessedROIImportError(f"{description} is an empty MATLAB cell.")
    obj = file_obj[reference]
    if not isinstance(obj, h5py.Dataset):
        raise ProcessedROIImportError(f"{description} does not contain an array.")
    return obj


def _read_hex_mask(mask_file, plane_refs, roi_refs_by_z,
                   z_index, roi_index, transpose):
    if z_index >= plane_refs.size:
        raise ProcessedROIImportError(
            f"hex_rois has {plane_refs.size} planes; requested plane {z_index + 1}.")
    if z_index not in roi_refs_by_z:
        plane = _dereference_dataset(
            mask_file, plane_refs[z_index], f"hex_rois plane {z_index + 1}")
        roi_refs_by_z[z_index] = _linear_references(
            plane, f"hex_rois plane {z_index + 1}")
    roi_refs = roi_refs_by_z[z_index]
    if roi_index >= roi_refs.size:
        raise ProcessedROIImportError(
            f"hex_rois plane {z_index + 1} has {roi_refs.size} ROIs; "
            f"requested ROI {roi_index + 1}.")
    mask_dataset = _dereference_dataset(
        mask_file, roi_refs[roi_index],
        f"hex_rois plane {z_index + 1}, ROI {roi_index + 1}")
    mask = np.squeeze(np.asarray(mask_dataset)).astype(bool, copy=False)
    if mask.ndim != 2:
        raise ProcessedROIImportError(
            f"ROI {roi_index + 1} on plane {z_index + 1} is not a 2D mask "
            f"(shape {mask.shape}).")
    # MATLAB v7.3 reverses array axes in HDF5.  This mirrors LazyMatStack's
    # display transform so imported masks and image pixels share coordinates.
    if transpose:
        mask = mask.T
    return np.ascontiguousarray(mask, dtype=bool)


def load_significant_rois(payload_path, mask_path, expected_shape, transpose=True):
    """Load and validate significant ROI masks.

    Returns a dictionary containing ``rois``.  Every ROI has a cropped Boolean
    ``mask`` in full-resolution pixels, its global ``bbox``, zero-based ``z``
    plane and one-based flattened ``index``. A one-based ``cluster`` identifier
    is included when ``cluster_labels`` is available. Only significant masks
    are read.
    """
    payload_path = str(Path(payload_path).expanduser().resolve())
    mask_path = str(Path(mask_path).expanduser().resolve())
    expected_shape = tuple(int(v) for v in expected_shape)

    try:
        payload_file = h5py.File(payload_path, "r")
    except OSError as exc:
        raise ProcessedROIImportError(
            f"Cannot open analysis payload as MATLAB v7.3/HDF5: {exc}") from exc

    try:
        labels = _vector(payload_file, "cluster_labels", required=False)
        n_z = _positive_integer_scalar(payload_file, "N_Z")
        n_rois = _positive_integer_scalar(payload_file, "N_ROIS")

        sig_values = _vector(payload_file, "sig_idx", required=False)
        if sig_values is None:
            if labels is None:
                raise ProcessedROIImportError(
                    "The payload needs 'sig_idx', or both 'cluster_labels' "
                    "and 'cluster_sig' so significant indices can be derived.")
            cluster_sig = _vector(payload_file, "cluster_sig")
            significant_clusters = np.flatnonzero(cluster_sig.astype(bool)) + 1
            sig_indices = np.flatnonzero(
                np.isin(labels.astype(np.int64), significant_clusters)) + 1
        else:
            sig_indices = _one_based_indices(sig_values, "sig_idx")

        # MATLAB's find() is sorted, but unique/sort also protects the GUI from
        # duplicate contours in hand-edited payload files.
        sig_indices = np.unique(sig_indices)
        if (labels is not None and sig_indices.size and
                sig_indices[-1] > labels.size):
            raise ProcessedROIImportError(
                f"sig_idx contains {sig_indices[-1]}, but cluster_labels has only "
                f"{labels.size} entries.")
        if sig_indices.size and sig_indices[-1] > n_z * n_rois:
            raise ProcessedROIImportError(
                "sig_idx is inconsistent with N_Z x N_ROIS "
                f"({n_z} x {n_rois}).")

        cluster_p = _vector(payload_file, "cluster_p", required=False)
        cluster_masses = _vector(payload_file, "cluster_masses", required=False)

        try:
            mask_file = h5py.File(mask_path, "r")
        except OSError as exc:
            raise ProcessedROIImportError(
                f"Cannot open ROI masks as MATLAB v7.3/HDF5: {exc}") from exc
        try:
            hex_rois = _dataset(mask_file, "hex_rois")
            plane_refs = _linear_references(hex_rois, "'hex_rois'")
            roi_refs_by_z = {}
            imported = []
            for flat_index in sig_indices:
                zero_index = int(flat_index) - 1
                z_index, roi_index = divmod(zero_index, n_rois)
                cluster = None
                if labels is not None:
                    label = int(round(float(labels[zero_index])))
                    if label > 0:
                        cluster = label
                mask = _read_hex_mask(
                    mask_file, plane_refs, roi_refs_by_z,
                    z_index, roi_index, transpose)
                if mask.shape != expected_shape:
                    raise ProcessedROIImportError(
                        f"ROI mask shape {mask.shape} does not match the displayed "
                        f"stack shape {expected_shape}. Check the selected stack and "
                        "the --no-transpose setting.")
                rows, cols = np.nonzero(mask)
                if rows.size == 0:
                    raise ProcessedROIImportError(
                        f"Significant ROI {flat_index} has an empty mask.")
                r0, r1 = int(rows.min()), int(rows.max()) + 1
                c0, c1 = int(cols.min()), int(cols.max()) + 1
                item = {
                    "index": int(flat_index),
                    "roi_index": roi_index + 1,
                    "z": z_index,
                    "mask": np.ascontiguousarray(mask[r0:r1, c0:c1]),
                    "bbox": (r0, r1, c0, c1),
                }
                if cluster is not None:
                    item["cluster"] = cluster
                if (cluster is not None and cluster_p is not None and
                        cluster <= cluster_p.size):
                    item["cluster_p"] = float(cluster_p[cluster - 1])
                if (cluster is not None and cluster_masses is not None and
                        cluster <= cluster_masses.size):
                    item["cluster_mass"] = float(cluster_masses[cluster - 1])
                imported.append(item)
        finally:
            mask_file.close()
    finally:
        payload_file.close()

    return {
        "payload_path": payload_path,
        "mask_path": mask_path,
        "n_z": n_z,
        "n_rois": n_rois,
        "rois": imported,
        "clusters": sorted({item["cluster"] for item in imported
                            if "cluster" in item}),
    }
