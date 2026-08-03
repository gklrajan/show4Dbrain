import time

import h5py
import numpy as np

from show4dbrain import LazyMatStack, Viewer
from pyqtgraph.Qt import QtCore, QtWidgets


def _write_stack(path):
    frames = []
    base = np.arange(4, dtype=np.float32)[:, None] * 10
    base = base + np.arange(5, dtype=np.float32)[None, :]
    for frame_index in range(6):
        display_frame = base + frame_index * 100
        frames.append(display_frame.T)  # MATLAB-v7.3/HDF5 spatial order
    with h5py.File(path, "w") as file_obj:
        file_obj.create_dataset("stack", data=np.stack(frames))


def _process_until(app, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert predicate(), "Qt background operation did not finish in time"


def test_imported_mask_stays_available_while_trace_toggles(tmp_path):
    stack_path = tmp_path / "stack.mat"
    _write_stack(stack_path)
    stack = LazyMatStack(stack_path, varname="stack", slices_per_volume=2)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = Viewer(stack, ram_frac=0.02)

    mask = np.ones((2, 2), dtype=bool)
    data = {
        "payload_path": str(tmp_path / "payload.mat"),
        "mask_path": str(tmp_path / "hex_rois.mat"),
        "n_z": 2,
        "n_rois": 1,
        "clusters": [3],
        "rois": [
            {
                "index": 1,
                "roi_index": 1,
                "z": 0,
                "cluster": 3,
                "mask": mask,
                "bbox": (1, 3, 2, 4),
            },
            {
                "index": 2,
                "roi_index": 1,
                "z": 1,
                "mask": mask.copy(),
                "bbox": (1, 3, 2, 4),
            },
        ],
    }

    try:
        viewer._install_imported_rois(data)
        imported, imported_z1 = viewer.imported_rois
        item, item_z1 = imported["roi"], imported_z1["roi"]
        assert item.isVisible()
        assert not item_z1.isVisible()
        assert item.shape().contains(QtCore.QPointF(2.5, 1.5))
        assert not imported["active"] and not imported_z1["active"]
        assert len(viewer.rois) == 0

        viewer.bin = 2
        viewer._refresh_imported_geometry()
        assert item.shape().contains(QtCore.QPointF(1.5, 0.5))
        viewer.bin = 1
        viewer._refresh_imported_geometry()

        viewer._toggle_imported_roi(imported)
        viewer._toggle_imported_roi(imported_z1)
        assert imported["active"] and imported_z1["active"]
        assert len(viewer.rois) == 2
        rid, rid_z1 = id(item), id(item_z1)
        _process_until(
            app, lambda: rid in viewer.raw_traces and rid_z1 in viewer.raw_traces)
        np.testing.assert_allclose(
            viewer.raw_traces[rid], np.array([17.5, 217.5, 417.5]))
        np.testing.assert_allclose(
            viewer.raw_traces[rid_z1], np.array([117.5, 317.5, 517.5]))

        viewer._set_z(1)
        assert not item.isVisible()
        assert item_z1.isVisible()
        np.testing.assert_allclose(
            viewer.raw_traces[rid], np.array([17.5, 217.5, 417.5]))

        viewer._toggle_imported_roi(imported)
        viewer._toggle_imported_roi(imported_z1)
        assert not imported["active"] and not imported_z1["active"]
        assert len(viewer.rois) == 0
        assert rid not in viewer.raw_traces
        assert rid_z1 not in viewer.raw_traces
        assert item in [entry["roi"] for entry in viewer.imported_rois]
    finally:
        viewer.close()
        app.processEvents()
