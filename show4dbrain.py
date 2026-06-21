"""
show4dbrain.py
==============
Memory-efficient GUI to browse a large volumetric image stack
(MATLAB v7.3 .mat / HDF5) and pull full-length ROI time-series.

Design (robust on most machines)
--------------------------------
* The file is opened lazily with h5py; the whole array is NEVER loaded.
* Stack is [Y x X x T] with T = n_volumes * slices_per_volume.
  Frame for slice z, volume v is:   frame = v * slices_per_volume + z
* IMAGE = a sliding WINDOW of volumes ("tiles"). Only a few tiles for the
  current Z are kept resident; the window size is sized automatically so the
  resident image memory stays near a fraction of free RAM (default 25%).
  Neighbouring tiles are prefetched in the background for smooth scrubbing.
* TRACES = FULL length. An ROI is a small bounding box, so its time-series is
  read directly from disk one sub-rectangle per volume (kilobytes per ROI),
  never whole frames. The trace spans the entire recording and the time-slider
  dot scrubs across all of it.
* dF/F per ROI = rolling-baseline (F - F0)/F0, F0 = moving mean/median/pct.
* Optional grey stimulus shading (dur / inter-stim / offset, in volumes).

Memory, in short
----------------
Resident = a few image tiles (~25% of free RAM, auto-sized) + tiny per-ROI
trace vectors. Independent of recording length and of full image resolution.

Install
-------
    pip install h5py numpy scipy pyqtgraph PyQt5
    # optional, for RAM-aware window sizing:  pip install psutil

Run
---
    python show4dbrain.py --file your_stack.mat
    # variable auto-detected (largest 3D dataset); override with --var
    # slices per volume default 40; override with --spv
    # image window uses ~25% of free RAM; override with --ram-frac
    # if the image looks rotated 90 deg, add --no-transpose

Notes
-----
* MATLAB -v7.3 arrays are HDF5 with reversed axis order, so a MATLAB Y x X x T
  array appears to h5py as (T, X, Y). The time axis is auto-detected as the
  largest dimension. Frames are transposed by default to match MATLAB's
  on-screen orientation; ROI extraction is consistent either way.
"""

import sys
import argparse
import threading
import collections

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("Missing dependency: pip install h5py")

try:
    from scipy.ndimage import uniform_filter1d, median_filter
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
except ImportError:
    sys.exit("Missing dependency: pip install pyqtgraph PyQt5")


pg.setConfigOptions(imageAxisOrder="row-major", background="k", foreground="w")

ROI_COLORS = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
    (255, 127, 0), (166, 86, 40), (247, 129, 191), (153, 153, 153),
    (0, 206, 209), (220, 220, 0),
]

TILES_RESIDENT = 3      # max image tiles kept in RAM (current +/- prefetch)
K_MAX = 1000            # cap on volumes per tile (responsiveness)


def available_ram_bytes():
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 8 * 1024 ** 3


# ----------------------------------------------------------------------------
# Lazy data access
# ----------------------------------------------------------------------------
class LazyMatStack:
    """Lazy read-only access to a 3D array inside an HDF5 / MATLAB-v7.3 file."""

    def __init__(self, path, varname=None, slices_per_volume=40, transpose=True):
        self.path = path
        self.transpose = transpose
        self._io_lock = threading.Lock()   # h5py reads serialized across threads
        self.file = h5py.File(path, "r")
        self.dset = self._find_dataset(varname)
        if self.dset.ndim != 3:
            raise ValueError(
                f"Expected a 3D dataset, got shape {self.dset.shape}. "
                "Pass --var to select the right variable.")
        self.shape = tuple(self.dset.shape)
        self.time_axis = int(np.argmax(self.shape))
        self.n_frames = int(self.shape[self.time_axis])
        self.spatial_axes = [i for i in range(3) if i != self.time_axis]  # ascending

        sh0 = self.shape[self.spatial_axes[0]]
        sh1 = self.shape[self.spatial_axes[1]]
        if self.transpose:
            self.H, self.W = sh1, sh0
        else:
            self.H, self.W = sh0, sh1

        self.set_spv(slices_per_volume)

    # -- discovery ------------------------------------------------------------
    def _find_dataset(self, varname):
        cands = {}

        def visit(name, obj):
            if name.split("/")[0] in ("#refs#", "#subsystem#"):
                return
            if isinstance(obj, h5py.Dataset):
                cands[name] = obj

        self.file.visititems(visit)
        if varname:
            if varname in self.file and isinstance(self.file[varname], h5py.Dataset):
                return self.file[varname]
            if varname in cands:
                return cands[varname]
            raise ValueError(f"Variable '{varname}' not found. "
                             f"Available: {sorted(cands.keys())}")
        best = None
        for o in cands.values():
            if o.ndim == 3 and (best is None or np.prod(o.shape) > np.prod(best.shape)):
                best = o
        if best is None:
            raise ValueError(f"No 3D dataset found. Datasets: {sorted(cands.keys())}")
        return best

    # -- indexing -------------------------------------------------------------
    def set_spv(self, spv):
        self.spv = max(1, int(spv))
        self.n_volumes = self.n_frames // self.spv

    def frame_index(self, z, v):
        return v * self.spv + z

    @property
    def vol_nbytes(self):
        return self.H * self.W * 4

    # -- full-frame read (for the image window) -------------------------------
    def read_frame(self, z, v):
        t = self.frame_index(z, v)
        idx = [slice(None)] * 3
        idx[self.time_axis] = t
        with self._io_lock:
            arr = np.asarray(self.dset[tuple(idx)], dtype=np.float32)
        if self.transpose:
            arr = arr.T
        return np.ascontiguousarray(arr, dtype=np.float32)

    def read_window(self, z, v0, v1, progress=None, should_stop=None):
        n = v1 - v0
        out = np.empty((n, self.H, self.W), dtype=np.float32)
        for i, v in enumerate(range(v0, v1)):
            if should_stop is not None and should_stop():
                return None
            out[i] = self.read_frame(z, v)
            if progress is not None and (i % 4 == 0 or i == n - 1):
                progress(int((i + 1) / n * 100))
        return out

    # -- sub-rectangle read (for full-length ROI traces) ----------------------
    def _read_region(self, t, r0, r1, c0, c1):
        """Read a display-oriented sub-rectangle [r0:r1, c0:c1] of frame t."""
        idx = [slice(None)] * 3
        idx[self.time_axis] = t
        if self.transpose:
            # display H <- spatial_axes[1], W <- spatial_axes[0]
            idx[self.spatial_axes[1]] = slice(r0, r1)
            idx[self.spatial_axes[0]] = slice(c0, c1)
            with self._io_lock:
                a = np.asarray(self.dset[tuple(idx)], dtype=np.float32)
            return a.T  # (c-range, r-range) -> (r-range, c-range)
        else:
            idx[self.spatial_axes[0]] = slice(r0, r1)
            idx[self.spatial_axes[1]] = slice(c0, c1)
            with self._io_lock:
                a = np.asarray(self.dset[tuple(idx)], dtype=np.float32)
            return a

    def read_bbox_timeseries(self, z, r0, r1, c0, c1, progress=None, should_stop=None):
        """(n_volumes, r1-r0, c1-c0) array for one ROI bbox across all volumes."""
        n = self.n_volumes
        out = np.empty((n, r1 - r0, c1 - c0), dtype=np.float32)
        for v in range(n):
            if should_stop is not None and should_stop():
                return None
            out[v] = self._read_region(self.frame_index(z, v), r0, r1, c0, c1)
            if progress is not None and (v % 8 == 0 or v == n - 1):
                progress(int((v + 1) / n * 100))
        return out

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Background workers
# ----------------------------------------------------------------------------
class TileLoader(QtCore.QThread):
    finished_tile = QtCore.pyqtSignal(int, int, object)  # z, tile_index, arr
    progress = QtCore.pyqtSignal(int)
    error = QtCore.pyqtSignal(str)

    def __init__(self, stack, z, ti, v0, v1):
        super().__init__()
        self.stack, self.z, self.ti, self.v0, self.v1 = stack, z, ti, v0, v1
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            arr = self.stack.read_window(self.z, self.v0, self.v1,
                                         progress=self.progress.emit,
                                         should_stop=lambda: self._stop)
        except MemoryError:
            self.error.emit("Out of memory loading image window; lower --ram-frac.")
            return
        except Exception as e:
            self.error.emit(f"Image load failed: {e}")
            return
        if arr is not None and not self._stop:
            self.finished_tile.emit(self.z, self.ti, arr)


class TraceLoader(QtCore.QThread):
    finished_traces = QtCore.pyqtSignal(int, object)  # z, {rid: raw_trace}
    progress = QtCore.pyqtSignal(int)
    error = QtCore.pyqtSignal(str)

    def __init__(self, stack, z, specs):
        super().__init__()
        self.stack, self.z, self.specs = stack, z, specs
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        out = {}
        m = max(1, len(self.specs))
        try:
            for j, s in enumerate(self.specs):
                if self._stop:
                    return
                box = self.stack.read_bbox_timeseries(
                    self.z, s["r0"], s["r1"], s["c0"], s["c1"],
                    should_stop=lambda: self._stop)
                if box is None:
                    return
                mask = s["mask"]
                if mask.any():
                    tr = box.reshape(box.shape[0], -1)[:, mask.ravel()].mean(axis=1)
                else:
                    tr = np.full(box.shape[0], np.nan)
                out[s["rid"]] = tr.astype(np.float64)
                self.progress.emit(int((j + 1) / m * 100))
        except Exception as e:
            self.error.emit(f"Trace computation failed: {e}")
            return
        if not self._stop:
            self.finished_traces.emit(self.z, out)


# ----------------------------------------------------------------------------
# dF/F
# ----------------------------------------------------------------------------
def _sliding_percentile(F, window, p):
    pad = window // 2
    Fp = np.pad(F, (pad, window - 1 - pad), mode="edge")
    sw = np.lib.stride_tricks.sliding_window_view(Fp, window)
    return np.percentile(sw, p, axis=1)


def compute_dff(trace, window, method="mean"):
    F = np.asarray(trace, dtype=np.float64)
    n = F.size
    if n == 0:
        return F
    window = int(np.clip(window, 1, n))
    if method == "mean":
        F0 = uniform_filter1d(F, size=window, mode="nearest") if _HAVE_SCIPY \
            else _sliding_percentile(F, window, 50)
    elif method == "median":
        F0 = median_filter(F, size=window, mode="nearest") if _HAVE_SCIPY \
            else _sliding_percentile(F, window, 50)
    elif method.startswith("pct"):
        F0 = _sliding_percentile(F, window, float(method.split("_")[1]))
    else:
        F0 = uniform_filter1d(F, size=window, mode="nearest")
    F0 = np.where(np.abs(F0) < 1e-6, 1e-6, F0)
    return (F - F0) / F0


# ----------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------
class Viewer(QtWidgets.QMainWindow):
    def __init__(self, stack: LazyMatStack, ram_frac=0.25):
        super().__init__()
        self.stack = stack
        self.ram_frac = ram_frac
        self.current_z = 0
        self.current_v = 0
        self.levels = None
        self.bin = 1                 # lateral display/ROI binning factor (1 = off)

        # image tiles
        self.tiles = collections.OrderedDict()   # tile_index -> (k,H,W) array
        self._tile_loader = None
        self._tile_queue = []

        # rois / traces
        self.rois = []
        self.selected_roi = None
        self.raw_traces = {}
        self.dff_traces = {}
        self.raw_curves = {}
        self.dff_curves = {}
        self._dirty_rids = set()
        self._trace_loader = None
        self._roi_counter = 0

        self.setWindowTitle(f"show4Dbrain  —  {stack.path}")
        self.resize(1500, 900)
        self._build_ui()
        self._recompute_window_size()
        self._set_z(0)

    # ---- window sizing ------------------------------------------------------
    def _recompute_window_size(self):
        budget = self.ram_frac * available_ram_bytes()
        k = int(budget / (TILES_RESIDENT * max(1, self.stack.vol_nbytes)))
        self.k = max(1, min(K_MAX, self.stack.n_volumes, k))

    def _n_tiles(self):
        return (self.stack.n_volumes + self.k - 1) // self.k

    # ---- UI -----------------------------------------------------------------
    def _build_ui(self):
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(splitter)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)

        glw = pg.GraphicsLayoutWidget()
        self.vb = glw.addViewBox(lockAspect=True, invertY=True)
        self.img_item = pg.ImageItem()
        self.vb.addItem(self.img_item)
        self.hist = pg.HistogramLUTItem(image=self.img_item)
        glw.addItem(self.hist)
        lv.addWidget(glw, stretch=1)

        # Z + slices/vol
        zrow = QtWidgets.QHBoxLayout()
        zrow.addWidget(QtWidgets.QLabel("Z slice"))
        self.z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.z_slider.setRange(0, self.stack.spv - 1)
        self.z_spin = QtWidgets.QSpinBox()
        self.z_spin.setRange(0, self.stack.spv - 1)
        zrow.addWidget(self.z_slider, stretch=1)
        zrow.addWidget(self.z_spin)
        zrow.addSpacing(12)
        zrow.addWidget(QtWidgets.QLabel("slices/vol"))
        self.spv_spin = QtWidgets.QSpinBox()
        self.spv_spin.setRange(1, max(1, self.stack.n_frames))
        self.spv_spin.setValue(self.stack.spv)
        self.spv_spin.setToolTip("Optical slices per volume.\n"
                                 "Changing this re-slices the stack and reloads.")
        zrow.addWidget(self.spv_spin)
        lv.addLayout(zrow)

        # lateral display binning (NxN average; affects what you see AND where
        # ROIs sit / what their traces average -- memory is unaffected)
        brow = QtWidgets.QHBoxLayout()
        self.bin_on = QtWidgets.QCheckBox("bin display")
        self.bin_on.setToolTip("Show the image laterally NxN-averaged so sparse, "
                               "low-contrast signal pools up and becomes visible.\n"
                               "ROIs are placed on, and traces computed from, the "
                               "binned image. Changing the factor clears ROIs.")
        brow.addWidget(self.bin_on)
        brow.addWidget(QtWidgets.QLabel("factor"))
        self.bin_spin = QtWidgets.QSpinBox()
        self.bin_spin.setRange(2, 200)
        self.bin_spin.setValue(14)   # ~18 um hexagon at ~1.3 um/px; editable
        self.bin_spin.setEnabled(False)
        brow.addWidget(self.bin_spin)
        brow.addStretch(1)
        lv.addLayout(brow)

        # time
        trow = QtWidgets.QHBoxLayout()
        trow.addWidget(QtWidgets.QLabel("Volume (time)"))
        self.t_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.t_slider.setRange(0, self.stack.n_volumes - 1)
        self.t_spin = QtWidgets.QSpinBox()
        self.t_spin.setRange(0, self.stack.n_volumes - 1)
        trow.addWidget(self.t_slider, stretch=1)
        trow.addWidget(self.t_spin)
        lv.addLayout(trow)

        # roi tools
        roirow = QtWidgets.QHBoxLayout()
        self.btn_add_rect = QtWidgets.QPushButton("+ Rect ROI")
        self.btn_add_ell = QtWidgets.QPushButton("+ Ellipse ROI")
        self.btn_del_sel = QtWidgets.QPushButton("Delete selected")
        self.btn_del_sel.setToolTip("Click an ROI to select it, then Del / this button. "
                                    "Right-click an ROI also offers Remove.")
        self.btn_clear = QtWidgets.QPushButton("Clear ROIs")
        for b in (self.btn_add_rect, self.btn_add_ell, self.btn_del_sel, self.btn_clear):
            roirow.addWidget(b)
        lv.addLayout(roirow)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximum(100)
        lv.addWidget(self.progress)
        self.status = QtWidgets.QLabel("")
        lv.addWidget(self.status)
        splitter.addWidget(left)

        # right: baseline + plots
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(4, 4, 4, 4)

        ctl = QtWidgets.QGridLayout()
        ctl.addWidget(QtWidgets.QLabel("dF/F baseline"), 0, 0)
        self.base_method = QtWidgets.QComboBox()
        self.base_method.addItems(["mean", "median", "pct_10", "pct_20"])
        ctl.addWidget(self.base_method, 0, 1)
        ctl.addWidget(QtWidgets.QLabel("window (vols)"), 0, 2)
        self.base_win = QtWidgets.QSpinBox()
        self.base_win.setRange(1, max(1, self.stack.n_volumes))
        self.base_win.setValue(min(50, self.stack.n_volumes))
        ctl.addWidget(self.base_win, 0, 3)

        self.stim_on = QtWidgets.QCheckBox("Stim shading")
        ctl.addWidget(self.stim_on, 1, 0)
        ctl.addWidget(QtWidgets.QLabel("dur"), 1, 1)
        self.stim_dur = QtWidgets.QSpinBox()
        self.stim_dur.setRange(1, 100000)
        self.stim_dur.setValue(6)
        ctl.addWidget(self.stim_dur, 1, 2)
        ctl.addWidget(QtWidgets.QLabel("inter-stim"), 1, 3)
        self.stim_gap = QtWidgets.QSpinBox()
        self.stim_gap.setRange(0, 100000)
        self.stim_gap.setValue(17)
        self.stim_gap.setToolTip("OFF gap between stims (volumes). Cycle = dur + inter-stim.")
        ctl.addWidget(self.stim_gap, 1, 4)
        ctl.addWidget(QtWidgets.QLabel("offset"), 1, 5)
        self.stim_off = QtWidgets.QSpinBox()
        self.stim_off.setRange(0, 100000)
        self.stim_off.setValue(0)
        ctl.addWidget(self.stim_off, 1, 6)
        rv.addLayout(ctl)

        self.raw_plot = pg.PlotWidget(title="Raw ROI mean intensity (full session)")
        self.raw_plot.setLabel("bottom", "Volume")
        self.raw_plot.setLabel("left", "F (a.u.)")
        self.raw_plot.addLegend(offset=(10, 10))
        self.dff_plot = pg.PlotWidget(title="Rolling-baseline dF/F (full session)")
        self.dff_plot.setLabel("bottom", "Volume")
        self.dff_plot.setLabel("left", "dF/F")
        self.dff_plot.setXLink(self.raw_plot)
        rv.addWidget(self.raw_plot, stretch=1)
        rv.addWidget(self.dff_plot, stretch=1)

        self.raw_marker = pg.ScatterPlotItem(size=10, pen=pg.mkPen("w"))
        self.dff_marker = pg.ScatterPlotItem(size=10, pen=pg.mkPen("w"))
        self.raw_plot.addItem(self.raw_marker)
        self.dff_plot.addItem(self.dff_marker)
        self.time_line = pg.InfiniteLine(angle=90, movable=True,
                                         pen=pg.mkPen("w", width=1, style=QtCore.Qt.DashLine))
        self.dff_line = pg.InfiniteLine(angle=90, movable=False,
                                        pen=pg.mkPen("w", width=1, style=QtCore.Qt.DashLine))
        self.raw_plot.addItem(self.time_line)
        self.dff_plot.addItem(self.dff_line)
        splitter.addWidget(right)
        splitter.setSizes([720, 780])

        # signals
        self.z_slider.valueChanged.connect(self._on_z_changed)
        self.z_spin.valueChanged.connect(self.z_slider.setValue)
        self.t_slider.valueChanged.connect(self._on_t_changed)
        self.t_spin.valueChanged.connect(self.t_slider.setValue)
        self.spv_spin.editingFinished.connect(self._on_spv_changed)
        self.btn_add_rect.clicked.connect(lambda: self._add_roi("rect"))
        self.btn_add_ell.clicked.connect(lambda: self._add_roi("ellipse"))
        self.btn_del_sel.clicked.connect(self._delete_selected)
        self.btn_clear.clicked.connect(self._clear_rois)
        for key in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            QtWidgets.QShortcut(QtGui.QKeySequence(key), self).activated.connect(
                self._delete_selected)
        self.base_method.currentIndexChanged.connect(self._recompute_dff_only)
        self.base_win.valueChanged.connect(self._recompute_dff_only)
        for w in (self.stim_on, self.stim_dur, self.stim_gap, self.stim_off):
            (w.stateChanged if isinstance(w, QtWidgets.QCheckBox)
             else w.valueChanged).connect(self._draw_stim)
        self.time_line.sigPositionChanged.connect(self._on_timeline_drag)
        self.bin_on.stateChanged.connect(self._on_bin_changed)
        self.bin_spin.editingFinished.connect(self._on_bin_changed)

    # ---- Z / spv ------------------------------------------------------------
    def _on_z_changed(self, z):
        self.z_spin.blockSignals(True)
        self.z_spin.setValue(z)
        self.z_spin.blockSignals(False)
        self._set_z(z)

    def _set_z(self, z):
        self.current_z = z
        # tiles hold frames for a specific slice -> invalid on Z change
        if self._tile_loader is not None and self._tile_loader.isRunning():
            self._tile_loader.stop()
            self._tile_loader.wait(2000)
        self.tiles.clear()
        self._tile_queue = []
        self._need_volume(self.current_v)
        # data changed -> all traces stale
        self._dirty_rids = set(id(e["roi"]) for e in self.rois)
        self._pump_traces()

    def _on_spv_changed(self):
        new_spv = self.spv_spin.value()
        if new_spv == self.stack.spv:
            return
        self.stack.set_spv(new_spv)
        self._recompute_window_size()
        zmax, tmax = self.stack.spv - 1, self.stack.n_volumes - 1
        for w in (self.z_slider, self.z_spin):
            w.blockSignals(True); w.setMaximum(zmax); w.blockSignals(False)
        for w in (self.t_slider, self.t_spin):
            w.blockSignals(True); w.setMaximum(tmax); w.blockSignals(False)
        self.base_win.setMaximum(max(1, self.stack.n_volumes))
        self.current_z = min(self.current_z, zmax)
        self.current_v = min(self.current_v, tmax)
        self.z_slider.blockSignals(True); self.z_slider.setValue(self.current_z); self.z_slider.blockSignals(False)
        self.t_slider.blockSignals(True); self.t_slider.setValue(self.current_v); self.t_slider.blockSignals(False)
        self._set_z(self.current_z)
        self._draw_stim()

    # ---- image tiles --------------------------------------------------------
    def _tile_of(self, v):
        return v // self.k

    def _need_volume(self, v):
        ti = self._tile_of(v)
        offset = v - ti * self.k
        nbrs = []
        if offset > 0.6 * self.k and ti + 1 < self._n_tiles():
            nbrs.append(ti + 1)
        if offset < 0.4 * self.k and ti - 1 >= 0:
            nbrs.append(ti - 1)
        self._tile_queue = [t for t in nbrs if t not in self.tiles]
        if ti in self.tiles:
            self._display_from_tile(v)
            self._pump_tiles()
        else:
            self.status.setText(f"Loading volumes {ti*self.k}-{min((ti+1)*self.k, self.stack.n_volumes)-1} ...")
            if self._tile_loader is not None and self._tile_loader.isRunning():
                self._tile_loader.stop()
                self._tile_loader.wait(2000)
            self._tile_queue.insert(0, ti)
            self._pump_tiles()

    def _pump_tiles(self):
        if self._tile_loader is not None and self._tile_loader.isRunning():
            return
        while self._tile_queue and self._tile_queue[0] in self.tiles:
            self._tile_queue.pop(0)
        if not self._tile_queue:
            return
        ti = self._tile_queue.pop(0)
        v0, v1 = ti * self.k, min((ti + 1) * self.k, self.stack.n_volumes)
        self._tile_loader = TileLoader(self.stack, self.current_z, ti, v0, v1)
        self._tile_loader.progress.connect(self.progress.setValue)
        self._tile_loader.finished_tile.connect(self._on_tile_loaded)
        self._tile_loader.error.connect(self._on_load_error)
        self._tile_loader.start()

    def _on_tile_loaded(self, z, ti, arr):
        if z != self.current_z:
            return
        self.tiles[ti] = arr
        self.tiles.move_to_end(ti)
        cur_ti = self._tile_of(self.current_v)
        while len(self.tiles) > TILES_RESIDENT:
            for key in list(self.tiles.keys()):
                if key != cur_ti:
                    del self.tiles[key]
                    break
            else:
                break
        if self.levels is None:
            pass  # levels are fit lazily in _display_from_tile (handles binning)
        if ti == cur_ti:
            self.progress.setValue(100)
            self._display_from_tile(self.current_v)
        self._pump_tiles()  # prefetch neighbours

    def _on_load_error(self, msg):
        self.status.setText(msg)

    def _display_from_tile(self, v):
        ti = self._tile_of(v)
        if ti not in self.tiles:
            return
        self.tiles.move_to_end(ti)
        disp = self._bin_image(self.tiles[ti][v - ti * self.k])
        if self.levels is None:
            lo, hi = np.percentile(disp, [1.0, 99.5])
            if hi <= lo:
                hi = lo + 1.0
            self.levels = (float(lo), float(hi))
            self.hist.setLevels(*self.levels)
        self.img_item.setImage(disp, autoLevels=False, levels=self.levels)
        binnote = "" if self.bin == 1 else f" | bin {self.bin}x -> {self.Hb}x{self.Wb}"
        self.status.setText(
            f"z = {self.current_z} | vol {v}/{self.stack.n_volumes-1} | "
            f"window {self.k} vols x {self.stack.H}x{self.stack.W} "
            f"(~{self.k*self.stack.vol_nbytes/1e6:.0f} MB/tile, "
            f"{len(self.tiles)} resident){binnote}")

    # ---- time ---------------------------------------------------------------
    def _on_t_changed(self, v):
        self.t_spin.blockSignals(True)
        self.t_spin.setValue(v)
        self.t_spin.blockSignals(False)
        self.current_v = v
        self._need_volume(v)
        self._update_markers(v)
        self.time_line.blockSignals(True)
        self.time_line.setValue(v)
        self.time_line.blockSignals(False)
        self.dff_line.setValue(v)

    def _on_timeline_drag(self):
        v = int(np.clip(round(self.time_line.value()), 0, self.stack.n_volumes - 1))
        self.t_slider.setValue(v)

    # ---- display binning ----------------------------------------------------
    @property
    def Hb(self):
        return self.stack.H // self.bin

    @property
    def Wb(self):
        return self.stack.W // self.bin

    def _bin_image(self, frame):
        b = self.bin
        if b <= 1:
            return frame
        H, W = frame.shape
        hc, wc = (H // b) * b, (W // b) * b
        return frame[:hc, :wc].reshape(H // b, b, W // b, b).mean(axis=(1, 3))

    def _on_bin_changed(self):
        self.bin_spin.setEnabled(self.bin_on.isChecked())
        new_bin = self.bin_spin.value() if self.bin_on.isChecked() else 1
        if new_bin == self.bin:
            return
        self.bin = new_bin
        self._clear_rois()        # ROI coords live in binned pixels -> invalid on change
        self.levels = None        # refit contrast for the binned image
        self._display_from_tile(self.current_v)
        self.vb.autoRange()       # image extent changed; refit view

    # ---- ROIs ---------------------------------------------------------------
    def _probe_frame(self):
        # zero array only used for shape in getArraySlice (values unused)
        return np.empty((self.Hb, self.Wb), dtype=np.float32)

    def _add_roi(self, kind):
        self._roi_counter += 1
        color = ROI_COLORS[(self._roi_counter - 1) % len(ROI_COLORS)]
        pen = pg.mkPen(color, width=2)
        cx, cy = self.Wb // 2, self.Hb // 2
        size = max(2, min(80, min(self.Hb, self.Wb) // 3))
        if kind == "rect":
            roi = pg.RectROI([cx - size // 2, cy - size // 2], [size, size],
                             pen=pen, removable=True)
        else:
            roi = pg.EllipseROI([cx - size // 2, cy - size // 2], [size, size],
                                pen=pen, removable=True)
        try:
            roi.rotateAllowed = False
        except Exception:
            pass
        try:
            roi.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        except Exception:
            pass
        self.vb.addItem(roi)
        self.rois.append({"roi": roi, "kind": kind, "color": color})
        name = f"ROI {self._roi_counter}"
        self.raw_curves[id(roi)] = self.raw_plot.plot(pen=pg.mkPen(color, width=2), name=name)
        self.dff_curves[id(roi)] = self.dff_plot.plot(pen=pg.mkPen(color, width=2))
        roi.sigRegionChangeFinished.connect(lambda r=roi: self._mark_dirty(id(r)))
        roi.sigRemoveRequested.connect(lambda r=roi: self._remove_roi(r))
        roi.sigClicked.connect(lambda r=roi, *a: self._select_roi(r))
        self._select_roi(roi)
        self._mark_dirty(id(roi))

    def _select_roi(self, roi):
        self.selected_roi = roi
        for e in self.rois:
            r = e["roi"]
            r.setPen(pg.mkPen(e["color"], width=3 if r is roi else 2))
            try:
                r.setSelected(r is roi)
            except Exception:
                pass

    def _delete_selected(self):
        if self.selected_roi is not None and any(e["roi"] is self.selected_roi for e in self.rois):
            self._remove_roi(self.selected_roi)
        elif self.rois:
            self._remove_roi(self.rois[-1]["roi"])

    def _remove_roi(self, roi):
        rid = id(roi)
        self.vb.removeItem(roi)
        self.rois = [e for e in self.rois if e["roi"] is not roi]
        if self.selected_roi is roi:
            self.selected_roi = None
        rc = self.raw_curves.pop(rid, None)
        if rc is not None:
            self.raw_plot.removeItem(rc)
        dc = self.dff_curves.pop(rid, None)
        if dc is not None:
            self.dff_plot.removeItem(dc)
        self.raw_traces.pop(rid, None)
        self.dff_traces.pop(rid, None)
        self._dirty_rids.discard(rid)
        self._refresh_curves()
        self._update_markers(self.current_v)

    def _clear_rois(self):
        for e in list(self.rois):
            self._remove_roi(e["roi"])
        self.selected_roi = None

    def _roi_spec(self, roi):
        # ROI is drawn on the binned display -> slices are in binned pixels
        sl, _ = roi.getArraySlice(self._probe_frame(), self.img_item,
                                  axes=(0, 1), returnSlice=True)
        rb0 = max(0, int(sl[0].start)); rb1 = min(self.Hb, int(sl[0].stop))
        cb0 = max(0, int(sl[1].start)); cb1 = min(self.Wb, int(sl[1].stop))
        hb, wb = rb1 - rb0, cb1 - cb0
        if hb <= 0 or wb <= 0:
            return None
        # binned mask
        if isinstance(roi, pg.EllipseROI):
            yy, xx = np.mgrid[0:hb, 0:wb]
            cy, cx = (hb - 1) / 2.0, (wb - 1) / 2.0
            ry, rx = max(hb / 2.0, 0.5), max(wb / 2.0, 0.5)
            mask_b = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        else:
            mask_b = np.ones((hb, wb), dtype=bool)
        b = self.bin
        # map binned bbox -> full-resolution pixels and upsample the mask, so the
        # trace read averages exactly the full-res pixels under the binned ROI
        # (mean over binned super-pixels == mean over those full-res pixels).
        r0, r1, c0, c1 = rb0 * b, rb1 * b, cb0 * b, cb1 * b
        if b > 1:
            mask = np.repeat(np.repeat(mask_b, b, axis=0), b, axis=1)
        else:
            mask = mask_b
        return {"rid": id(roi), "r0": r0, "r1": r1, "c0": c0, "c1": c1, "mask": mask}

    # ---- trace pipeline -----------------------------------------------------
    def _mark_dirty(self, rid):
        self._dirty_rids.add(rid)
        self._pump_traces()

    def _pump_traces(self):
        if self._trace_loader is not None and self._trace_loader.isRunning():
            return  # will re-pump on finish
        specs = []
        for e in self.rois:
            rid = id(e["roi"])
            if rid in self._dirty_rids:
                s = self._roi_spec(e["roi"])
                if s is not None:
                    specs.append(s)
        if not specs:
            self._dirty_rids = set(r for r in self._dirty_rids
                                   if any(id(e["roi"]) == r for e in self.rois))
            return
        self._trace_loader = TraceLoader(self.stack, self.current_z, specs)
        self._trace_loader.progress.connect(
            lambda p: self.status.setText(f"Computing traces ... {p}%"))
        self._trace_loader.finished_traces.connect(self._on_traces_ready)
        self._trace_loader.error.connect(self._on_load_error)
        self._trace_loader.start()

    def _on_traces_ready(self, z, raw_map):
        if z != self.current_z:
            return
        win, method = self.base_win.value(), self.base_method.currentText()
        for rid, raw in raw_map.items():
            if not any(id(e["roi"]) == rid for e in self.rois):
                continue
            self.raw_traces[rid] = raw
            self.dff_traces[rid] = compute_dff(raw, win, method)
            self._dirty_rids.discard(rid)
        self._refresh_curves()
        self._update_markers(self.current_v)
        self._display_from_tile(self.current_v)  # restore status line
        if self._dirty_rids:
            self._pump_traces()

    def _recompute_dff_only(self):
        win, method = self.base_win.value(), self.base_method.currentText()
        for e in self.rois:
            rid = id(e["roi"])
            if rid in self.raw_traces:
                self.dff_traces[rid] = compute_dff(self.raw_traces[rid], win, method)
        self._refresh_curves()
        self._update_markers(self.current_v)

    def _refresh_curves(self):
        x = np.arange(self.stack.n_volumes)
        for e in self.rois:
            rid = id(e["roi"])
            if rid in self.raw_traces and rid in self.raw_curves:
                self.raw_curves[rid].setData(x, self.raw_traces[rid])
            if rid in self.dff_traces and rid in self.dff_curves:
                self.dff_curves[rid].setData(x, self.dff_traces[rid])

    def _update_markers(self, v):
        raw_pts, dff_pts = [], []
        for e in self.rois:
            rid = id(e["roi"])
            br = pg.mkBrush(e["color"])
            if rid in self.raw_traces and v < len(self.raw_traces[rid]):
                y = self.raw_traces[rid][v]
                if np.isfinite(y):
                    raw_pts.append({"pos": (v, y), "brush": br})
            if rid in self.dff_traces and v < len(self.dff_traces[rid]):
                y = self.dff_traces[rid][v]
                if np.isfinite(y):
                    dff_pts.append({"pos": (v, y), "brush": br})
        self.raw_marker.setData(raw_pts)
        self.dff_marker.setData(dff_pts)

    # ---- stim shading -------------------------------------------------------
    def _draw_stim(self):
        if not hasattr(self, "_stim_items"):
            self._stim_items = []
        for plot, it in self._stim_items:
            plot.removeItem(it)
        self._stim_items = []
        if not self.stim_on.isChecked():
            return
        period = self.stim_dur.value() + self.stim_gap.value()
        dur, off, n = self.stim_dur.value(), self.stim_off.value(), self.stack.n_volumes
        start = off
        while start < n:
            for plot in (self.raw_plot, self.dff_plot):
                reg = pg.LinearRegionItem(values=(start, min(start + dur, n)),
                                          movable=False,
                                          brush=pg.mkBrush(200, 200, 200, 60),
                                          pen=pg.mkPen(None))
                reg.setZValue(-10)
                plot.addItem(reg)
                self._stim_items.append((plot, reg))
            start += period

    def closeEvent(self, ev):
        for ld in (self._tile_loader, self._trace_loader):
            if ld is not None and ld.isRunning():
                ld.stop()
                ld.wait(2000)
        self.stack.close()
        super().closeEvent(ev)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        prog="show4dbrain",
        description="show4Dbrain — lazy, memory-safe viewer for big 4D .mat/HDF5 stacks")
    ap.add_argument("--file", help="path to the .mat (HDF5/v7.3) file")
    ap.add_argument("--var", default=None, help="variable name (default: auto-detect)")
    ap.add_argument("--spv", type=int, default=40, help="slices per volume (default 40)")
    ap.add_argument("--ram-frac", type=float, default=0.25,
                    help="fraction of free RAM for the image window (default 0.25)")
    ap.add_argument("--no-transpose", action="store_true",
                    help="do not transpose frames (use if image looks rotated)")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    path = args.file
    if not path:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select .mat stack", "", "MAT files (*.mat *.mat.mat);;All files (*)")
        if not path:
            return

    try:
        stack = LazyMatStack(path, varname=args.var, slices_per_volume=args.spv,
                             transpose=not args.no_transpose)
    except Exception as e:
        QtWidgets.QMessageBox.critical(
            None, "Failed to open file",
            f"{e}\n\nIf this is an old (non-v7.3) .mat, re-save in MATLAB with:\n"
            "  save('file.mat','VarName','-v7.3')")
        return

    frac = float(np.clip(args.ram_frac, 0.02, 0.9))
    k = max(1, min(K_MAX, stack.n_volumes,
                   int(frac * available_ram_bytes() / (TILES_RESIDENT * stack.vol_nbytes))))
    print(f"Opened {path}")
    print(f"  shape (h5py order): {stack.shape} | time axis {stack.time_axis}")
    print(f"  slices/volume {stack.spv} | n_volumes {stack.n_volumes} | plane {stack.H}x{stack.W}")
    print(f"  image window: {k} vols/tile (~{k*stack.vol_nbytes/1e6:.0f} MB) x up to "
          f"{TILES_RESIDENT} tiles; traces are full-length via per-ROI bbox reads")

    w = Viewer(stack, ram_frac=frac)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
