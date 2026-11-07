"""
Loaded by every Python process that execute_python starts (AOD_SITE_DIR is put on PYTHONPATH).

- P3: seeds `random` and NumPy's legacy global RNG from HARNESS_SEED, so code that leaves
  random_state=None (scikit-learn) or calls np.random.* repeats exactly.
- P4: every matplotlib figure saved with savefig() also gets <file>.json beside it, holding the
  plotted data. The text-only view_plot_data tool shows that instead of the image.

Failures here never break the agent's script: plot recording is best effort.
"""
import importlib.abc
import json
import os
import random
import sys

_SEED = int(os.environ.get("HARNESS_SEED", "0"))
random.seed(_SEED)
try:
    import numpy as _np

    _np.random.seed(_SEED)
except ImportError:
    _np = None

_MAX_POINTS = 200


def _numbers(values):
    """At most _MAX_POINTS evenly spaced values as floats (or strings if not numeric)."""
    values = list(values)
    n = len(values)
    if n > _MAX_POINTS:
        step = (n - 1) / (_MAX_POINTS - 1)
        values = [values[round(i * step)] for i in range(_MAX_POINTS)]
    out = []
    for v in values:
        try:
            f = float(v)
            out.append(None if f != f else round(f, 6))  # NaN -> null
        except (TypeError, ValueError):
            out.append(str(v))
    return out


def _points(xy):
    xy = list(xy)
    return {"n": len(xy), "x": _numbers(p[0] for p in xy), "y": _numbers(p[1] for p in xy)}


def _axes_data(ax):
    from matplotlib.container import BarContainer

    entry = {
        "title": ax.get_title(), "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
        "xscale": ax.get_xscale(), "yscale": ax.get_yscale(),
        "xlim": _numbers(ax.get_xlim()), "ylim": _numbers(ax.get_ylim()),
        "lines": [], "bars": [], "scatter": [], "images": [], "texts": [], "legend": [],
    }
    for line in ax.get_lines():
        x, y = line.get_xdata(), line.get_ydata()
        entry["lines"].append({"label": line.get_label(), "n": len(x), "x": _numbers(x), "y": _numbers(y)})
    for container in ax.containers:
        if isinstance(container, BarContainer):
            rects = list(container)
            # hist() puts the user's label on the first bar; the container gets "_container0".
            labels = [container.get_label()] + ([rects[0].get_label()] if rects else [])
            label = next((lb for lb in labels if lb and not lb.startswith("_")), "")
            entry["bars"].append({
                "label": label, "n": len(rects),
                "left": _numbers(r.get_x() for r in rects),
                "width": _numbers(r.get_width() for r in rects),
                "height": _numbers(r.get_height() for r in rects),
            })
    for coll in ax.collections:
        offsets = coll.get_offsets() if hasattr(coll, "get_offsets") else []
        if len(offsets):
            entry["scatter"].append({"label": coll.get_label(), **_points(offsets)})
    for im in ax.get_images():
        arr = im.get_array()
        info = {"shape": list(arr.shape), "extent": _numbers(im.get_extent())}
        if _np is not None and arr.size:
            info["min"], info["max"] = float(_np.nanmin(arr)), float(_np.nanmax(arr))
        entry["images"].append(info)
    entry["texts"] = [t.get_text() for t in ax.texts if t.get_text()]
    legend = ax.get_legend()
    if legend is not None:
        entry["legend"] = [t.get_text() for t in legend.get_texts()]
    # Histograms drawn with histtype='step' are polygons, not bars.
    for patch in ax.patches:
        if type(patch).__name__ == "Polygon" and patch.get_label() and not patch.get_label().startswith("_"):
            entry["lines"].append({"label": patch.get_label(), **_points(patch.get_xy())})
    return entry


def _record(fig, fname, kwargs):
    import matplotlib

    path = os.fspath(fname)
    if not os.path.splitext(path)[1]:
        path += "." + (kwargs.get("format") or matplotlib.rcParams["savefig.format"])
    data = {"figure": os.path.basename(path), "axes": [_axes_data(ax) for ax in fig.get_axes()]}
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(data, f)


def _patch_figure(module):
    original = module.Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        result = original(self, fname, *args, **kwargs)
        if isinstance(fname, (str, os.PathLike)):
            try:
                _record(self, fname, kwargs)
            except Exception:
                pass
        return result

    module.Figure.savefig = savefig


class _FigureImportHook(importlib.abc.MetaPathFinder):
    """Patches matplotlib.figure right after it is imported, without importing it eagerly."""

    def find_spec(self, name, path, target=None):
        if name != "matplotlib.figure":
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch_figure(module)

        spec.loader.exec_module = exec_and_patch
        return spec


sys.meta_path.insert(0, _FigureImportHook())
