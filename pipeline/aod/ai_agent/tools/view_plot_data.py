"""
Tool that shows the data behind saved plots as text (hackathon patch P4, replaces view_images).

Every figure saved by executed code gets a <image>.json beside it (see harness_site/sitecustomize.py).
This tool renders that record compactly, so a text-only model can read what a plot shows.
"""
import json
import os

from tools.tool import Tool
from utility.md_reporter import MDReporter

_MAX_CHARS = 8000


def _fmt(values) -> str:
    return "[" + ", ".join("nan" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
                           for v in values) + "]"


def _render(data: dict) -> str:
    lines = [f"## {data['figure']}"]
    for i, ax in enumerate(data["axes"]):
        lines.append(f"### axes {i}: {ax['title'] or '(no title)'}")
        lines.append(f"x: {ax['xlabel'] or '(no label)'} ({ax['xscale']}, limits {_fmt(ax['xlim'])})")
        lines.append(f"y: {ax['ylabel'] or '(no label)'} ({ax['yscale']}, limits {_fmt(ax['ylim'])})")
        if ax["legend"]:
            lines.append(f"legend: {ax['legend']}")
        for s in ax["lines"]:
            lines.append(f"line '{s['label']}' ({s['n']} points): x={_fmt(s['x'])} y={_fmt(s['y'])}")
        for s in ax["bars"]:
            lines.append(f"bars '{s['label']}' ({s['n']} bars): left={_fmt(s['left'])} "
                         f"width={_fmt(s['width'])} height={_fmt(s['height'])}")
        for s in ax["scatter"]:
            lines.append(f"scatter '{s['label']}' ({s['n']} points): x={_fmt(s['x'])} y={_fmt(s['y'])}")
        for s in ax["images"]:
            lines.append(f"image shape={s['shape']} extent={_fmt(s['extent'])} "
                         f"range=[{s.get('min')}, {s.get('max')}]")
        for t in ax["texts"]:
            lines.append(f"text: {t}")
    text = "\n".join(lines)
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + "\n... (truncated)"
    return text


class ViewPlotData(Tool):
    """
    Shows the plotted data of saved figures as text.
    """
    def __init__(self, work_dir: str, reporter: MDReporter):
        name = "view_plot_data"
        schema = Tool.build_function_schema(
            name,
            """
            Shows the data behind saved plots as text: title, axis labels,
            scales and limits, legend, and the values of every line,
            histogram/bar, scatter and image (up to 200 points per series).
            Pass the image filenames your scripts saved, e.g. "mjj_hist.png".
            """,
            [Tool.build_parameter_schema(
                "filenames", "List of image filenames whose data you want to see.", "string", array=True)]
        )
        super().__init__(name, schema, self.view_plot_data, reporter)
        self.work_dir = work_dir

    def view_plot_data(self, filenames: list[str]) -> str:
        parts, missing = [], []
        for fn in filenames:
            path = os.path.join(self.work_dir, fn + ".json")
            if not os.path.isfile(path):
                missing.append(fn)
                continue
            with open(path, encoding="utf-8") as f:
                parts.append(_render(json.load(f)))
        self.reporter.report_metrics("view_plot_data_success", len(parts), mode="add")
        self.reporter.report_metrics("view_plot_data_failure", len(missing), mode="add")
        if missing:
            parts.append(f"No plot data recorded for: {missing} (file missing, or not saved with savefig).")
        return "\n\n".join(parts)
