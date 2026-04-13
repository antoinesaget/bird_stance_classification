"""Purpose: Provide shared dataset counting and plotting helpers for split/crop/dataset artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

LABEL_FIELDS = ["readability", "specie", "behavior", "substrate", "stance"]


def counts_by_value(frame, column: str) -> dict[str, int]:
    if frame.empty:
        return {}
    values = frame[column].fillna("<null>").value_counts().sort_index().to_dict()
    return {str(key): int(value) for key, value in values.items()}


def visible_label_counts(frame) -> dict[str, dict[str, int]]:
    if frame.empty:
        return {field: {} for field in LABEL_FIELDS}
    return {
        field: {key: value for key, value in counts_by_value(frame, field).items() if key != "<null>"}
        for field in LABEL_FIELDS
    }


def diff_nested_counts(
    current: dict[str, dict[str, int]],
    previous: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for field in sorted(set(current) | set(previous)):
        current_counts = current.get(field, {})
        previous_counts = previous.get(field, {})
        out[field] = {
            label: int(current_counts.get(label, 0)) - int(previous_counts.get(label, 0))
            for label in sorted(set(current_counts) | set(previous_counts))
        }
    return out


def absent_class_warnings(
    *,
    baseline: dict[str, dict[str, int]],
    test_counts: dict[str, dict[str, int]],
    fold_counts: dict[int, dict[str, dict[str, int]]],
) -> list[str]:
    warnings: list[str] = []
    for field, labels in baseline.items():
        for label, total in labels.items():
            if int(total) <= 0:
                continue
            if int(test_counts.get(field, {}).get(label, 0)) <= 0:
                warnings.append(f"test missing {field}={label}")
            for fold_id, counts in sorted(fold_counts.items()):
                if int(counts.get(field, {}).get(label, 0)) <= 0:
                    warnings.append(f"fold {fold_id} missing {field}={label}")
    return warnings


def import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to render dataset plots.") from exc
    return plt


def save_plot(fig, base_path: Path, *, tight_rect: tuple[float, float, float, float] | None = None) -> dict[str, str]:
    png_path = base_path.with_suffix(".png")
    svg_path = base_path.with_suffix(".svg")
    fig.tight_layout(rect=tight_rect)
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return {"png": png_path.name, "svg": svg_path.name}


def format_count(value: int | float) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.2f}"
    return f"{int(value):,}"


def add_bar_headroom(ax, values: list[int | float], *, extra_ratio: float = 0.18, min_top: float = 1.0) -> None:
    if not values:
        return
    max_value = max(float(value) for value in values)
    top = max(max_value * (1.0 + extra_ratio), min_top)
    bottom, current_top = ax.get_ylim()
    ax.set_ylim(bottom, max(top, current_top))


def annotate_bars(ax, bars, texts: list[str], *, padding: float = 3.0, fontsize: int = 9) -> None:
    for bar, text in zip(bars, texts):
        height = bar.get_height()
        baseline = max(height, 0)
        ax.annotate(
            text,
            xy=(bar.get_x() + bar.get_width() / 2.0, baseline),
            xytext=(0, padding),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#222222",
        )


def embed_plot_block(title: str, plot_name: str, plot_files: dict[str, dict[str, str]]) -> list[str]:
    files = plot_files[plot_name]
    return [
        f"### {title}",
        "",
        f"![{title}](plots/{files['png']})",
        "",
        f"`SVG:` `plots/{files['svg']}`",
        "",
    ]


def json_safe(value: Any) -> Any:
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return value
    return value
