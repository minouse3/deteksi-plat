from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import build_paper_style_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-style LPR evaluation tables")
    parser.add_argument("--predictions", required=True, help="Batch inference results.csv")
    parser.add_argument("--ground-truth", help="Optional ground truth CSV")
    parser.add_argument("--output-dir", default="outputs/evaluation/paper_style")
    parser.add_argument("--notebook", default="notebooks/evaluation_results.ipynb")
    parser.add_argument("--title", default="License Plate Recognition Evaluation")
    return parser.parse_args()


def pct(value: float) -> str:
    return f"{value:.2f}%"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        rows.append(values)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown(
    output_path: Path,
    title: str,
    details: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    total = len(details)
    detected = int(details["detected"].sum())
    plate_eval = int(details["has_plate_ground_truth"].sum())
    plate_true = int(details["plate_exact"].sum())
    validity_eval = int(details["has_validity_ground_truth"].sum())
    validity_true = int(details["validity_exact"].sum())

    lines = [
        f"# {title}",
        "",
        "## Results and Discussion",
        "",
        (
            "The recognition result is saved as CSV files for post-processing and "
            "analysis. Accuracy is calculated by comparing predicted plate text "
            "against the available ground truth."
        ),
        "",
        f"Total images evaluated: **{total}**",
        f"Detected plates: **{detected}/{total} ({pct(detected / total * 100 if total else 0)})**",
        (
            f"Plate recognition accuracy: **{plate_true}/{plate_eval} "
            f"({pct(plate_true / plate_eval * 100 if plate_eval else 0)})**"
        ),
        (
            f"Validity text accuracy: **{validity_true}/{validity_eval} "
            f"({pct(validity_true / validity_eval * 100 if validity_eval else 0)})**"
        ),
        "",
        "Accuracy formula:",
        "",
        "```text",
        "accuracy rate (%) = right data total / data total * 100%",
        "```",
        "",
        "## Accuracy Table",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Discussion Notes",
        "",
        (
            "- YOLO detection and OCR are measured separately because a detected "
            "plate can still have incorrect character recognition."
        ),
        (
            "- Character-level errors are counted with edit-distance alignment. "
            "Substitutions and missing expected characters count as errors for "
            "the expected character class."
        ),
        (
            "- Common failure causes are skewed plates, bright glare, protector "
            "covers, screws over characters, non-standard fonts, motion blur, and "
            "border/crop artifacts."
        ),
        (
            "- The KNN OCR baseline remains useful for comparison with the article, "
            "but the result shows why a stronger OCR model is needed for broader "
            "real-world data."
        ),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_charts(output_dir: Path, predictions: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Path]:
    chart_paths = {
        "accuracy": output_dir / "accuracy_by_category.png",
        "validity": output_dir / "validity_status_distribution.png",
        "confidence": output_dir / "detection_confidence_histogram.png",
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(summary["Characters"], summary["Accuracy Rate"], color=["#2f6f8f", "#5f8f3f", "#a45b3f"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy Rate (%)")
    ax.set_title("Accuracy by Category")
    ax.bar_label(bars, fmt="%.1f%%")
    fig.tight_layout()
    fig.savefig(chart_paths["accuracy"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    status_counts = predictions["validity_status"].fillna("unknown").value_counts()
    ax.pie(status_counts.values, labels=status_counts.index, autopct="%1.1f%%")
    ax.set_title("Validity Status Distribution")
    fig.tight_layout()
    fig.savefig(chart_paths["validity"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    confidence = pd.to_numeric(predictions["detection_confidence"], errors="coerce").dropna()
    ax.hist(confidence, bins=20, color="#4f6f9f")
    ax.set_title("YOLO Detection Confidence Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Image Count")
    fig.tight_layout()
    fig.savefig(chart_paths["confidence"], dpi=160)
    plt.close(fig)

    return chart_paths


def notebook_cell(cell_type: str, source: list[str], **kwargs) -> dict:
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cell.update(kwargs)
    return cell


def write_notebook(
    notebook_path: Path,
    predictions_path: Path,
    details_path: Path,
    summary_path: Path,
    chart_paths: dict[str, Path],
) -> None:
    rel_predictions = predictions_path.as_posix()
    rel_details = details_path.as_posix()
    rel_summary = summary_path.as_posix()
    rel_charts = {
        key: Path("..") / path
        for key, path in chart_paths.items()
    }
    cells = [
        notebook_cell(
            "markdown",
            [
                "# License Plate Recognition Evaluation\n",
                "\n",
                "Notebook ini menyimpan analisis hasil batch inference, tabel akurasi bergaya paper, dan grafik ringkas.\n",
            ],
        ),
        notebook_cell(
            "markdown",
            [
                "## Accuracy Formula\n",
                "\n",
                "```text\n",
                "accuracy rate (%) = right data total / data total * 100%\n",
                "```\n",
            ],
        ),
        notebook_cell(
            "markdown",
            [
                "## Generated Charts\n",
                "\n",
                f"![Accuracy by Category]({rel_charts['accuracy'].as_posix()})\n",
                "\n",
                f"![Validity Status Distribution]({rel_charts['validity'].as_posix()})\n",
                "\n",
                f"![Detection Confidence Histogram]({rel_charts['confidence'].as_posix()})\n",
            ],
        ),
        notebook_cell(
            "code",
            [
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                f"predictions_path = Path('{rel_predictions}')\n",
                f"details_path = Path('{rel_details}')\n",
                f"summary_path = Path('{rel_summary}')\n",
                "\n",
                "predictions = pd.read_csv(predictions_path)\n",
                "details = pd.read_csv(details_path)\n",
                "summary = pd.read_csv(summary_path)\n",
                "summary\n",
            ],
        ),
        notebook_cell(
            "code",
            [
                "total = len(details)\n",
                "detected = int(details['detected'].sum())\n",
                "plate_eval = int(details['has_plate_ground_truth'].sum())\n",
                "plate_true = int(details['plate_exact'].sum())\n",
                "validity_eval = int(details['has_validity_ground_truth'].sum())\n",
                "validity_true = int(details['validity_exact'].sum())\n",
                "\n",
                "pd.DataFrame([\n",
                "    {'Metric': 'Detection', 'True': detected, 'Total': total, 'Accuracy Rate': detected / total * 100 if total else 0},\n",
                "    {'Metric': 'Plate exact', 'True': plate_true, 'Total': plate_eval, 'Accuracy Rate': plate_true / plate_eval * 100 if plate_eval else 0},\n",
                "    {'Metric': 'Validity exact', 'True': validity_true, 'Total': validity_eval, 'Accuracy Rate': validity_true / validity_eval * 100 if validity_eval else 0},\n",
                "])\n",
            ],
        ),
        notebook_cell(
            "code",
            [
                "ax = summary.plot.bar(x='Characters', y='Accuracy Rate', legend=False, color=['#2f6f8f', '#5f8f3f', '#a45b3f'])\n",
                "ax.set_ylim(0, 100)\n",
                "ax.set_ylabel('Accuracy Rate (%)')\n",
                "ax.set_title('Accuracy by Category')\n",
                "for container in ax.containers:\n",
                "    ax.bar_label(container, fmt='%.1f%%')\n",
                "plt.tight_layout()\n",
            ],
        ),
        notebook_cell(
            "code",
            [
                "status_counts = predictions['validity_status'].fillna('unknown').value_counts()\n",
                "ax = status_counts.plot.pie(autopct='%1.1f%%', ylabel='', title='Validity Status Distribution')\n",
                "plt.tight_layout()\n",
            ],
        ),
        notebook_cell(
            "code",
            [
                "conf = pd.to_numeric(predictions['detection_confidence'], errors='coerce').dropna()\n",
                "ax = conf.plot.hist(bins=20, color='#4f6f9f')\n",
                "ax.set_title('YOLO Detection Confidence Distribution')\n",
                "ax.set_xlabel('Confidence')\n",
                "plt.tight_layout()\n",
            ],
        ),
        notebook_cell(
            "code",
            [
                "mismatches = details[details['has_plate_ground_truth'] & ~details['plate_exact']]\n",
                "cols = ['image', 'plate_expected_norm', 'plate_predicted_norm', 'validity_expected_norm', 'validity_predicted_norm', 'detection_confidence', 'error']\n",
                "mismatches[cols].head(30)\n",
            ],
        ),
        notebook_cell(
            "markdown",
            [
                "## Discussion\n",
                "\n",
                "Kesalahan OCR paling sering muncul pada plate miring, glare, baut/protector yang menutup karakter, font non-standar, serta hasil crop yang masih menyertakan border atau area luar plate. KNN OCR dipertahankan sebagai baseline sesuai artikel, tetapi batch besar ini menunjukkan bahwa tahap OCR perlu ditingkatkan untuk penggunaan real-world.\n",
            ],
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(predictions_path)
    ground_truth = pd.read_csv(args.ground_truth) if args.ground_truth else None
    details, summary = build_paper_style_report(predictions, ground_truth=ground_truth)

    details_path = output_dir / "paper_style_details.csv"
    summary_path = output_dir / "paper_style_summary.csv"
    markdown_path = output_dir / "results_and_discussion.md"
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_markdown(markdown_path, args.title, details, summary)
    chart_paths = write_charts(output_dir, predictions, summary)
    write_notebook(Path(args.notebook), predictions_path, details_path, summary_path, chart_paths)

    print(summary.to_string(index=False, formatters={"Accuracy Rate": "{:.2f}%".format}))
    print(f"\nWrote details to {details_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote discussion to {markdown_path}")
    print(f"Wrote notebook to {args.notebook}")
    for path in chart_paths.values():
        print(f"Wrote chart to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
