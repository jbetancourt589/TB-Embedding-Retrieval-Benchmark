from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap




# PATHS
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent

OUTPUTS_DIR = PROJECT_DIR / "Outputs"
ANALYSIS_DIR = PROJECT_DIR / "Analysis"
FIGURES_DIR = PROJECT_DIR / "Figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


EMBEDDING_SUMMARY_PATH = (
    ANALYSIS_DIR / "embedding_summary.csv"
)

FAISS_SUMMARY_PATH = (
    ANALYSIS_DIR / "faiss_summary.csv"
)


INDEX_ORDER = [
    "Flat",
    "HNSW",
    "IVF 1024/32",
    "IVF 2048/64",
    "IVF 4096/64",
]




# GENERAL HELPERS
def safe_filename(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def save_figure(filename):
    output_path = FIGURES_DIR / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def load_summaries():
    if not EMBEDDING_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Run collect_results.py first. Missing: "
            f"{EMBEDDING_SUMMARY_PATH}"
        )

    if not FAISS_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Run collect_results.py first. Missing: "
            f"{FAISS_SUMMARY_PATH}"
        )

    embedding_summary = pd.read_csv(
        EMBEDDING_SUMMARY_PATH
    )

    faiss_summary = pd.read_csv(
        FAISS_SUMMARY_PATH
    )

    return embedding_summary, faiss_summary




# FIGURE 1: EMBEDDING RUNTIME
def plot_embedding_runtime(embedding_summary):
    data = embedding_summary.dropna(
        subset=["total_embedding_hours"]
    )

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        data["model"],
        data["total_embedding_hours"],
    )

    plt.xlabel("Embedding model")
    plt.ylabel("Total embedding time (hours)")
    plt.title("Total embedding runtime by model")
    plt.xticks(rotation=25, ha="right")

    plt.bar_label(
        bars,
        fmt="%.2f",
        padding=3,
    )

    plt.tight_layout()

    save_figure("embedding_total_runtime.png")


def plot_average_embedding_runtime(embedding_summary):
    data = embedding_summary.dropna(
        subset=["average_seconds_per_isolate"]
    )

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        data["model"],
        data["average_seconds_per_isolate"],
    )

    plt.xlabel("Embedding model")
    plt.ylabel("Average time per isolate (seconds)")
    plt.title("Average embedding runtime per isolate")
    plt.xticks(rotation=25, ha="right")

    plt.bar_label(
        bars,
        fmt="%.2f",
        padding=3,
    )

    plt.tight_layout()

    save_figure("embedding_average_runtime.png")




# FIGURE 2: FAISS PERFORMANCE
def grouped_faiss_plot(
    faiss_summary,
    metric,
    ylabel,
    title,
    filename,
    multiply_by_100=False,
):
    data = faiss_summary.copy()

    if multiply_by_100:
        data[metric] = data[metric] * 100

    pivot = data.pivot_table(
        index="index",
        columns="model",
        values=metric,
        aggfunc="mean",
    )

    available_order = [
        index_name
        for index_name in INDEX_ORDER
        if index_name in pivot.index
    ]

    additional_indexes = [
        index_name
        for index_name in pivot.index
        if index_name not in available_order
    ]

    pivot = pivot.reindex(
        available_order + additional_indexes
    )

    axis = pivot.plot(
        kind="bar",
        figsize=(11, 6),
    )

    axis.set_xlabel("FAISS index")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(
        axis="x",
        rotation=25,
    )

    axis.legend(
        title="Embedding model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()

    save_figure(filename)


def plot_faiss_performance(faiss_summary):
    grouped_faiss_plot(
        faiss_summary=faiss_summary,
        metric="build_time_seconds",
        ylabel="Build time (seconds)",
        title="FAISS index build time",
        filename="faiss_build_time.png",
    )

    grouped_faiss_plot(
        faiss_summary=faiss_summary,
        metric="search_time_seconds",
        ylabel="Total search time (seconds)",
        title="FAISS search time",
        filename="faiss_search_time.png",
    )

    grouped_faiss_plot(
        faiss_summary=faiss_summary,
        metric="index_size_mb",
        ylabel="Index size (MB)",
        title="FAISS index size",
        filename="faiss_index_size.png",
    )

    grouped_faiss_plot(
        faiss_summary=faiss_summary,
        metric="average_match_rate",
        ylabel="Average label match rate (%)",
        title="Top-5 antibiotic-label match rate",
        filename="faiss_match_rate.png",
        multiply_by_100=True,
    )



# FIGURE 3: UMAP
def load_model_for_umap(model_row):
    model_dir = (
        OUTPUTS_DIR / model_row["model_folder"]
    )

    embeddings_path = (
        model_dir / "isolate_embeddings.npy"
    )

    metadata_path = (
        model_dir / "isolate_metadata.csv"
    )

    if not embeddings_path.exists():
        raise FileNotFoundError(embeddings_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    embeddings = np.load(
        embeddings_path
    ).astype("float32")

    metadata = pd.read_csv(metadata_path)

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"{model_row['model']}: "
            "embedding and metadata row counts differ."
        )

    return embeddings, metadata


def run_umap(
    embeddings,
    n_neighbors=15,
    min_dist=0.1,
):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )

    return reducer.fit_transform(embeddings)


def plot_umap_comparison(
    embedding_summary,
    color_column="RIF",
):
    umap_results = []

    for _, model_row in embedding_summary.iterrows():
        model_name = model_row["model"]

        embeddings, metadata = load_model_for_umap(
            model_row
        )

        if color_column not in metadata.columns:
            print(
                f"Skipping UMAP for {model_name}: "
                f"missing column '{color_column}'."
            )
            continue

        print(
            f"Running UMAP for {model_name}: "
            f"{embeddings.shape}"
        )

        coordinates = run_umap(embeddings)

        model_umap = pd.DataFrame(
            {
                "model": model_name,
                "isolate_id": metadata["isolate_id"],
                "UMAP_1": coordinates[:, 0],
                "UMAP_2": coordinates[:, 1],
                color_column: metadata[color_column],
            }
        )

        coordinate_path = (
            ANALYSIS_DIR
            / (
                safe_filename(model_name)
                + f"_umap_{color_column}.csv"
            )
        )

        model_umap.to_csv(
            coordinate_path,
            index=False,
        )

        umap_results.append(
            (model_name, model_umap)
        )

    if not umap_results:
        print("No UMAP figures were generated.")
        return

    number_of_models = len(umap_results)
    number_of_columns = min(2, number_of_models)
    number_of_rows = ceil(
        number_of_models / number_of_columns
    )

    figure, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(
            7 * number_of_columns,
            6 * number_of_rows,
        ),
        squeeze=False,
    )

    flat_axes = axes.flatten()

    for axis, (model_name, model_umap) in zip(
        flat_axes,
        umap_results,
    ):
        color_values = pd.to_numeric(
            model_umap[color_column],
            errors="coerce",
        )

        valid = color_values.notna()

        scatter = axis.scatter(
            model_umap.loc[valid, "UMAP_1"],
            model_umap.loc[valid, "UMAP_2"],
            c=color_values.loc[valid],
            s=8,
            alpha=0.65,
        )

        axis.set_title(model_name)
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")

        figure.colorbar(
            scatter,
            ax=axis,
            label=f"{color_column} label",
        )

    for unused_axis in flat_axes[len(umap_results):]:
        unused_axis.set_visible(False)

    figure.suptitle(
        f"UMAP projections colored by {color_column}",
        y=1.01,
    )

    plt.tight_layout()

    save_figure(
        f"umap_comparison_{color_column}.png"
    )



# MAIN
def main():
    embedding_summary, faiss_summary = (
        load_summaries()
    )

    plot_embedding_runtime(
        embedding_summary
    )

    plot_average_embedding_runtime(
        embedding_summary
    )

    plot_faiss_performance(
        faiss_summary
    )

    plot_umap_comparison(
        embedding_summary,
        color_column="RIF",
    )

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()