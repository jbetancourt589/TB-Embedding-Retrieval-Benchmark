from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent

OUTPUTS_DIR = PROJECT_DIR / "Outputs"
ANALYSIS_DIR = PROJECT_DIR / "Analysis"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)



# HELPERS TO GET DATA FROM OUTPUTS
def find_column(dataframe, possible_names):

    #Return the first matching column name.
    #Matching is case-insensitive and ignores surrounding spaces.

    normalized_columns = {
        column.strip().lower(): column
        for column in dataframe.columns
    }

    for name in possible_names:
        normalized_name = name.strip().lower()

        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    return None


 #Find a column and return its first valid numeric value.
def get_first_numeric_value(dataframe, possible_columns):

    column = find_column(dataframe, possible_columns)

    if column is None:
        return np.nan

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.iloc[0])


 #Get each index's results by looking at folder name
def clean_index_name(index_folder_name):

    name = index_folder_name.lower()

    if name == "flat":
        return "Flat"

    if name == "hnsw":
        return "HNSW"

    if "1024" in name and "32" in name:
        return "IVF 1024/32"

    if "2048" in name and "64" in name:
        return "IVF 2048/64"

    if "4096" in name and "64" in name:
        return "IVF 4096/64"

    return index_folder_name.replace("_", " ").title()

# Turn a model directory name into a readable graph label.
def readable_model_name(model_folder_name):

    return (
        model_folder_name
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )



# EMBEDDING RESULTS
def collect_embedding_results(model_dir):
    metadata_path = model_dir / "isolate_metadata.csv"
    embeddings_path = model_dir / "isolate_embeddings.npy"

    if not metadata_path.exists():
        return None

    if not embeddings_path.exists():
        return None

    metadata = pd.read_csv(metadata_path)
    embeddings = np.load(
        embeddings_path,
        mmap_mode="r",
    )

    runtime_column = find_column(
        metadata,
        [
            "runtime_seconds",
            "runtime",
            "embedding_runtime_seconds",
            "embedding_time_seconds",
            "time_seconds",
        ],
    )

    gene_units_column = find_column(
        metadata,
        [
            "num_gene_units_embedded",
            "num_gene_units",
            "gene_units",
            "num_genes",
        ],
    )

    chunk_column = find_column(
        metadata,
        [
            "num_chunks",
            "total_chunks",
            "chunks",
            "num_chunks_embedded",
        ],
    )

    result = {
        "model_folder": model_dir.name,
        "model": readable_model_name(model_dir.name),
        "num_isolates": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]),
        "total_embedding_seconds": np.nan,
        "total_embedding_hours": np.nan,
        "average_seconds_per_isolate": np.nan,
        "median_seconds_per_isolate": np.nan,
        "minimum_seconds_per_isolate": np.nan,
        "maximum_seconds_per_isolate": np.nan,
        "average_gene_units": np.nan,
        "average_chunks": np.nan,
    }

    if runtime_column is not None:
        runtimes = pd.to_numeric(
            metadata[runtime_column],
            errors="coerce",
        ).dropna()

        if not runtimes.empty:
            total_seconds = runtimes.sum()

            result.update(
                {
                    "total_embedding_seconds": total_seconds,
                    "total_embedding_hours": total_seconds / 3600,
                    "average_seconds_per_isolate": runtimes.mean(),
                    "median_seconds_per_isolate": runtimes.median(),
                    "minimum_seconds_per_isolate": runtimes.min(),
                    "maximum_seconds_per_isolate": runtimes.max(),
                }
            )

    if gene_units_column is not None:
        gene_units = pd.to_numeric(
            metadata[gene_units_column],
            errors="coerce",
        ).dropna()

        if not gene_units.empty:
            result["average_gene_units"] = gene_units.mean()

    if chunk_column is not None:
        chunks = pd.to_numeric(
            metadata[chunk_column],
            errors="coerce",
        ).dropna()

        if not chunks.empty:
            result["average_chunks"] = chunks.mean()

    return result



# FAISS RESULTS
def calculate_match_rate(metrics):
    """
    Calculate one overall match rate from a metrics CSV.
    """
    match_column = find_column(
        metrics,
        [
            "average_match_rate_at_top_k",
            "average_match_rate",
            "avg_match_rate",
            "match_rate",
            "label_match_rate",
            "mean_match_rate",
            "accuracy",
        ],
    )

    if match_column is None:
        return np.nan

    values = pd.to_numeric(
        metrics[match_column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.mean())


def collect_faiss_result(model_dir, index_dir):
    runtime_files = list(index_dir.glob("*runtime*.csv"))
    metrics_files = list(index_dir.glob("*metrics*.csv"))

    if not runtime_files and not metrics_files:
        return None

    result = {
        "model_folder": model_dir.name,
        "model": readable_model_name(model_dir.name),
        "index": clean_index_name(index_dir.name),
        "build_time_seconds": np.nan,
        "search_time_seconds": np.nan,
        "average_search_seconds_per_isolate": np.nan,
        "index_size_mb": np.nan,
        "average_match_rate": np.nan,
    }

    if runtime_files:
        runtime = pd.read_csv(runtime_files[0])

        result["build_time_seconds"] = get_first_numeric_value(
            runtime,
            [
                "build_time_seconds",
                "build_time",
                "index_build_time_seconds",
                "training_build_time_seconds",
            ],
        )

        result["search_time_seconds"] = get_first_numeric_value(
            runtime,
            [
                "search_time_seconds",
                "search_time",
                "total_search_time_seconds",
            ],
        )

        result["average_search_seconds_per_isolate"] = (
            get_first_numeric_value(
                runtime,
                [
                    "average_search_time_per_isolate_seconds",
                    "average_search_seconds_per_isolate",
                    "avg_search_time_per_isolate",
                    "average_search_time",
                    "avg_search_time",
                ],
            )
        )

        result["index_size_mb"] = get_first_numeric_value(
            runtime,
            [
                "index_size_mb",
                "size_mb",
                "file_size_mb",
            ],
        )

    if metrics_files:
        metrics = pd.read_csv(metrics_files[0])

        result["average_match_rate"] = calculate_match_rate(
            metrics
        )

    return result


def find_faiss_index_directories(model_dir):
    faiss_root = model_dir / "faiss_indexes"

    if not faiss_root.exists():
        return []

    index_directories = []

    for directory in faiss_root.rglob("*"):
        if not directory.is_dir():
            continue

        has_runtime = any(directory.glob("*runtime*.csv"))
        has_metrics = any(directory.glob("*metrics*.csv"))

        if has_runtime or has_metrics:
            index_directories.append(directory)

    return index_directories



# MAIN COLLECTION
def main():
    if not OUTPUTS_DIR.exists():
        raise FileNotFoundError(
            f"Outputs directory not found: {OUTPUTS_DIR}"
        )

    model_directories = sorted(
        directory
        for directory in OUTPUTS_DIR.iterdir()
        if directory.is_dir()
    )

    embedding_results = []
    faiss_results = []

    for model_dir in model_directories:
        embedding_result = collect_embedding_results(model_dir)

        if embedding_result is None:
            print(
                f"Skipping {model_dir.name}: "
                "missing embeddings or metadata."
            )
            continue

        embedding_results.append(embedding_result)

        print(f"Collected embedding results: {model_dir.name}")

        index_directories = find_faiss_index_directories(
            model_dir
        )

        for index_dir in index_directories:
            faiss_result = collect_faiss_result(
                model_dir=model_dir,
                index_dir=index_dir,
            )

            if faiss_result is not None:
                faiss_results.append(faiss_result)

                print(
                    f"  Collected FAISS results: "
                    f"{index_dir.name}"
                )

    embedding_summary = pd.DataFrame(embedding_results)
    faiss_summary = pd.DataFrame(faiss_results)

    embedding_output = (
        ANALYSIS_DIR / "embedding_summary.csv"
    )

    faiss_output = (
        ANALYSIS_DIR / "faiss_summary.csv"
    )

    embedding_summary.to_csv(
        embedding_output,
        index=False,
    )

    faiss_summary.to_csv(
        faiss_output,
        index=False,
    )

    print("\nSaved summary files:")
    print(embedding_output)
    print(faiss_output)


if __name__ == "__main__":
    main()
