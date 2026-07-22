# Builds IVF FAISS indexes from existing isolate_embeddings.npy files.
# This script does not create embeddings.

from pathlib import Path
import time

import faiss
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"

TOP_K = 5

IVF_CONFIGS = [
    (1024, 32),
    (2048, 64),
    (4096, 64),
]

ANTIBIOTICS = [
    "AMI", "BDQ", "CFZ", "DLM", "EMB", "ETH", "INH",
    "KAN", "LEV", "LZD", "MXF", "RIF", "RFB"
]

REQUIRED_FILES = ("isolate_embeddings.npy", "isolate_metadata.csv")


def discover_model_dirs():
    if not OUTPUTS_DIR.exists():
        raise FileNotFoundError(f"Outputs directory not found: {OUTPUTS_DIR}")

    model_dirs = []
    skipped_dirs = []

    for model_dir in sorted(path for path in OUTPUTS_DIR.iterdir() if path.is_dir()):
        missing_files = [
            filename for filename in REQUIRED_FILES
            if not (model_dir / filename).exists()
        ]

        if missing_files:
            skipped_dirs.append((model_dir.name, missing_files))
            continue

        model_dirs.append(model_dir)

    print(f"Models found: {len(model_dirs)}")
    for model_dir in model_dirs:
        print(f"  - {model_dir.name}")

    if skipped_dirs:
        print("\nSkipped folders missing required files:")
        for folder_name, missing_files in skipped_dirs:
            print(f"  - {folder_name}: missing {', '.join(missing_files)}")

    print()
    return model_dirs


def load_embeddings_and_metadata(model_dir):
    embeddings_path = model_dir / "isolate_embeddings.npy"
    metadata_path = model_dir / "isolate_metadata.csv"

    embeddings = np.load(embeddings_path).astype("float32")
    metadata = pd.read_csv(metadata_path)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    num_vectors, dimension = embeddings.shape

    if len(metadata) != num_vectors:
        raise ValueError("Metadata rows do not match embedding rows.")

    print("Embeddings shape:", embeddings.shape)
    print("Metadata shape:", metadata.shape)

    return embeddings, metadata, num_vectors, dimension


def build_neighbor_results(metadata, scores, neighbor_indices, num_vectors):
    results = []

    for query_idx in range(num_vectors):
        query_id = metadata.iloc[query_idx]["isolate_id"]
        neighbors_added = 0

        for rank, neighbor_idx in enumerate(neighbor_indices[query_idx]):
            if neighbor_idx == query_idx:
                continue

            if neighbor_idx < 0:
                continue

            if neighbors_added >= TOP_K:
                break

            neighbor_id = metadata.iloc[neighbor_idx]["isolate_id"]
            similarity_score = scores[query_idx][rank]

            row = {
                "query_index": query_idx,
                "query_isolate": query_id,
                "neighbor_rank": neighbors_added + 1,
                "neighbor_index": int(neighbor_idx),
                "neighbor_isolate": neighbor_id,
                "similarity_score": similarity_score,
            }

            for drug in ANTIBIOTICS:
                query_label = metadata.iloc[query_idx].get(drug)
                neighbor_label = metadata.iloc[neighbor_idx].get(drug)

                row[f"query_{drug}"] = query_label
                row[f"neighbor_{drug}"] = neighbor_label

                if pd.isna(query_label) or pd.isna(neighbor_label):
                    row[f"{drug}_match"] = np.nan
                else:
                    row[f"{drug}_match"] = int(query_label == neighbor_label)

            results.append(row)
            neighbors_added += 1

    return pd.DataFrame(results)


def calculate_metrics(results_df, model_name):
    metrics = []

    for drug in ANTIBIOTICS:
        match_col = f"{drug}_match"

        if match_col in results_df.columns:
            valid_matches = results_df[match_col].dropna()
        else:
            valid_matches = pd.Series(dtype="float64")

        metrics.append({
            "model_name": model_name,
            "drug": drug,
            "average_match_rate_at_top_k": (
                valid_matches.mean() if len(valid_matches) > 0 else np.nan
            ),
            "valid_comparisons": len(valid_matches),
            "top_k": TOP_K,
        })

    return pd.DataFrame(metrics)


def run_ivf_configuration(
    model_name,
    model_dir,
    embeddings,
    metadata,
    num_vectors,
    dimension,
    nlist,
    nprobe,
):
    index_name = f"ivf_{nlist}_{nprobe}"
    index_dir = model_dir / "faiss_indexes" / index_name
    index_dir.mkdir(parents=True, exist_ok=True)

    index_out = index_dir / f"{index_name}.index"
    metadata_out = index_dir / f"{index_name}_metadata.csv"
    neighbors_out = index_dir / f"{index_name}_neighbors.csv"
    metrics_out = index_dir / f"{index_name}_metrics.csv"
    runtime_out = index_dir / f"{index_name}_runtime.csv"

    print(f"Running {index_name}...")

    if nlist > num_vectors:
        raise ValueError(
            f"{index_name} cannot run because nlist={nlist} "
            f"is larger than number of vectors={num_vectors}."
        )

    print("Building index...")
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        nlist,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.nprobe = nprobe

    start_build = time.time()
    index.train(embeddings)
    index.add(embeddings)
    build_time = time.time() - start_build

    faiss.write_index(index, str(index_out))
    metadata.to_csv(metadata_out, index=False)

    print("Searching all isolates...")
    start_search = time.time()
    scores, neighbor_indices = index.search(embeddings, TOP_K + 1)
    search_time = time.time() - start_search

    print("Saving results...")
    results_df = build_neighbor_results(metadata, scores, neighbor_indices, num_vectors)
    results_df.to_csv(neighbors_out, index=False)

    metrics_df = calculate_metrics(results_df, model_name)
    metrics_df.to_csv(metrics_out, index=False)

    index_size_mb = index_out.stat().st_size / (1024 * 1024)

    runtime_df = pd.DataFrame([{
        "model_name": model_name,
        "index_name": index_name,
        "index_type": "ivf",
        "nlist": nlist,
        "nprobe": nprobe,
        "build_time_seconds": build_time,
        "search_time_seconds": search_time,
        "average_search_time_per_isolate_seconds": search_time / num_vectors,
        "index_size_mb": index_size_mb,
        "number_of_vectors": num_vectors,
        "embedding_dimension": dimension,
        "top_k": TOP_K,
    }])
    runtime_df.to_csv(runtime_out, index=False)


def process_model(model_dir):
    model_name = model_dir.name

    print("Loading existing embeddings...")
    embeddings, metadata, num_vectors, dimension = load_embeddings_and_metadata(model_dir)

    faiss.normalize_L2(embeddings)

    for nlist, nprobe in IVF_CONFIGS:
        run_ivf_configuration(
            model_name,
            model_dir,
            embeddings,
            metadata,
            num_vectors,
            dimension,
            nlist,
            nprobe,
        )


def print_summary(successful_models, failed_models):
    print("\nSummary")
    print(f"Successful models: {len(successful_models)}")
    for model_name in successful_models:
        print(f"  - {model_name}")

    print(f"Failed models: {len(failed_models)}")
    for model_name, error in failed_models:
        print(f"  - {model_name}: {error}")


def main():
    model_dirs = discover_model_dirs()
    successful_models = []
    failed_models = []

    for model_number, model_dir in enumerate(model_dirs, start=1):
        print(f"[{model_number}/{len(model_dirs)}] Processing {model_dir.name}")

        try:
            process_model(model_dir)
        except Exception as error:
            print(f"Failed {model_dir.name}: {error}\n")
            failed_models.append((model_dir.name, error))
            continue

        print(f"Completed {model_dir.name}\n")
        successful_models.append(model_dir.name)

    print_summary(successful_models, failed_models)


if __name__ == "__main__":
    main()
