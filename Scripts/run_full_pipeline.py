from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data"
FASTA_DIR = DATA_DIR / "IR_Variable"
TARGETS_JSON = DATA_DIR / "cryptic_targets_all.json"

EMBEDDING_SCRIPTS = {
    "nucleotide-transformer": PROJECT_DIR
    / "Scripts"
    / "model-1-nucleotide-transformer-500m-human-ref"
    / "embed_isolates.py",
    "dnabert-1-6mer": PROJECT_DIR
    / "Scripts"
    / "model-2-dnabert-1-6mer"
    / "embed_isolates.py",
}

FAISS_SCRIPTS = {
    "flat": PROJECT_DIR / "Scripts" / "faiss_indexes" / "flat_build_search.py",
    "hnsw": PROJECT_DIR / "Scripts" / "faiss_indexes" / "hnsw_build_search.py",
    "ivf": PROJECT_DIR / "Scripts" / "faiss_indexes" / "ivf_build_search.py",
}

ANALYSIS_SCRIPTS = [
    PROJECT_DIR / "Scripts" / "analysis" / "collect_results.py",
    PROJECT_DIR / "Scripts" / "analysis" / "generate_figures.py",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run embeddings, FAISS build/search, summary collection, "
            "and figure generation for the TB isolate benchmark."
        )
    )

    parser.add_argument(
        "--max-fasta-files",
        type=int,
        default=None,
        help="Limit each embedding model to the first N FASTA files for a quick test.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(EMBEDDING_SCRIPTS),
        default=sorted(EMBEDDING_SCRIPTS),
        help="Embedding models to run.",
    )
    parser.add_argument(
        "--indexes",
        nargs="+",
        choices=sorted(FAISS_SCRIPTS),
        default=sorted(FAISS_SCRIPTS),
        help="FAISS index families to build and search.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Reuse existing files in Outputs/<model>/ and skip embedding generation.",
    )
    parser.add_argument(
        "--skip-faiss",
        action="store_true",
        help="Reuse existing FAISS outputs and skip index build/search.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip Analysis/*.csv and Figures/*.png generation.",
    )

    return parser.parse_args()


def validate_data():
    if not FASTA_DIR.exists():
        raise FileNotFoundError(
            f"Missing FASTA directory: {FASTA_DIR}\n"
            "Create Data/IR_Variable and add *_IR_Genes.fasta files."
        )

    fasta_files = sorted(FASTA_DIR.glob("*.fasta"))

    if not fasta_files:
        raise FileNotFoundError(
            f"No FASTA files found in: {FASTA_DIR}\n"
            "Add files named like <isolate-id>_IR_Genes.fasta."
        )

    if not TARGETS_JSON.exists():
        raise FileNotFoundError(
            f"Missing target-label JSON: {TARGETS_JSON}\n"
            "Add Data/cryptic_targets_all.json before running embeddings."
        )


def run_script(script_path: Path, env: dict[str, str]):
    relative_path = script_path.relative_to(PROJECT_DIR)
    print(f"\nRunning {relative_path}")

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_DIR,
        env=env,
        check=True,
    )


def print_output_summary():
    print("\nPipeline complete. Generated outputs are under:")
    print("  Outputs/<model>/")
    print("  Outputs/<model>/faiss_indexes/<index>/")
    print("  Analysis/")
    print("  Figures/")


def main():
    args = parse_args()
    env = os.environ.copy()

    if args.max_fasta_files is not None:
        if args.max_fasta_files <= 0:
            raise ValueError("--max-fasta-files must be a positive integer.")

        env["TB_MAX_FASTA_FILES"] = str(args.max_fasta_files)

    if not args.skip_embeddings:
        validate_data()

        for model_name in args.models:
            run_script(EMBEDDING_SCRIPTS[model_name], env)

    if not args.skip_faiss:
        for index_name in args.indexes:
            run_script(FAISS_SCRIPTS[index_name], env)

    if not args.skip_analysis:
        for script_path in ANALYSIS_SCRIPTS:
            run_script(script_path, env)

    print_output_summary()


if __name__ == "__main__":
    main()
