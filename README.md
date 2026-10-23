# TB Isolate Embedding and Retrieval Benchmark

**Genomic Foundation Models | Machine Learning | FAISS Vector Search | Computational Genomics**

## Overview

This project implements a Python-based genomic dense-retrieval framework for analyzing *Mycobacterium tuberculosis* (MTB) isolates using pretrained DNA language models and approximate nearest-neighbor search.

The framework transforms genomic sequences into numerical embeddings, indexes those embeddings using FAISS, and retrieves genetically similar isolates to evaluate relationships between genomic representations and antibiotic resistance phenotypes.

The project focuses on three research questions:

1. How effectively can pretrained DNA foundation models represent genomic variation across MTB isolates?
2. How do different FAISS indexing strategies compare in retrieval speed, memory requirements, and label agreement?
3. How does the choice of genomic sequence representation influence antibiotic resistance-related retrieval performance?

The system is intended for computational genomics research and benchmarking, not clinical diagnosis.

## Technologies

**Language:** Python

**Machine Learning:** PyTorch, Hugging Face Transformers, Nucleotide Transformer 500M, DNABERT-1

**Vector Search:** FAISS Flat, HNSW, IVF

**Data Processing:** NumPy, pandas, FASTA processing

**Analysis:** Matplotlib, UMAP, classification and retrieval metrics

**Research Computing:** Linux, CUDA, Conda, SLURM, UF HiPerGator

Some additional experiments are conducted in an HPC environment and are not fully represented by the public repository.

## Pipeline

```text
MTB Isolate FASTA Files
         |
         v
Genomic Sequence Processing
         |
         v
DNA Foundation Models
(Nucleotide Transformer / DNABERT)
         |
         v
Gene-Level Embeddings
         |
         v
Isolate-Level Embedding Aggregation
         |
         v
FAISS Vector Indexing
(Flat / HNSW / IVF)
         |
         v
Top-5 Nearest-Neighbor Retrieval
         |
         v
Antibiotic Resistance Label Comparison
         |
         v
Performance Evaluation and Visualization
```

Antibiotic susceptibility labels are **not used to determine nearest neighbors**. Retrieval is based on genomic embedding similarity. Labels are compared afterward to evaluate phenotype agreement.

## Research Components

### 1. Genomic Embedding Generation

The framework generates isolate-level representations using two pretrained DNA language models:

- `InstaDeepAI/nucleotide-transformer-500m-human-ref`
- `zhihan1996/DNA_bert_6`

Both models are used as feature extractors without additional fine-tuning.

The provided FASTA files contain gene-centered sequence regions organized into groups of three:

```text
BEFORE region
GENE region
AFTER region
```

For each valid gene unit, the pipeline combines these regions and generates a sequence embedding.

The BEFORE and AFTER regions preserve neighboring genomic context, which may contain regulatory variation relevant to antibiotic resistance.

Gene-unit representations are then aggregated through mean pooling to produce a fixed-length embedding for each isolate.

### 2. FAISS Similarity Search

The project benchmarks multiple FAISS index types:

| Index | Description |
|---|---|
| Flat | Exact nearest-neighbor search |
| HNSW | Graph-based approximate nearest-neighbor search |
| IVF 1024/32 | Inverted-file search with 1,024 clusters and 32 search probes |
| IVF 2048/64 | Inverted-file search with 2,048 clusters and 64 search probes |
| IVF 4096/64 | Inverted-file search with 4,096 clusters and 64 search probes |

Each index operates on the generated isolate embeddings.

The comparison evaluates tradeoffs between search speed, index construction time, storage requirements, and retrieval behavior.

### 3. Antibiotic Resistance Evaluation

For each query isolate, FAISS retrieves the five nearest neighbors based on embedding similarity.

The pipeline then compares antibiotic susceptibility labels to determine how frequently retrieved neighbors share the query isolate's phenotype.

The principal retrieval metric is **Match@5**, which summarizes agreement between the query's antibiotic label and the labels of its five retrieved neighbors.

Because antibiotic resistance labels may be imbalanced, high agreement does not necessarily imply strong detection of resistant isolates. Additional classification metrics are useful for evaluating resistant-class performance.

### 4. Benchmarking and Visualization

The framework collects:

- Embedding generation runtime
- Average embedding runtime per isolate
- FAISS index build time
- Nearest-neighbor query latency
- Index storage size
- Top-5 antibiotic label agreement
- UMAP visualizations of embedding representations

Outputs support comparisons between embedding models and FAISS indexing approaches.

## Repository Structure

```text
.
├── Data/
│   ├── IR_Variable/
│   └── cryptic_targets_all.json
├── Scripts/
│   ├── model-1-nucleotide-transformer-500m-human-ref/
│   │   └── embed_isolates.py
│   ├── model-2-dnabert-1-6mer/
│   │   └── embed_isolates.py
│   ├── faiss_indexes/
│   │   ├── flat_build_search.py
│   │   ├── hnsw_build_search.py
│   │   └── ivf_build_search.py
│   ├── analysis/
│   │   ├── collect_results.py
│   │   └── generate_figures.py
│   └── run_full_pipeline.py
├── test.py
├── requirements.txt
└── README.md
```

Real genomic datasets, generated embeddings, indexes, and most benchmarking outputs are maintained locally rather than committed to GitHub.

## Installation

Clone the repository:

```bash
git clone https://github.com/jbetancourt589/TB-Embedding-Retrieval-Benchmark.git
cd TB-Embedding-Retrieval-Benchmark
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

For GPU acceleration, install a PyTorch version compatible with your CUDA environment.

## Data Requirements

The embedding scripts expect:

```text
Data/IR_Variable/
Data/cryptic_targets_all.json
```

### FASTA inputs

- Files must end in `.fasta`.
- Isolate identifiers are extracted from filenames by removing `_IR_Genes.fasta`.
- Gene-centered sequence records follow the BEFORE/GENE/AFTER organization.

### Antibiotic susceptibility labels

`cryptic_targets_all.json` contains susceptibility annotations associated with isolate identifiers.

Identifiers must match those extracted from the FASTA filenames.

Refer to `Data/README.md` for the expected input structure.

**The underlying genomic datasets are not included in the public repository.**

## Running the Pipeline

### Full benchmark

```bash
python Scripts/run_full_pipeline.py
```

### Smaller test

```bash
python Scripts/run_full_pipeline.py --max-fasta-files 10
```

### Reuse existing embeddings

```bash
python Scripts/run_full_pipeline.py --skip-embeddings
```

### Model-loading smoke test

```bash
python test.py
```

Individual stages can also be executed independently.

**Generate embeddings:**

```bash
python Scripts/model-1-nucleotide-transformer-500m-human-ref/embed_isolates.py
python Scripts/model-2-dnabert-1-6mer/embed_isolates.py
```

**Build and search FAISS indexes:**

```bash
python Scripts/faiss_indexes/flat_build_search.py
python Scripts/faiss_indexes/hnsw_build_search.py
python Scripts/faiss_indexes/ivf_build_search.py
```

**Generate benchmark summaries and visualizations:**

```bash
python Scripts/analysis/collect_results.py
python Scripts/analysis/generate_figures.py
```

When running embedding scripts individually, use the `TB_MAX_FASTA_FILES` environment variable to limit processing.

Example in PowerShell:

```powershell
$env:TB_MAX_FASTA_FILES = "10"
python Scripts/model-1-nucleotide-transformer-500m-human-ref/embed_isolates.py
```

## Generated Outputs

### Embeddings

```text
Outputs/<model-name>/
├── isolate_embeddings.npy
├── isolate_metadata.csv
├── readable_embeddings.csv
└── faiss_indexes/
```

`isolate_embeddings.npy` is a two-dimensional NumPy array with shape:

```text
(number_of_isolates, embedding_dimension)
```

`isolate_metadata.csv` records isolate identifiers, filenames, embedded gene-unit counts, and antibiotic susceptibility annotations.

The embedding and metadata files are aligned by row index.

### Analysis

```text
Analysis/
├── embedding_summary.csv
└── faiss_summary.csv
```

### Visualizations

```text
Figures/
├── embedding_total_runtime.png
├── embedding_average_runtime.png
├── faiss_build_time.png
├── faiss_search_time.png
├── faiss_index_size.png
├── faiss_match_rate.png
└── umap_comparison_RIF.png
```

These outputs are generated locally and excluded from Git unless separately approved for public release.

## Research Extensions

Beyond the public benchmarking pipeline, related experiments investigate how different genomic representations affect retrieval quality.

These include:

- Ordered gene representations
- Combined gene and intergenic-region representations
- Resistance-associated gene subsets
- Resistance-associated genes with intergenic context
- Alternative gene-level aggregation strategies

These experiments explore whether biologically informed representation choices improve the usefulness of DNA foundation-model embeddings for antibiotic resistance research.

Not all extensions are implemented in the public repository.

## Limitations

- Embedding similarity does not necessarily imply identical antibiotic resistance phenotypes.
- Resistance-label imbalance may inflate overall neighbor agreement.
- Approximate FAISS indexes can trade retrieval exactness for performance.
- Model and index comparisons depend on the chosen genomic representation and benchmarking configuration.
- The full research datasets and HPC workflows are not publicly included.

## Research Context

This work explores the use of pretrained DNA foundation models and dense vector retrieval for comparative genomic analysis of tuberculosis isolates.

It demonstrates practical applications of machine learning, genomic sequence analysis, vector search, and high-performance computing.

**Intended use:** Research and computational benchmarking only. The pipeline is not validated for clinical diagnosis or treatment decisions.

## Author

**Jonathan Betancourt**

Computer Science — University of Florida

Research interests: Computational Genomics, Bioinformatics, Machine Learning, and Biomedical Data Science

[GitHub](https://github.com/jbetancourt589) | [LinkedIn](https://www.linkedin.com/in/jonathan-a-betancourt)

## License

No license has been selected. Reuse and redistribution permissions are not established by this repository.
