# TB Isolate Embedding and Retrieval Benchmark

**Computational Genomics | DNA Foundation Models | FAISS Vector Search | Antibiotic Resistance Analysis**

## Overview

This project develops a genomic dense-retrieval framework for analyzing *Mycobacterium tuberculosis* (MTB) isolates using pretrained DNA language models and FAISS similarity search.

The pipeline converts genomic FASTA sequences into numerical embeddings, retrieves genetically similar isolates, and evaluates how well genomic similarity corresponds to antibiotic resistance phenotypes.

The research investigates:

- How effectively DNA foundation models represent genomic variation across MTB isolates.
- How FAISS indexing strategies compare in search efficiency, memory usage, and retrieval performance.
- How different genomic representations and aggregation methods influence antibiotic resistance retrieval and classification.

The broader research has evaluated more than 6,000 MTB isolates. The public repository contains the benchmarking implementation, while additional experiments have been conducted separately on the University of Florida's HiPerGator HPC system.

## Technologies

**Language:** Python

**Machine Learning:** PyTorch, Hugging Face Transformers, Nucleotide Transformer 500M, DNABERT-1

**Vector Retrieval:** FAISS Flat, HNSW, IVF

**Data Analysis:** NumPy, pandas, Matplotlib, UMAP

**Genomics:** FASTA processing, DNA sequence embeddings, antibiotic resistance phenotype analysis

**Research Computing:** CUDA, Linux, Conda, SLURM, UF HiPerGator

## Research Pipeline

```text
MTB Genomic FASTA Sequences
             |
             v
    Sequence Preprocessing
             |
             v
     DNA Foundation Models
  (NT 500M / DNABERT-1)
             |
             v
     Gene-Level Embeddings
             |
             v
   Isolate-Level Aggregation
             |
             v
      FAISS Vector Index
   (Flat / HNSW / IVF)
             |
             v
   Top-K Nearest Neighbors
             |
             v
 Antibiotic Phenotype Comparison
             |
             v
 Retrieval & Classification Metrics
```

Nearest neighbors are selected using genomic embedding similarity. Antibiotic susceptibility labels are used afterward for evaluation rather than for computing similarity.

## Research Components

### 1. Genomic Embedding Generation

The public pipeline uses two pretrained DNA language models:

- `InstaDeepAI/nucleotide-transformer-500m-human-ref`
- `zhihan1996/DNA_bert_6`

Both models are used as feature extractors without additional fine-tuning.

The embedding scripts process gene-centered genomic regions organized as:

```text
BEFORE region
GENE region
AFTER region
```

These sequences provide coding and neighboring genomic context that may be relevant to antibiotic resistance.

The public implementation combines valid gene-region sequences, generates embeddings, and averages gene-level representations to produce a fixed-length embedding for each isolate.

### 2. FAISS Nearest-Neighbor Retrieval

The pipeline benchmarks multiple FAISS indexes:

| Index | Search Method |
|---|---|
| Flat | Exact nearest-neighbor search |
| HNSW | Graph-based approximate search |
| IVF 1024/32 | 1,024 clusters, 32 search probes |
| IVF 2048/64 | 2,048 clusters, 64 search probes |
| IVF 4096/64 | 4,096 clusters, 64 search probes |

The benchmark measures index construction time, query latency, index storage size, and antibiotic label agreement among retrieved neighbors.

### 3. Genomic Representation Experiments

Additional research conducted on HiPerGator evaluates how the organization and aggregation of gene-level information affect retrieval performance.

Representation strategies include:

1. **Original chunked and averaged embeddings:** Aggregate embeddings across genomic sequence regions.
2. **Complete-case concatenation:** Concatenate gene representations for isolates with the required complete gene information.
3. **Padded/placeholder concatenation:** Preserve a consistent representation structure when some gene information is unavailable.
4. **Gene pooling:** Aggregate gene-level representations into an isolate-level embedding.

Related experiments also investigate ordered genes, intergenic regions, and resistance-associated gene subsets.

These experiments help assess whether preserving gene-level structure improves genomic retrieval and antibiotic resistance classification.

**Note:** These additional representation experiments are part of the broader research workflow and are not necessarily implemented in the public repository.

## Experimental Results

Two complementary evaluations compare genomic representation strategies across nine antibiotics.

### Figure 1 — Retrieval Performance (Match@5)

Match@5 measures the proportion of the five retrieved neighbors whose antibiotic phenotype agrees with the query isolate's phenotype, as reported in the experiment.

![Flat Match@5 by Antibiotic and Representation](Figures/flat_match_at_5.png)

**Key observations:**

- AMI achieved the highest displayed Match@5, reaching **96.4%** with complete-case concatenation.
- RIF Match@5 ranged from 77.3% to 90.7%, depending on the representation.
- EMB demonstrated substantial differences between representations, ranging from 63.6% to 82.9%.
- Representation choice affected retrieval agreement differently across antibiotics.

**Evaluation note:** Complete-case concatenation uses a reduced native cohort in this figure. Its results should not be interpreted as directly controlled improvements over methods evaluated on different isolate populations.

### Figure 2 — Resistant-Class F1 Score

While Match@5 measures nearest-neighbor label agreement, resistant-class F1 evaluates the balance between precision and recall when predicting resistant isolates.

This experiment uses Top-5 majority-vote classification on a matched complete-case cohort.

![Resistant F1 by Antibiotic and Representation](Figures/resistant_f1.png)

**Key observations:**

- RIF achieved a resistant-class F1 of **89.2%** using placeholder or complete-case concatenation.
- For INH, both concatenation methods achieved 89.0% resistant-class F1.
- For EMB, concatenation improved resistant-class F1 from 71.9% for the original representation to 77.0%.
- No single representation consistently produced the best results across every antibiotic.

**Evaluation note:** These results come from a matched complete-case cohort and should be distinguished from the cohort-specific Match@5 comparisons.

### Interpretation

The experiments illustrate the importance of evaluating genomic retrieval beyond overall nearest-neighbor agreement.

Match@5 describes phenotype agreement among retrieved neighbors, while resistant-class F1 provides a more targeted assessment of predictions for resistant isolates.

The findings suggest that genomic representation and aggregation choices can meaningfully affect retrieval and classification behavior, with differences depending on the antibiotic.

These results are research benchmarks, not validated clinical predictions.

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
├── Figures/
│   ├── flat_match_at_5.png
│   └── resistant_f1.png
├── test.py
├── requirements.txt
└── README.md
```

The `Data/` paths describe required local inputs. Genomic datasets, generated embeddings, and large benchmark outputs are excluded from public version control.

## Installation

Clone the repository:

```bash
git clone https://github.com/jbetancourt589/TB-Embedding-Retrieval-Benchmark.git
cd TB-Embedding-Retrieval-Benchmark
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the appropriate CUDA-compatible PyTorch version.

## Input Data

The scripts expect:

```text
Data/IR_Variable/
Data/cryptic_targets_all.json
```

The input requirements include:

- Isolate FASTA files ending in `.fasta`.
- Isolate identifiers extracted by removing `_IR_Genes.fasta` from filenames.
- Gene-centered BEFORE/GENE/AFTER sequence groups.
- Antibiotic susceptibility labels indexed by matching isolate identifiers.

See `Data/README.md` for local data setup information.

**The underlying MTB datasets are not included in this public repository.**

## Running the Pipeline

### Full Benchmark

```bash
python Scripts/run_full_pipeline.py
```

### Smaller Test Run

```bash
python Scripts/run_full_pipeline.py --max-fasta-files 10
```

### Reuse Existing Embeddings

```bash
python Scripts/run_full_pipeline.py --skip-embeddings
```

### Model-Loading Smoke Test

```bash
python test.py
```

### Run Individual Pipeline Stages

Generate isolate embeddings:

```bash
python Scripts/model-1-nucleotide-transformer-500m-human-ref/embed_isolates.py
python Scripts/model-2-dnabert-1-6mer/embed_isolates.py
```

Build and search FAISS indexes:

```bash
python Scripts/faiss_indexes/flat_build_search.py
python Scripts/faiss_indexes/hnsw_build_search.py
python Scripts/faiss_indexes/ivf_build_search.py
```

Collect results and generate figures:

```bash
python Scripts/analysis/collect_results.py
python Scripts/analysis/generate_figures.py
```

For smaller individual embedding runs, set the `TB_MAX_FASTA_FILES` environment variable.

PowerShell example:

```powershell
$env:TB_MAX_FASTA_FILES = "10"
python Scripts/model-1-nucleotide-transformer-500m-human-ref/embed_isolates.py
```

## Generated Outputs

The pipeline generates embedding arrays, isolate metadata, FAISS indexes, and benchmark summaries.

```text
Outputs/<model-name>/
├── isolate_embeddings.npy
├── isolate_metadata.csv
├── readable_embeddings.csv
└── faiss_indexes/
```

`isolate_embeddings.npy` contains an array with shape:

```text
(number_of_isolates, embedding_dimension)
```

Each embedding row corresponds to the same row in `isolate_metadata.csv`.

Metadata includes isolate identifiers, source filenames, embedded gene-unit counts, and available antibiotic susceptibility labels.

Benchmark summaries are generated under:

```text
Analysis/
├── embedding_summary.csv
└── faiss_summary.csv
```

Additional runtime, search, storage, and visualization outputs may be generated locally.

## Limitations

- Genomic embedding similarity does not guarantee identical antibiotic resistance.
- Class imbalance can affect label agreement and classification performance.
- Some representations are evaluated using different isolate cohorts.
- Approximate nearest-neighbor indexes may trade exactness for retrieval efficiency.
- The public pipeline does not include every newer research experiment.
- Full experiments require external genomic data and substantial computational resources.

## Research Context

This project investigates how pretrained DNA language models and vector retrieval methods can support comparative genomic analysis of tuberculosis isolates.

It combines machine learning, bioinformatics, large-scale data processing, and high-performance computing to study antibiotic resistance-related genomic patterns.

**This framework is intended for research and computational benchmarking only and is not validated for clinical diagnosis or treatment selection.**

## Author

**Jonathan Betancourt**

Computer Science — University of Florida

Research interests: Computational Genomics, Bioinformatics, Machine Learning, and Biomedical Data Science

[GitHub](https://github.com/jbetancourt589) | [LinkedIn](https://www.linkedin.com/in/jonathan-a-betancourt)

## License

No license has been selected. Reuse and redistribution permissions have not been granted through a repository license.
