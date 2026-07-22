# TB Isolate Embedding and Retrival Benchmark

A benchmarking framework for genomic dense retrieval of Mycobacterium tuberculosis (MTB) isolates. This project compares multiple genomic embedding models and FAISS index types by generating isolate-level embeddings, retrieving nearest neighbors, and evaluating retrieval performance using antibiotic resistance labels
## Project Status

This project currently:

- Loads tuberculosis isolate FASTA files from `Data/IR_Variable`
- Loads antibiotic susceptibility labels from `Data/cryptic_targets_all.json`
- Generates isolate-level embeddings with multiple pretrained DNA language models
- Compares embedding runtime and output size between models
- Builds FAISS nearest-neighbor indexes from each model's isolate embeddings
- Compares index build time, search time, index size, and top-5 label-match behavior across Flat, HNSW, and IVF indexes

## Pipeline

```text
Data/IR_Variable + Data/cryptic_targets_all.json
      |
      v
Generate embeddings for each model
      |
      v
Save Outputs/<model>/isolate_embeddings.npy and metadata CSVs
      |
      v
Build and search FAISS indexes for each model
Flat, HNSW, IVF 1024/32, IVF 2048/64, IVF 4096/64
      |
      v
Collect benchmark summaries in Analysis/
      |
      v
Generate comparison figures in Figures/
```

FAISS searches use only embedding similarity. Antibiotic labels are used after search to measure whether each isolate's top-5 neighbors share the same resistance or susceptibility phenotype.

## Repository Structure

```text
.
+-- Data/
|   +-- IR_Variable/              # Input FASTA files for isolates
|   +-- cryptic_targets_all.json  # Antibiotic susceptibility labels by isolate
+-- Scripts/
|   +-- model-1-nucleotide-transformer-500m-human-ref/
|   |   +-- embed_isolates.py      # Nucleotide Transformer embedding pipeline
|   +-- model-2-dnabert-1-6mer/
|   |   +-- embed_isolates.py      # DNABERT-1 6-mer embedding pipeline
|   +-- faiss_indexes/
|   |   +-- flat_build_search.py   # Exact FAISS index build/search
|   |   +-- hnsw_build_search.py   # HNSW FAISS index build/search
|   |   +-- ivf_build_search.py    # IVF FAISS index build/search
|   +-- analysis/
|       +-- collect_results.py     # Summarize embedding and FAISS outputs
|       +-- generate_figures.py    # Generate plots from summary files
+-- test.py                       # Smoke test for model loading and one FASTA sequence
+-- requirements.txt
+-- README.md
```

Generated files are written locally and are ignored by Git:

```text
Outputs/
+-- <model-name>/
|   +-- isolate_embeddings.npy
|   +-- isolate_metadata.csv
|   +-- readable_embeddings.csv
|   +-- faiss_indexes/
Analysis/
+-- embedding_summary.csv
+-- faiss_summary.csv
Figures/
+-- embedding_total_runtime.png
+-- embedding_average_runtime.png
+-- faiss_build_time.png
+-- faiss_search_time.png
+-- faiss_index_size.png
+-- faiss_match_rate.png
+-- umap_comparison_RIF.png
```

## Requirements

This project uses Python and the Hugging Face Transformers ecosystem.

Install the required packages:

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch build that matches your CUDA version from the official PyTorch installation instructions.

## Embedding Models

This project currently compares these pretrained models:

```text
InstaDeepAI/nucleotide-transformer-500m-human-ref
zhihan1996/DNA_bert_6
```

No additional fine-tuning is performed. The models are used as feature extractors to generate fixed-length isolate embeddings.

## Data Expectations

The embedding scripts expect:

- FASTA files in `Data/IR_Variable`
- FASTA filenames ending in `.fasta`
- Isolate IDs derived from filenames by removing `_IR_Genes.fasta`
- A target-label JSON file at `Data/cryptic_targets_all.json`
- Matching isolate IDs between the FASTA filenames and the target-label JSON

See `Data/README.md` for the expected local data layout. The real data files are ignored by Git.

The FASTA parsing logic looks for groups of three records:

```text
BEFORE region
gene region
AFTER region
```

When a valid group is found, the script concatenates the BEFORE, gene, and AFTER sequences, embeds the combined sequence, and later averages all valid gene-region embeddings for that isolate.

The BEFORE and AFTER regions are included because regulatory mutations outside coding regions may contribute to antibiotic resistance and gene expression changes.

Each isolate contains many gene units. The embedding generated for each gene unit is averaged to produce a single fixed-length embedding representing the overall genomic characteristics of the isolate.

## Usage

Run the full pipeline:

```bash
python Scripts/run_full_pipeline.py
```

Run a smaller test pipeline:

```bash
python Scripts/run_full_pipeline.py --max-fasta-files 10
```

Reuse existing embeddings and only rebuild FAISS outputs, summaries, and figures:

```bash
python Scripts/run_full_pipeline.py --skip-embeddings
```

Run a quick model-loading smoke test:

```bash
python test.py
```

You can also run each stage manually.

Generate isolate embeddings:

```bash
python Scripts/model-1-nucleotide-transformer-500m-human-ref/embed_isolates.py
python Scripts/model-2-dnabert-1-6mer/embed_isolates.py
```

Build FAISS indexes and run top-5 nearest-neighbor searches:

```bash
python Scripts/faiss_indexes/flat_build_search.py
python Scripts/faiss_indexes/hnsw_build_search.py
python Scripts/faiss_indexes/ivf_build_search.py
```

Collect summary metrics and generate figures:

```bash
python Scripts/analysis/collect_results.py
python Scripts/analysis/generate_figures.py
```

When running individual embedding scripts, set `TB_MAX_FASTA_FILES` for a smaller test run. Leave it unset to process all isolates.

PowerShell:

```powershell
$env:TB_MAX_FASTA_FILES = "10"
python Scripts/model-1-nucleotide-transformer-500m-human-ref/embed_isolates.py
```

Clear the limit with `Remove-Item Env:TB_MAX_FASTA_FILES`.

## Outputs

`isolate_embeddings.npy` contains a 2D NumPy array:

```text
number_of_isolates x embedding_dimension
```

Each row represents one isolate embedding, and each column corresponds to one embedding dimension produced by the pretrained model.

`isolate_metadata.csv` contains:

- `isolate_id`
- `filename`
- `num_gene_units_embedded`
- Antibiotic susceptibility labels from `cryptic_targets_all.json`

The CSV provides the mapping between each embedding and its corresponding isolate, along with antibiotic susceptibility labels and embedding statistics. It acts as the lookup table for interpreting the rows in `isolate_embeddings.npy`.

These two files are meant to stay aligned by row index. Row `i` in `isolate_embeddings.npy` corresponds to row `i` in `isolate_metadata.csv`.

## Benchmark Workflow

The benchmark workflow is:

1. Generate embeddings for the same tuberculosis isolates with each model.
2. Build and search the same FAISS index families for each model output.
3. Compare embedding runtime, FAISS build time, FAISS search time, index size, and top-5 label-match rates.
4. Use the generated CSV summaries and figures to compare model/index combinations.

This should be treated as a research and decision-support workflow, not a standalone clinical diagnostic system.

## Privacy and Data Notes

Before uploading this project to GitHub, confirm that the included data files do not contain protected health information or any data that should remain private. If the FASTA files, labels, or generated outputs are sensitive, exclude them from the public repository and provide instructions for placing the data locally.

## License

No license has been selected yet. Add a license before publishing if you want others to know how they may use, modify, or distribute this code.
