# LLMTB

LLMTB is an early-stage tuberculosis genomics project for creating isolate-level embeddings from FASTA sequences. The long-term goal is to use these embeddings in a vector database so a new tuberculosis isolate can be compared against past isolates with similar genomes.

Each isolate is paired with antibiotic susceptibility labels. Once the embeddings are indexed, nearest-neighbor search can support exploratory resistance analysis: for example, if the five closest historical isolates are resistant to a drug, the new isolate may be more likely to behave similarly. This repository currently generates isolate embeddings, builds FAISS indexes, retrieves nearest neighbors, and evaluates whether nearby embeddings share antibiotic resistance phenotypes.

## Project Status

This project currently:

- Loads tuberculosis isolate FASTA files from `Data/IR_Variable`
- Loads antibiotic susceptibility labels from `Data/cryptic_targets_all.json`
- Embeds each gene unit, defined as `BEFORE region + gene + AFTER region`, using `InstaDeepAI/nucleotide-transformer-500m-human-ref`
- Averages gene-unit embeddings into one fixed-length embedding per isolate
- Saves embeddings as a NumPy array and isolate metadata as a CSV file
- Builds FAISS nearest-neighbor indexes over isolate embeddings
- Compares each isolate's top 5 neighbors against antibiotic susceptibility labels

## Pipeline

```text
FASTA isolate files
      |
      v
Extract gene units
(BEFORE region + gene region + AFTER region)
      |
      v
Nucleotide Transformer
InstaDeepAI/nucleotide-transformer-500m-human-ref
      |
      v
Gene-unit embeddings
      |
      v
Average all gene-unit embeddings for each isolate
      |
      v
One fixed-length isolate embedding
      |
      v
isolate_embeddings.npy
      |
      +----------------------------+
      |                            |
      v                            v
isolate_metadata.csv         readable_embeddings.csv
(isolate ID + labels)        (metadata + embedding columns)
      |
      v
Build FAISS indexes
flat, hnsw, ivf_1024_32, ivf_2048_64, ivf_4096_64
      |
      v
Search each isolate against the index
      |
      v
Remove the isolate itself and keep the top 5 nearest neighbors
      |
      v
Compare query antibiotic labels with neighbor antibiotic labels
AMI, BDQ, CFZ, DLM, EMB, ETH, INH, KAN, LEV, LZD, MXF, RIF, RFB
      |
      v
Calculate per-antibiotic top-5 match scores
      |
      v
Compare index behavior by match score, runtime, size, and neighbor overlap
```

The FAISS index does not use antibiotic labels to choose neighbors. It only uses embedding similarity. The antibiotic labels are used after nearest-neighbor search to evaluate whether nearby isolates share the same resistance or susceptibility phenotype.

For each antibiotic, the match score is:

```text
number of valid query-neighbor pairs with the same antibiotic label
/
number of valid query-neighbor pairs for that antibiotic
```

With 6,150 isolates and `top_k = 5`, each antibiotic can have up to 30,750 query-neighbor comparisons. Missing labels are skipped, so the valid comparison count can be lower.

The index-level match score used for comparison is the average of the per-antibiotic match scores:

```text
(AMI + BDQ + CFZ + DLM + EMB + ETH + INH + KAN + LEV + LZD + MXF + RIF + RFB) / 13
```

This index-level average is useful for comparing search methods, but the per-antibiotic scores are more biologically meaningful. Very high scores for rare-resistance drugs can happen because most isolates share the susceptible label, so those scores should be interpreted alongside resistance prevalence or a random same-label baseline.

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
+-- *.png
```

## Requirements

This project uses Python and the Hugging Face Transformers ecosystem.

Install the required packages:

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch build that matches your CUDA version from the official PyTorch installation instructions.

## Embedding Model

This project currently uses the pretrained model:

```text
InstaDeepAI/nucleotide-transformer-500m-human-ref
```

No additional fine-tuning is currently performed. The model is used as a feature extractor to generate fixed-length genomic embeddings.

## Data Expectations

The embedding script expects:

- FASTA files in `Data/IR_Variable`
- FASTA filenames ending in `.fasta`
- Isolate IDs derived from filenames by removing `_IR_Genes.fasta`
- A target-label JSON file at `Data/cryptic_targets_all.json`
- Matching isolate IDs between the FASTA filenames and the target-label JSON

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

Run a quick smoke test:

```bash
python test.py
```

Generate Nucleotide Transformer isolate embeddings:

```bash
python Scripts/model-1-nucleotide-transformer-500m-human-ref/embed_isolates.py
```

Generate DNABERT-1 6-mer isolate embeddings:

```bash
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

Each embedding script uses `NUM_FASTA_FILES` to control how many FASTA files are processed:

```python
NUM_FASTA_FILES = None
```

Set this to an integer for a smaller test run, or leave it as `None` to process all isolates.

## Outputs

`isolate_embeddings.npy` contains a 2D NumPy array:

```text
number_of_isolates x embedding_dimension
```

Each row represents one isolate embedding, and each column corresponds to one embedding dimension produced by the pretrained nucleotide transformer.

`isolate_metadata.csv` contains:

- `isolate_id`
- `filename`
- `num_gene_units_embedded`
- Antibiotic susceptibility labels from `cryptic_targets_all.json`

The CSV provides the mapping between each embedding and its corresponding isolate, along with antibiotic susceptibility labels and embedding statistics. It acts as the lookup table for interpreting the rows in `isolate_embeddings.npy`.

These two files are meant to stay aligned by row index. Row `i` in `isolate_embeddings.npy` corresponds to row `i` in `isolate_metadata.csv`.

## Current Search Workflow

The current FAISS workflow is:

1. Generate an embedding for a new tuberculosis isolate.
2. Search a FAISS index for the closest historical isolate embeddings.
3. Retrieve antibiotic susceptibility labels for the nearest isolates.
4. Compare nearest-neighbor labels as supporting evidence for resistance or susceptibility patterns.

This should be treated as a research and decision-support workflow, not a standalone clinical diagnostic system.

## Privacy and Data Notes

Before uploading this project to GitHub, confirm that the included data files do not contain protected health information or any data that should remain private. If the FASTA files, labels, or generated outputs are sensitive, exclude them from the public repository and provide instructions for placing the data locally.

## License

No license has been selected yet. Add a license before publishing if you want others to know how they may use, modify, or distribute this code.
