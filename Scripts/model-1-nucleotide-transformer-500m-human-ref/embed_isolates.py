
#1.LOADS TRANSFORMER
#2.READS 'N' FASTA FILES
#3.COMBINES: BEFORE + GENE + AFTER
#4.CREATES EMBEDDING ISOLATE


import os
import json
import time
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
MODEL_DIR_NAME = "model-1-nucleotide-transformer-500m-human-ref"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_DIR = os.path.join(PROJECT_DIR, "Data")
FASTA_DIR = os.path.join(DATA_DIR, "IR_Variable") #contains all TB isolate FASTA files
TARGETS_JSON = os.path.join(DATA_DIR, "cryptic_targets_all.json") #contains the labels for every isolate

OUTPUT_DIR = os.path.join(PROJECT_DIR, "Outputs", MODEL_DIR_NAME)

#contains the vectors that will be searched through later using FAISS
EMBEDDINGS_OUT = os.path.join(OUTPUT_DIR, "isolate_embeddings.npy")

#shows which isolate each vector corresponds to
METADATA_OUT = os.path.join(OUTPUT_DIR, "isolate_metadata.csv")

READABLE_EMBEDDINGS_OUT = os.path.join(OUTPUT_DIR, "readable_embeddings.csv")

MAX_LENGTH = 1000


def get_num_fasta_files():
    value = os.environ.get("TB_MAX_FASTA_FILES")

    if value is None:
        return None

    value = value.strip()

    if value.lower() in ("", "all", "none"):
        return None

    number = int(value)

    if number <= 0:
        raise ValueError("TB_MAX_FASTA_FILES must be a positive integer, 'all', or 'none'.")

    return number


NUM_FASTA_FILES = get_num_fasta_files()

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("Loading model...")
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print("Device:", device)

with open(TARGETS_JSON, "r") as f:
    targets = json.load(f)


def clean_sequence(seq):
    seq = str(seq).upper()
    return "".join(base for base in seq if base in "ACGTN")


@torch.no_grad()
def embed_sequence(sequence):
    sequence = clean_sequence(sequence)

    if len(sequence) < 50:
        return None

    sequence = sequence[:MAX_LENGTH]

    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)

    last_hidden_state = outputs.hidden_states[-1]
    attention_mask = inputs["attention_mask"].unsqueeze(-1)

    embedding = (last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

    return embedding.squeeze().cpu().numpy()


def embed_isolate(fasta_path):
    unit_embeddings = []
    records = list(SeqIO.parse(fasta_path, "fasta"))

    i = 0
    while i < len(records) - 2:
        before_record = records[i]
        gene_record = records[i + 1]
        after_record = records[i + 2]

        before_header = before_record.description
        after_header = after_record.description

        if "BEFORE" in before_header and "AFTER" in after_header:
            combined_sequence = (
                str(before_record.seq).upper()
                + str(gene_record.seq).upper()
                + str(after_record.seq).upper()
            )

            emb = embed_sequence(combined_sequence)

            if emb is not None:
                unit_embeddings.append(emb)

            i += 3
        else:
            i += 1

    if len(unit_embeddings) == 0:
        return None, 0

    isolate_embedding = np.mean(unit_embeddings, axis=0)

    return isolate_embedding, len(unit_embeddings)


fasta_files = [
    f for f in os.listdir(FASTA_DIR)
    if f.endswith(".fasta")
]

if NUM_FASTA_FILES is not None:
    fasta_files = fasta_files[:NUM_FASTA_FILES]

print("FASTA files found:", len(fasta_files))

all_embeddings = []
metadata = []
isolate_times = []
skipped_count = 0

total_start_time = time.time()

for filename in tqdm(fasta_files):
    isolate_start_time = time.time()

    isolate_id = filename.replace("_IR_Genes.fasta", "")
    fasta_path = os.path.join(FASTA_DIR, filename)

    if isolate_id not in targets:
        print(f"Skipping {isolate_id}: no target labels found")
        skipped_count += 1
        continue

    isolate_embedding, gene_count = embed_isolate(fasta_path)

    if isolate_embedding is None:
        print(f"Skipping {isolate_id}: no valid gene embeddings")
        skipped_count += 1
        continue

    isolate_runtime = time.time() - isolate_start_time
    isolate_times.append(isolate_runtime)

    all_embeddings.append(isolate_embedding)

    row = {
        "isolate_id": isolate_id,
        "filename": filename,
        "num_gene_units_embedded": gene_count,
        "runtime_seconds": round(isolate_runtime, 3)
    }

    row.update(targets[isolate_id])
    metadata.append(row)

total_runtime = time.time() - total_start_time

if len(all_embeddings) == 0:
    raise ValueError("No embeddings were created. Check FASTA files and target labels.")

embeddings_array = np.vstack(all_embeddings)
metadata_df = pd.DataFrame(metadata)

np.save(EMBEDDINGS_OUT, embeddings_array)
metadata_df.to_csv(METADATA_OUT, index=False)

embedding_columns = [f"dim_{i}" for i in range(embeddings_array.shape[1])]
embeddings_df = pd.DataFrame(embeddings_array, columns=embedding_columns)

readable_df = pd.concat(
    [metadata_df[["isolate_id", "filename", "num_gene_units_embedded"]], embeddings_df],
    axis=1
)

readable_df.to_csv(READABLE_EMBEDDINGS_OUT, index=False)

average_time = sum(isolate_times) / len(isolate_times)

print("Done.")
print("Embeddings saved to:", EMBEDDINGS_OUT)
print("Readable embeddings CSV saved to:", READABLE_EMBEDDINGS_OUT)
print("Metadata saved to:", METADATA_OUT)
print("Embeddings shape:", embeddings_array.shape)
print("Metadata shape:", metadata_df.shape)

print("\nRuntime Metrics")
print("Total runtime seconds:", round(total_runtime, 3))
print("Total runtime minutes:", round(total_runtime / 60, 3))
print("Average time per isolate seconds:", round(average_time, 3))
print("Successfully embedded isolates:", len(all_embeddings))
print("Skipped isolates:", skipped_count)
