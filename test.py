from pathlib import Path

import torch
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-500m-human-ref"

PROJECT_DIR = Path(__file__).resolve().parent
FASTA_DIR = PROJECT_DIR / "Data" / "IR_Variable"

if not FASTA_DIR.exists():
    raise FileNotFoundError(f"FASTA directory not found: {FASTA_DIR}")

fasta_files = [
    path
    for path in sorted(FASTA_DIR.iterdir())
    if path.name.endswith(".fasta")
]

if not fasta_files:
    raise FileNotFoundError(f"No .fasta files found in: {FASTA_DIR}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("Loading model...")
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print("Device:", device)

print("Number of FASTA files found:", len(fasta_files))

FASTA_FILE = fasta_files[0]
print("Testing file:", FASTA_FILE)

record = next(SeqIO.parse(FASTA_FILE, "fasta"))
sequence = str(record.seq).upper()

print("First gene ID:", record.id)
print("Gene length:", len(sequence))

sequence = sequence[:1000]

inputs = tokenizer(
    sequence,
    return_tensors="pt",
    truncation=True,
    max_length=1000
)

inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

last_hidden_state = outputs.hidden_states[-1]

attention_mask = inputs["attention_mask"].unsqueeze(-1)
embedding = (last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

print("Embedding shape:", embedding.shape)
print("Success!")
