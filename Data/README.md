# Data Setup

Place the input isolate data here before running the pipeline.

Required layout:

```text
Data/
+-- IR_Variable/
|   +-- <isolate-id>_IR_Genes.fasta
+-- cryptic_targets_all.json
```

The FASTA files and target-label JSON are intentionally ignored by Git because they may be large or sensitive. The scripts expect isolate IDs to match between each FASTA filename, after removing `_IR_Genes.fasta`, and the keys in `cryptic_targets_all.json`.
