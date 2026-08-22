# ADHD200 local dataset

This directory contains the normalized ADHD200 subset currently available in
this workspace.

## Main data

- `adhd200_preprocessed_phenotypics.tsv`: phenotype rows aligned to local AAL data.
- `AAL_TCs_filtfix/{site}/{subject_id}/`: normalized AAL time-series directories.
- `clean_dataset.py`: optional validator/normalizer for newly added data.

Subject directory names use normalized numeric IDs, so `0016058` is stored as
`16058`. The current local data contains 768 valid subjects; the shortest
available AAL sequence has 72 time points.

## Archive

`archive/` stores the original consolidated phenotype and site-level phenotype
releases. These files are retained for provenance and are not used directly by
the Graph-BEC loader.

## Refresh after adding data

Run:

```bash
python dataset/ADHD200/clean_dataset.py
```

Then verify the cleaned phenotype and AAL directories before running Graph-BEC.
