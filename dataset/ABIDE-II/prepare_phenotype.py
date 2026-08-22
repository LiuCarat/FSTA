"""Convert ABIDE-II phenotype into an ABIDE-I-like canonical CSV."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ALIASES = {
    "site": ("SITE_ID", "site_id", "site"),
    "subject": ("participant_id", "SUB_ID", "subject_id", "subject"),
    "dx": ("dx_group", "DX_GROUP", "diagnosis", "DX"),
    "age": ("age_at_scan", "AGE_AT_SCAN", "age_at_scan "),
    "sex": ("sex", "SEX", "gender", "Gender"),
    "handedness_category": ("handedness_category", "HANDEDNESS_CATEGORY"),
    "handedness_score": ("handedness_scores", "handedness_score", "HANDEDNESS_SCORES"),
    "fiq": ("fiq", "FIQ", "Full4 IQ"),
    "viq": ("viq", "VIQ", "Verbal IQ"),
    "piq": ("piq", "PIQ", "Performance IQ"),
}


def find_column(frame, aliases, required=False):
    columns = {str(column).strip().lower(): column for column in frame.columns}
    for alias in aliases:
        found = columns.get(alias.strip().lower())
        if found is not None:
            return found
    if required:
        raise ValueError(f"Missing phenotype field; tried {aliases}")
    return None


def value(frame, column, default=np.nan):
    return frame[column] if column else pd.Series(default, index=frame.index)


def normalize_subject(series):
    values = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return values.str.removeprefix("sub-").map(lambda item: f"sub-{item}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("ABIDEII/phenotype/ABIDEII_phenotype_merged.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/ABIDE-II/Phenotypic_Processing.csv"))
    parser.add_argument("--eligible", type=Path, default=Path("ABIDEII/phenotype/phenotype_eligible_subjects.csv"))
    parser.add_argument("--require-fiq-piq", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    columns = {name: find_column(frame, aliases, required=name in {"site", "subject", "dx", "age", "sex"}) for name, aliases in ALIASES.items()}
    output = pd.DataFrame(index=frame.index)
    output["SITE_ID"] = value(frame, columns["site"]).astype(str).str.replace(r"^ABIDEII-", "", regex=True)
    output["SUB_ID"] = normalize_subject(value(frame, columns["subject"]))
    output["FILE_ID"] = "ABIDEII-" + output["SITE_ID"] + "/" + output["SUB_ID"]
    output["DX_GROUP"] = pd.to_numeric(value(frame, columns["dx"]), errors="coerce")
    output["AGE_AT_SCAN"] = pd.to_numeric(value(frame, columns["age"]), errors="coerce")
    output["SEX"] = pd.to_numeric(value(frame, columns["sex"]), errors="coerce")
    output["HANDEDNESS_CATEGORY"] = value(frame, columns["handedness_category"])
    output["HANDEDNESS_SCORES"] = pd.to_numeric(value(frame, columns["handedness_score"]), errors="coerce")
    output["FIQ"] = pd.to_numeric(value(frame, columns["fiq"]), errors="coerce")
    output["VIQ"] = pd.to_numeric(value(frame, columns["viq"]), errors="coerce")
    output["PIQ"] = pd.to_numeric(value(frame, columns["piq"]), errors="coerce")
    output["DATASET"] = "ABIDE-II"
    output["SOURCE_ROW"] = np.arange(len(output))

    if args.require_fiq_piq:
        output = output[output["FIQ"].notna() & output["PIQ"].notna()].copy()
    elif args.eligible.exists():
        eligible = pd.read_csv(args.eligible)
        eligible["subject"] = normalize_subject(eligible["subject"])
        eligible_sites = eligible["SITE_ID"].astype(str).str.replace(r"^ABIDEII-", "", regex=True)
        keys = set(zip(eligible_sites, eligible["subject"]))
        mask = [key in keys for key in zip(output["SITE_ID"], output["SUB_ID"])]
        output = output.loc[mask].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(output)} phenotype rows: {args.output.resolve()}")
    print("Fields: " + ", ".join(output.columns))


if __name__ == "__main__":
    main()
