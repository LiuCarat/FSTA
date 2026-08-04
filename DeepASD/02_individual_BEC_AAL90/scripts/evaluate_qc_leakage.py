from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def vectorize(bec):
    mask = ~np.eye(bec.shape[1], dtype=bool)
    return bec[:, mask]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    rows = []
    for fold in range(1, args.folds + 1):
        fold_dir = Path(args.result_dir) / f"fold_{fold:02d}"
        train = np.load(
            fold_dir / "train_individual_bec.npz",
            allow_pickle=True,
        )
        test = np.load(
            fold_dir / "test_individual_bec.npz",
            allow_pickle=True,
        )

        if train["qc_true_standardized"].shape[1] == 0:
            raise ValueError("QC evaluation requires mode 3.")

        x_train = vectorize(train["bec"])
        x_test = vectorize(test["bec"])
        y_train = train["qc_true_standardized"]
        y_test = test["qc_true_standardized"]

        for index in range(y_train.shape[1]):
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=100.0)),
                ]
            )
            model.fit(x_train, y_train[:, index])
            prediction = model.predict(x_test)
            rows.append(
                {
                    "fold": fold,
                    "qc_index": index,
                    "r2": r2_score(y_test[:, index], prediction),
                }
            )

    pd.DataFrame(rows).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
