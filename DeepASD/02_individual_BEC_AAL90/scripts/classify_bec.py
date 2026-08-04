from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
    parser.add_argument("--seed", type=int, default=42)
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

        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=args.seed,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        model.fit(vectorize(train["bec"]), train["label"])
        probability = model.predict_proba(vectorize(test["bec"]))[:, 1]
        prediction = (probability >= 0.5).astype(int)

        rows.append(
            {
                "fold": fold,
                "balanced_accuracy": balanced_accuracy_score(
                    test["label"],
                    prediction,
                ),
                "auc": roc_auc_score(test["label"], probability),
                "weighted_precision": precision_score(
                    test["label"],
                    prediction,
                    average="weighted",
                    zero_division=0,
                ),
                "weighted_recall": recall_score(
                    test["label"],
                    prediction,
                    average="weighted",
                    zero_division=0,
                ),
                "weighted_f1": f1_score(
                    test["label"],
                    prediction,
                    average="weighted",
                    zero_division=0,
                ),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_csv, index=False)
    for name in [
        "balanced_accuracy",
        "auc",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]:
        print(
            f"{name}={frame[name].mean() * 100:.2f}"
            f"±{frame[name].std(ddof=0) * 100:.2f}"
        )


if __name__ == "__main__":
    main()
