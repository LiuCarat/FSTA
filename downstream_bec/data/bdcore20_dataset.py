import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_BY_GROUP = {"HC": 0, "BD": 1}


class BDCore20Dataset(Dataset):
    def __init__(self, data_root="dataset/BDCore20", skiprows=1):
        self.data_root = Path(data_root)
        self.skiprows = skiprows
        self.subject_ids = []
        self.labels = []
        subject_data = []
        expected_shape = None

        for group in ("HC", "BD"):
            group_dir = self.data_root / group
            subject_files = sorted(group_dir.glob("*.txt"))
            if not subject_files:
                raise ValueError(f"No subject files found in {group_dir}")

            for subject_file in subject_files:
                data = np.loadtxt(
                    subject_file,
                    skiprows=self.skiprows,
                    delimiter="\t",
                    dtype=np.float32,
                )
                if data.ndim != 2:
                    raise ValueError(
                        f"Expected 2D data in {subject_file}, got {data.shape}"
                    )
                if expected_shape is None:
                    expected_shape = data.shape
                elif data.shape != expected_shape:
                    raise ValueError(
                        f"Inconsistent shape in {subject_file}: "
                        f"{data.shape} != {expected_shape}"
                    )

                subject_data.append(data)
                self.subject_ids.append(subject_file.stem)
                self.labels.append(LABEL_BY_GROUP[group])

        self.data = torch.from_numpy(np.stack(subject_data, axis=0))
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.time_num = self.data.shape[1]
        self.nodes_num = self.data.shape[2]

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, index):
        return self.data[index], self.labels[index], index


def load_roi_names(labels_path="dataset/BDCore20/atlas/roi_labels.tsv"):
    labels_path = Path(labels_path)
    with labels_path.open(newline="") as labels_file:
        reader = csv.DictReader(labels_file, delimiter="\t")
        rows = sorted(reader, key=lambda row: int(row["label_id"]))
    roi_names = [row["roi_name"] for row in rows]
    if not roi_names:
        raise ValueError(f"No ROI names found in {labels_path}")
    return roi_names
