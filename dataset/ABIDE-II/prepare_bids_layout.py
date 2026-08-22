"""Create a single BIDS view from the site-separated ABIDE-II download."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("ABIDEII/raw"))
    parser.add_argument("--bids-root", type=Path, default=Path("ABIDEII/bids"))
    parser.add_argument("--copy", action="store_true", help="copy instead of symlink")
    args = parser.parse_args()

    args.bids_root.mkdir(parents=True, exist_ok=True)
    subjects = []
    for site_root in sorted(args.raw_root.glob("ABIDEII-*")):
        if not site_root.is_dir():
            continue
        site = site_root.name.removeprefix("ABIDEII-")
        for subject_root in sorted(site_root.glob("sub-*")):
            if not subject_root.is_dir():
                continue
            subject = subject_root.name
            destination = args.bids_root / subject
            if destination.exists() or destination.is_symlink():
                if destination.is_symlink() and destination.resolve() == subject_root.resolve():
                    pass
                else:
                    raise FileExistsError(
                        f"Subject collision: {subject} exists for multiple sites"
                    )
            elif args.copy:
                import shutil
                shutil.copytree(subject_root, destination)
            else:
                destination.symlink_to(subject_root.resolve(), target_is_directory=True)
            subjects.append({"site": site, "subject": subject})

    description = args.bids_root / "dataset_description.json"
    if not description.exists():
        description.write_text(
            json.dumps(
                {
                    "Name": "ABIDE-II",
                    "BIDSVersion": "1.8.0",
                    "DatasetType": "raw",
                    "GeneratedBy": [{"Name": "ABIDE-II site merge"}],
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    manifest = args.bids_root / "abideii_bids_manifest.tsv"
    manifest.write_text(
        "site\tsubject\n" + "\n".join(f"{row['site']}\t{row['subject']}" for row in subjects) + "\n",
        encoding="utf-8",
    )
    print(f"BIDS subjects: {len(subjects)}")
    print(f"BIDS root: {args.bids_root.resolve()}")
    print(f"Manifest: {manifest.resolve()}")


if __name__ == "__main__":
    main()
