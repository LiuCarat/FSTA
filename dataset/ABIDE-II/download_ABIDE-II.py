"""
ABIDE II 原始数据 + 表型下载脚本
================================

功能：
1. 从 FCP-INDI S3 下载 ABIDE II 原始 BIDS 数据
2. 自动下载各站点 participants.tsv
3. 合并生成 ABIDEII_phenotype_merged.csv
4. 支持 ASD / TDC / both 筛选
5. 支持指定站点
6. 支持只下载 T1 + resting-state fMRI
7. HTTP Range 断点续传、失败重试
8. 缓存 participants、subject 名单和 S3 listing
9. 多线程并发下载

推荐流程：
python download_abide2.py --dry-run --site USM_1 --max-subjects 2
python download_abide2.py --site USM_1 --max-subjects 2
python download_abide2.py --workers 4

下载中断后直接重新执行同一条命令即可；只有需要更新远端目录时才使用：
python download_abide2.py --refresh-metadata --refresh-listings
"""

import os
import sys
import time
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import StringIO
from urllib.parse import quote

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# ============================================================
# Windows UTF-8
# ============================================================

if sys.platform == "win32":
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )


# ============================================================
# 配置
# ============================================================

S3_ROOT = "https://s3.amazonaws.com/fcp-indi"

PROJECT_PREFIX = "data/Projects/ABIDE2/RawData"

TIMEOUT = (15, 120)
MAX_RETRIES = 5
RETRY_DELAY = 5
CHUNK_SIZE = 1024 * 1024  # 1 MB
CACHE_DIRNAME = ".download_cache"


# ABIDE II 横断面主要站点
DEFAULT_SITES = [
    "BNI_1",
    "EMC_1",
    "ETHZ_1",
    "GU_1",
    "IP_1",
    "IU_1",
    "KKI_1",
    "KUL_3",
    "NYU_1",
    "NYU_2",
    "OHSU_1",
    "ONRC_2",
    "SDSU_1",
    "TCD_1",
    "UCD_1",
    "UCLA_1",
    "USM_1",
]

PHENOTYPE_ALIASES = {
    "dx": ("dx_group", "DX_GROUP", "diagnosis", "DX"),
    "age": ("age_at_scan", "AGE_AT_SCAN", "age_at_scan "),
    "sex": ("sex", "SEX", "gender", "Gender"),
    "fiq": ("fiq", "FIQ", "Full4 IQ"),
    "viq": ("viq", "VIQ", "Verbal IQ"),
    "piq": ("piq", "PIQ", "Performance IQ"),
    "handedness": (
        "handedness_scores", "handedness_score", "HANDEDNESS_SCORES",
        "handedness_category", "HANDEDNESS_CATEGORY",
    ),
}

CANDIDATE_FIELD_SETS = (
    ("dx",),
    ("dx", "age", "sex"),
    ("dx", "age", "sex", "fiq"),
    ("dx", "age", "sex", "fiq", "piq"),
    ("dx", "age", "sex", "fiq", "piq", "handedness"),
)


# ============================================================
# 下载
# ============================================================

def progress_message(message):
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def download_file(
    url, save_path, expected_size=None, progress_position=None, progress_label=None
):
    """
    下载文件，支持重试和临时 .part 文件。
    """

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    if os.path.exists(save_path) and os.path.getsize(save_path) > 0 and (
        expected_size is None or os.path.getsize(save_path) == expected_size
    ):
        size_mb = os.path.getsize(save_path) / 1024 / 1024
        progress_message(
            f"    已存在: {os.path.basename(save_path)} "
            f"({size_mb:.1f} MB)"
        )
        return True

    tmp_path = save_path + ".part"

    if os.path.exists(save_path) and expected_size is not None:
        if not os.path.exists(tmp_path) or os.path.getsize(save_path) > os.path.getsize(tmp_path):
            os.replace(save_path, tmp_path)
        else:
            os.remove(save_path)

    if expected_size is not None and os.path.exists(tmp_path):
        partial_size = os.path.getsize(tmp_path)
        if partial_size == expected_size:
            os.replace(tmp_path, save_path)
            progress_message(f"    已完成断点文件: {os.path.basename(save_path)}")
            return True
        if partial_size > expected_size:
            os.remove(tmp_path)

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            resume_from = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            headers = {"User-Agent": "Mozilla/5.0"}
            if resume_from:
                headers["Range"] = f"bytes={resume_from}-"

            with requests.get(url, stream=True, timeout=TIMEOUT, headers=headers) as response:

                response.raise_for_status()

                resumed = resume_from > 0 and response.status_code == 206
                if resume_from and not resumed:
                    resume_from = 0

                total_size = expected_size
                if total_size is None:
                    content_length = getattr(response, "headers", {}).get("Content-Length")
                    if content_length:
                        total_size = int(content_length) + (resume_from if resumed else 0)
                label = progress_label or os.path.basename(save_path)
                file_progress = None
                if tqdm is not None:
                    file_progress = tqdm(
                        total=total_size,
                        initial=resume_from if resumed else 0,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"{label[:34]:34s}",
                        position=progress_position,
                        leave=False,
                        dynamic_ncols=True,
                    )

                with open(tmp_path, "ab" if resumed else "wb") as f:

                    for chunk in response.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):
                        if chunk:
                            f.write(chunk)
                            if file_progress is not None:
                                file_progress.update(len(chunk))
                if file_progress is not None:
                    file_progress.close()

            os.replace(
                tmp_path,
                save_path
            )

            if expected_size is not None and os.path.getsize(save_path) != expected_size:
                raise IOError(
                    f"size mismatch: got {os.path.getsize(save_path)}, "
                    f"expected {expected_size}"
                )

            size_mb = os.path.getsize(save_path) / 1024 / 1024

            progress_message(f"    完成: {label} ({size_mb:.1f} MB)")

            return True

        except requests.exceptions.HTTPError as e:

            code = (
                e.response.status_code
                if e.response is not None
                else "?"
            )

            progress_message(f"    {os.path.basename(save_path)}: HTTP {code}")

            if code == 404:
                return False

        except requests.exceptions.Timeout:

            progress_message(f"    {os.path.basename(save_path)}: TIMEOUT")

        except requests.exceptions.ConnectionError:

            progress_message(f"    {os.path.basename(save_path)}: CONNECTION ERROR")

        except Exception as e:

            progress_message(
                f"{type(e).__name__}: "
                f"{str(e)[:80]}"
            )

        if attempt < MAX_RETRIES:

            wait = RETRY_DELAY * (
                2 ** (attempt - 1)
            )

            progress_message(f"      等待 {wait}s 后重试...")

            time.sleep(wait)

    return False


# ============================================================
# S3 Listing
# ============================================================

def list_s3_objects(prefix):
    """
    使用匿名 S3 ListObjectsV2 获取指定前缀下的所有文件。
    """

    objects = []
    continuation_token = None

    while True:

        params = {
            "list-type": "2",
            "prefix": prefix,
            "max-keys": 1000,
        }

        if continuation_token:
            params["continuation-token"] = continuation_token

        response = requests.get(
            S3_ROOT,
            params=params,
            timeout=TIMEOUT,
        )

        response.raise_for_status()

        root = ET.fromstring(
            response.content
        )

        ns = {
            "s3":
            "http://s3.amazonaws.com/doc/2006-03-01/"
        }

        for item in root.findall(
            "s3:Contents",
            ns
        ):

            key_node = item.find("s3:Key", ns)
            size_node = item.find("s3:Size", ns)

            if key_node is not None:
                objects.append({
                    "key": key_node.text,
                    "size": int(size_node.text) if size_node is not None else None,
                })

        truncated = root.find(
            "s3:IsTruncated",
            ns
        )

        if (
            truncated is None
            or truncated.text != "true"
        ):
            break

        token_node = root.find(
            "s3:NextContinuationToken",
            ns
        )

        if token_node is None:
            break

        continuation_token = token_node.text

    return objects


def list_s3_keys(prefix):
    """Backward-compatible key-only S3 listing."""
    return [item["key"] for item in list_s3_objects(prefix)]


def _cache_path(cache_root, prefix):
    digest = hashlib.sha1(prefix.encode("utf-8")).hexdigest()
    return os.path.join(cache_root, "listings", f"{digest}.json")


def cached_s3_objects(prefix, cache_root, refresh=False):
    path = _cache_path(cache_root, prefix)
    if not refresh and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            pass
    objects = list_s3_objects(prefix)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".part"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(objects, handle)
    os.replace(temporary, path)
    return objects


# ============================================================
# 表型
# ============================================================

def load_site_participants(site, cache_dir=None, refresh=False, local_root=None):
    """
    下载站点 participants.tsv 到内存。
    """

    site_dir = f"ABIDEII-{site}"

    key = (
        f"{PROJECT_PREFIX}/"
        f"{site_dir}/participants.tsv"
    )

    url = (
        f"{S3_ROOT}/"
        f"{quote(key, safe='/')}"
    )

    local_path = None
    if local_root:
        local_path = os.path.join(
            local_root, f"ABIDEII-{site}", "participants.tsv"
        )
        if os.path.isfile(local_path):
            try:
                df = pd.read_csv(local_path, sep="\t")
                df["SITE_ID"] = site
                print(f"  {site}: 使用本地 participants.tsv")
                return df
            except Exception as error:
                print(f"  {site}: 本地 participants.tsv 读取失败: {error}")

    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, "participants", f"{site}.tsv")
        if not refresh and os.path.isfile(cache_path):
            try:
                df = pd.read_csv(cache_path, sep="\t")
                df["SITE_ID"] = site
                return df
            except Exception:
                pass

    try:

        response = requests.get(
            url,
            timeout=TIMEOUT,
        )

        if response.status_code != 200:
            print(
                f"  {site}: participants.tsv "
                f"不可用 ({response.status_code})"
            )
            return None

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path + ".part", "w", encoding="utf-8") as handle:
                handle.write(response.text)
            os.replace(cache_path + ".part", cache_path)
        df = pd.read_csv(StringIO(response.text), sep="\t")

        df["SITE_ID"] = site

        return df

    except Exception as e:

        print(
            f"  {site}: 表型读取失败: {e}"
        )

        return None


def find_subject_column(df):

    candidates = [
        "participant_id",
        "SUB_ID",
        "subject_id",
        "subject",
    ]

    for col in candidates:

        if col in df.columns:
            return col

    return None


def find_dx_column(df):

    candidates = [
        "dx_group",
        "DX_GROUP",
        "diagnosis",
        "DX",
    ]

    for col in candidates:

        if col in df.columns:
            return col

    return None


def normalize_subject_id(value):
    """
    转为 sub-xxxxx。
    """

    s = str(value).strip()

    if s.endswith(".0"):
        s = s[:-2]

    if s.startswith("sub-"):
        return s

    return f"sub-{s}"


def load_subject_list(path, max_subjects=None):
    """Load a two-column SITE_ID/subject download list."""
    subject_df = pd.read_csv(path)
    required = {"SITE_ID", "subject"}
    missing = required - set(subject_df.columns)
    if missing:
        raise ValueError(f"Subject list missing columns: {sorted(missing)}")
    subject_df = subject_df[["SITE_ID", "subject"]].copy()
    subject_df["SITE_ID"] = (
        subject_df["SITE_ID"].astype(str).str.strip().str.replace(
            r"^ABIDEII-", "", regex=True
        )
    )
    subject_df["subject"] = subject_df["subject"].map(normalize_subject_id)
    subject_df = subject_df.drop_duplicates().reset_index(drop=True)
    if max_subjects is not None:
        subject_df = subject_df.head(max_subjects)
    return subject_df


def _find_column(df, aliases):
    normalized = {str(column).strip().lower(): column for column in df.columns}
    for alias in aliases:
        column = normalized.get(str(alias).strip().lower())
        if column is not None:
            return column
    return None


def _valid_phenotype_value(value):
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text not in {"", "nan", "n/a", "na", "none", "-999", "-9999"}


def build_phenotype_inventory(merged_pheno, diagnosis="both", sites=None):
    """Normalize common ABIDE-II phenotype fields without touching imaging data."""
    subject_column = _find_column(
        merged_pheno, ("participant_id", "SUB_ID", "subject_id", "subject")
    )
    site_column = _find_column(merged_pheno, ("SITE_ID", "site_id", "site"))
    if subject_column is None or site_column is None:
        raise ValueError("Merged phenotype is missing subject or site columns")

    field_columns = {
        name: _find_column(merged_pheno, aliases)
        for name, aliases in PHENOTYPE_ALIASES.items()
    }
    records = []
    selected_sites = set(sites or [])
    for _, row in merged_pheno.iterrows():
        site = str(row[site_column]).strip()
        site = site.removeprefix("ABIDEII-")
        if selected_sites and site not in selected_sites:
            continue
        subject = normalize_subject_id(row[subject_column])
        record = {"SITE_ID": site, "subject": subject}
        for name, column in field_columns.items():
            record[name] = row[column] if column is not None else np.nan
            record[f"has_{name}"] = _valid_phenotype_value(record[name])
        dx = pd.to_numeric(pd.Series([record["dx"]]), errors="coerce").iloc[0]
        record["diagnosis"] = "asd" if dx == 1 else "tdc" if dx == 2 else "unknown"
        if diagnosis != "both" and record["diagnosis"] != diagnosis:
            continue
        records.append(record)
    return pd.DataFrame(records)


def save_phenotype_audit(inventory, phenotype_dir, required_fields=()):
    """Save field availability, candidate counts, and an optional eligible list."""
    os.makedirs(phenotype_dir, exist_ok=True)
    inventory_path = os.path.join(phenotype_dir, "phenotype_inventory.csv")
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")

    required_fields = tuple(required_fields or ())
    valid_mask = pd.Series(True, index=inventory.index)
    for field in required_fields:
        valid_mask &= inventory[f"has_{field}"]

    eligible = inventory.loc[valid_mask].copy()
    ineligible = inventory.loc[~valid_mask].copy()
    if required_fields:
        ineligible["missing_fields"] = ineligible.apply(
            lambda row: "+".join(
                field for field in required_fields if not row[f"has_{field}"]
            ),
            axis=1,
        )

    eligible.to_csv(
        os.path.join(phenotype_dir, "phenotype_eligible_subjects.csv"),
        index=False, encoding="utf-8-sig",
    )
    ineligible.to_csv(
        os.path.join(phenotype_dir, "phenotype_ineligible_subjects.csv"),
        index=False, encoding="utf-8-sig",
    )

    summary = {
        "total_subjects": int(len(inventory)),
        "diagnosis_counts": {
            str(key): int(value)
            for key, value in inventory["diagnosis"].value_counts().items()
        },
        "field_available": {
            field: int(inventory[f"has_{field}"].sum())
            for field in PHENOTYPE_ALIASES
        },
        "candidate_sets": {
            "+".join(fields): int(
                inventory[[f"has_{field}" for field in fields]].all(axis=1).sum()
            )
            for fields in CANDIDATE_FIELD_SETS
        },
        "required_fields": list(required_fields),
        "eligible_subjects": int(len(eligible)),
    }
    summary_path = os.path.join(phenotype_dir, "phenotype_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return eligible, summary, inventory_path, summary_path


def print_phenotype_summary(summary):
    print("\n表型数据概览")
    print(f"  总人数: {summary['total_subjects']}")
    print(f"  诊断分布: {summary['diagnosis_counts']}")
    print("  单字段有效人数:")
    for field, count in summary["field_available"].items():
        print(f"    {field:12s}: {count}")
    print("  常用字段组合完整人数:")
    for fields, count in summary["candidate_sets"].items():
        print(f"    {fields}: {count}")
    if summary["required_fields"]:
        fields = "+".join(summary["required_fields"])
        print(f"  当前要求 {fields}: {summary['eligible_subjects']} 人")


# ============================================================
# 判断需要下载哪些文件
# ============================================================

def wanted_file(key):
    """
    下载 fMRIPrep 最需要的内容：
    - T1w
    - resting-state BOLD
    - JSON sidecars
    - fieldmap
    - scans.tsv
    """

    name = os.path.basename(key)

    # T1
    if name.endswith("_T1w.nii.gz"):
        return True

    if name.endswith("_T1w.json"):
        return True

    # resting BOLD
    if (
        "task-rest" in name
        and name.endswith("_bold.nii.gz")
    ):
        return True

    if (
        "task-rest" in name
        and name.endswith("_bold.json")
    ):
        return True

    # field maps
    if "/fmap/" in key:
        if name.endswith(".nii.gz"):
            return True
        if name.endswith(".json"):
            return True

    # session metadata
    if name.endswith("_scans.tsv"):
        return True

    return False


def download_site_level_sidecars(site, out_dir, cache_root, refresh_listing=False):
    """
    ABIDE II 很多 JSON 使用 BIDS inheritance，
    放在站点根目录，而不是每个 subject 内。
    """

    site_dir = f"ABIDEII-{site}"

    prefix = (
        f"{PROJECT_PREFIX}/{site_dir}/"
    )

    objects = cached_s3_objects(prefix, cache_root, refresh=refresh_listing)

    for item in objects:
        key = item["key"]

        relative_after_site = key.replace(
            prefix,
            "",
            1
        )

        # 只下载根目录文件
        if "/" in relative_after_site:
            continue

        name = os.path.basename(key)

        if name not in [
            "dataset_description.json",
            "participants.tsv",
            "task-rest_bold.json",
            "T1w.json",
        ]:
            continue

        relative = key.replace(
            PROJECT_PREFIX + "/",
            "",
            1
        )

        save_path = os.path.join(
            out_dir,
            relative
        )

        url = (
            f"{S3_ROOT}/"
            f"{quote(key, safe='/')}"
        )

        download_file(url, save_path, item.get("size"))


def download_subject(row, out_dir, cache_root, refresh_listing=False, progress_position=1):
    """List and download one subject; safe to run in a worker thread."""
    site = row["SITE_ID"]
    subject = row["subject"]
    site_dir = f"ABIDEII-{site}"
    prefix = f"{PROJECT_PREFIX}/{site_dir}/{subject}/"
    try:
        objects = cached_s3_objects(prefix, cache_root, refresh=refresh_listing)
    except Exception as error:
        return subject, False, f"S3 listing 失败: {error}"

    files = [item for item in objects if wanted_file(item["key"])]
    if not files:
        return subject, False, "未找到可用 T1/BOLD"

    subject_ok = True
    for item in files:
        key = item["key"]
        relative = key.replace(PROJECT_PREFIX + "/", "", 1)
        save_path = os.path.join(out_dir, "raw", relative)
        url = f"{S3_ROOT}/{quote(key, safe='/')}"
        if not download_file(
            url,
            save_path,
            item.get("size"),
            progress_position=progress_position,
            progress_label=f"{site}/{subject}",
        ):
            subject_ok = False
    return subject, subject_ok, f"找到 {len(files)} 个文件"


def download_subject_with_slot(row, out_dir, cache_root, refresh_listing, slots):
    position = slots.get()
    try:
        return download_subject(
            row, out_dir, cache_root, refresh_listing, position
        )
    finally:
        slots.put(position)


# ============================================================
# 主函数
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "ABIDE II raw BIDS + phenotype downloader"
    )

    parser.add_argument(
        "-o",
        "--out-dir",
        default="./ABIDEII",
        help="输出目录"
    )

    parser.add_argument(
        "--diagnosis",
        choices=[
            "asd",
            "tdc",
            "both"
        ],
        default="both",
    )

    parser.add_argument(
        "--site",
        default=None,
        help="例如 USM_1 / NYU_1"
    )

    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="只取前 N 个被试，测试时使用"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示文件，不下载"
    )

    parser.add_argument(
        "--workers", type=int, default=4,
        help="并发下载线程数，默认 4"
    )
    parser.add_argument(
        "--refresh-metadata", action="store_true",
        help="重新下载 participants.tsv 并重建 subject 列表"
    )
    parser.add_argument(
        "--refresh-listings", action="store_true",
        help="重新请求 S3 文件列表；默认使用本地缓存"
    )
    parser.add_argument(
        "--phenotype-only", action="store_true",
        help="只整理和统计表型，不下载 T1/BOLD"
    )
    parser.add_argument(
        "--required-fields", nargs="+",
        choices=sorted(PHENOTYPE_ALIASES),
        default=["dx"],
        help="表型预筛选要求完整的字段；默认只要求 dx"
    )
    parser.add_argument(
        "--subject-list",
        help="直接使用 SITE_ID,subject 两列 CSV 下载，不重新生成全量名单"
    )

    args = parser.parse_args()

    out_dir = os.path.abspath(
        args.out_dir
    )

    if args.workers < 1:
        parser.error("--workers 必须大于等于 1")

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    sites = (
        [args.site]
        if args.site
        else DEFAULT_SITES
    )

    cache_root = os.path.join(out_dir, CACHE_DIRNAME)
    selection_path = os.path.join(
        out_dir, "phenotype", "download_selection.json"
    )
    selection = {
        "diagnosis": args.diagnosis,
        "site": args.site,
        "max_subjects": args.max_subjects,
        "sites": sites,
    }


    # ========================================================
    # Step 1 phenotype
    # ========================================================

    print(
        "=" * 70
    )
    print(
        "Step 1: 读取各站点 participants.tsv"
    )
    print(
        "=" * 70
    )

    phenotype_frames = []
    subject_records = []
    reuse_subject_list = False
    subject_list_path = os.path.join(out_dir, "phenotype", "download_subjects.csv")
    phenotype_dir = os.path.join(out_dir, "phenotype")
    existing_merged_path = os.path.join(
        phenotype_dir, "ABIDEII_phenotype_merged.csv"
    )

    if args.phenotype_only and not args.refresh_metadata and os.path.isfile(existing_merged_path):
        merged_pheno = pd.read_csv(existing_merged_path, encoding="utf-8-sig")
        inventory = build_phenotype_inventory(
            merged_pheno, diagnosis=args.diagnosis, sites=sites
        )
        eligible, summary, inventory_path, summary_path = save_phenotype_audit(
            inventory, phenotype_dir, args.required_fields
        )
        candidate_path = os.path.join(
            phenotype_dir, "phenotype_download_subjects.csv"
        )
        eligible[["SITE_ID", "subject"]].to_csv(candidate_path, index=False)
        print_phenotype_summary(summary)
        print(f"  读取已有表型: {existing_merged_path}")
        print(f"  表型盘点: {inventory_path}")
        print(f"  表型统计: {summary_path}")
        print(f"  候选下载名单: {candidate_path}")
        print("\n--phenotype-only：已完成表型盘点，不下载影像文件。")
        return

    if args.subject_list:
        subject_list_path = os.path.abspath(args.subject_list)
        subject_df = load_subject_list(subject_list_path, args.max_subjects)
        reuse_subject_list = True
        sites = sorted(subject_df["SITE_ID"].unique().tolist())
        print(f"使用指定下载名单，跳过表型和全量 subject 列表流程: {subject_list_path}")
    elif not args.refresh_metadata and os.path.isfile(selection_path) and os.path.isfile(subject_list_path):
        try:
            with open(selection_path, encoding="utf-8") as handle:
                reuse_subject_list = json.load(handle) == selection
        except (OSError, ValueError):
            reuse_subject_list = False

    if reuse_subject_list and not args.subject_list:
        subject_df = pd.read_csv(subject_list_path)
        print("使用本地 subject 列表和 S3 listing 缓存，跳过重复的元数据流程")
    elif not reuse_subject_list:
        for site in sites:
            print(f"\n[{site}]")
            df = load_site_participants(
                site,
                cache_dir=cache_root,
                refresh=args.refresh_metadata,
                local_root=out_dir,
            )
            if df is None:
                continue
            phenotype_frames.append(df.copy())
            subject_col = find_subject_column(df)
            dx_col = find_dx_column(df)
            print(f"  表型人数: {len(df)}")
            print(f"  Subject column: {subject_col}")
            print(f"  Diagnosis column: {dx_col}")
            if subject_col is None:
                print("  WARNING: 找不到 subject ID，跳过")
                continue
            selected = df.copy()
            if dx_col is not None and args.diagnosis != "both":
                dx = pd.to_numeric(selected[dx_col], errors="coerce")
                if args.diagnosis == "asd":
                    selected = selected[dx == 1]
                elif args.diagnosis == "tdc":
                    selected = selected[dx == 2]
            for _, row in selected.iterrows():
                subject = normalize_subject_id(row[subject_col])
                subject_records.append({"SITE_ID": site, "subject": subject})


    # ========================================================
    # 保存 phenotype / phenotype-only inventory
    # ========================================================

    os.makedirs(
        phenotype_dir,
        exist_ok=True
    )

    if not reuse_subject_list and phenotype_frames:

        merged_pheno = pd.concat(
            phenotype_frames,
            ignore_index=True,
            sort=False
        )

        pheno_path = os.path.join(
            phenotype_dir,
            "ABIDEII_phenotype_merged.csv"
        )

        merged_pheno.to_csv(
            pheno_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            "\n合并表型已保存:"
        )

        print(
            f"  {pheno_path}"
        )

        print(
            f"  总记录: {len(merged_pheno)}"
        )

        print(
            f"  总字段: {len(merged_pheno.columns)}"
        )


    # ========================================================
    # subject list
    # ========================================================

    if not reuse_subject_list:
        subject_df = pd.DataFrame(subject_records)

    if not reuse_subject_list and args.max_subjects is not None:

        subject_df = subject_df.head(
            args.max_subjects
        )

    subject_list_path = os.path.join(
        phenotype_dir,
        "download_subjects.csv"
    )

    if not reuse_subject_list:
        subject_df.to_csv(subject_list_path, index=False)
        os.makedirs(os.path.dirname(selection_path), exist_ok=True)
        with open(selection_path + ".part", "w", encoding="utf-8") as handle:
            json.dump(selection, handle, ensure_ascii=False, indent=2)
        os.replace(selection_path + ".part", selection_path)

    if not reuse_subject_list and phenotype_frames:
        inventory = build_phenotype_inventory(
            pd.concat(phenotype_frames, ignore_index=True, sort=False),
            diagnosis=args.diagnosis,
            sites=sites,
        )
        eligible, summary, inventory_path, summary_path = save_phenotype_audit(
            inventory, phenotype_dir, args.required_fields
        )
        print_phenotype_summary(summary)
        print(f"  表型盘点: {inventory_path}")
        print(f"  表型统计: {summary_path}")
        eligible[["SITE_ID", "subject"]].to_csv(
            os.path.join(phenotype_dir, "phenotype_download_subjects.csv"),
            index=False,
        )

    if args.phenotype_only:
        print("\n--phenotype-only：已完成表型盘点，不下载影像文件。")
        return

    print(
        f"\n准备下载被试数: {len(subject_df)}"
    )

    print(
        f"名单: {subject_list_path}"
    )


    # ========================================================
    # Step 2 site-level files
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "Step 2: 下载 BIDS 根目录元数据"
    )

    print(
        "=" * 70
    )

    if not args.dry_run:

        for site in sorted(
            subject_df["SITE_ID"].unique()
        ):

            print(
                f"\n[{site}]"
            )

            download_site_level_sidecars(
                site, out_dir, cache_root,
                refresh_listing=args.refresh_listings,
            )


    # ========================================================
    # Step 3 subjects
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "Step 3: 下载 T1 + resting-state fMRI"
    )

    print(
        "=" * 70
    )

    success = 0
    failed = 0

    total = len(
        subject_df
    )

    if args.dry_run:
        for index, row in subject_df.iterrows():
            site = row["SITE_ID"]
            subject = row["subject"]
            prefix = f"{PROJECT_PREFIX}/ABIDEII-{site}/{subject}/"
            try:
                objects = cached_s3_objects(
                    prefix, cache_root, refresh=args.refresh_listings
                )
                files = [item["key"] for item in objects if wanted_file(item["key"])]
                print(f"\n[{index + 1}/{total}] {site} / {subject}: {len(files)} files")
                for key in files:
                    print(f"    {key}")
            except Exception as error:
                print(f"\n[{index + 1}/{total}] {site} / {subject}: {error}")
                failed += 1
    else:
        print(f"并发线程数: {args.workers}")
        rows = [row for _, row in subject_df.iterrows()]
        progress = tqdm(
            total=total,
            desc="Subject 总进度",
            unit="subject",
            position=0,
            dynamic_ncols=True,
        ) if tqdm is not None else None
        progress_slots = Queue()
        for position in range(1, args.workers + 1):
            progress_slots.put(position)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    download_subject_with_slot,
                    row, out_dir, cache_root, args.refresh_listings,
                    progress_slots,
                ): (index, row["SITE_ID"], row["subject"])
                for index, row in enumerate(rows, 1)
            }
            for future in as_completed(futures):
                index, site, subject = futures[future]
                try:
                    _, subject_ok, message = future.result()
                except Exception as error:
                    subject_ok, message = False, str(error)
                progress_message(f"[{index}/{total}] {site} / {subject}: {message}")
                if progress is not None:
                    progress.update(1)
                if subject_ok:
                    success += 1
                else:
                    failed += 1
        if progress is not None:
            progress.close()


    # ========================================================
    # Summary
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "完成"
    )

    print(
        "=" * 70
    )

    print(
        f"成功 subjects: {success}"
    )

    print(
        f"失败 subjects: {failed}"
    )

    print(
        f"输出目录: {out_dir}"
    )

    print(
        f"表型文件: {phenotype_dir}"
    )


if __name__ == "__main__":
    main()
