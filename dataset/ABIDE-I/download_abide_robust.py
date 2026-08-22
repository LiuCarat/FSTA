"""
ABIDE 健壮下载脚本 v2
=====================
- 使用 requests 库，严格超时控制（不会卡死）
- 断点续传（已下载的文件自动跳过）
- 失败自动重试（指数退避，最多 5 次）
- 支持代理配置
- 执行方法: python data/download_abide_robust.py
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
import numpy as np

# Windows: force UTF-8 to avoid UnicodeEncodeError on CJK systems
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ======================== 配置 ========================
S3_PREFIX = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
PHENO_URL = f"{S3_PREFIX}/Phenotypic_V1_0b_preprocessed1.csv"

TIMEOUT = (15, 60)       # (connect_timeout, read_timeout) 秒
MAX_RETRIES = 5           # 每文件最大重试次数
RETRY_DELAY = 5           # 初始重试间隔 (秒)，指数增长
CHUNK_SIZE = 65536        # 64KB 下载块

# 代理配置 (如需使用，取消注释并修改)
PROXY = None


def download_file(url, save_path, retries=MAX_RETRIES):
    """下载单个文件，支持重试和指数退避。使用 requests 流式下载，超时可控。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    if PROXY:
        session.proxies.update(PROXY)

    for attempt in range(1, retries + 1):
        try:
            print(f"  [{attempt}/{retries}] {os.path.basename(save_path)}", end=' ', flush=True)

            resp = session.get(url, timeout=TIMEOUT, stream=True)
            resp.raise_for_status()

            total = int(resp.headers.get('content-length', 0))
            downloaded = 0
            with open(save_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            size_kb = os.path.getsize(save_path) / 1024
            print(f"OK ({size_kb:.0f} KB)")
            return True

        except requests.exceptions.Timeout:
            print(f"FAIL (timeout)")
        except requests.exceptions.ConnectionError as e:
            print(f"FAIL (connection: {str(e)[:40]})")
        except requests.exceptions.HTTPError as e:
            print(f"FAIL (HTTP {e.response.status_code if e.response else '?'})")
        except Exception as e:
            print(f"FAIL ({type(e).__name__}: {str(e)[:50]})")

        # 清理不完整的文件
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except (PermissionError, OSError):
                pass  # file locked on Windows, will be overwritten next attempt

        if attempt < retries:
            wait = RETRY_DELAY * (2 ** (attempt - 1))  # 5, 10, 20, 40, ...
            print(f"    等待 {wait}s...", flush=True)
            time.sleep(wait)
        else:
            print(f"    已达最大重试次数，放弃", flush=True)
            return False

    return False


def main():
    parser = argparse.ArgumentParser(description="ABIDE 健壮下载脚本 v2")
    parser.add_argument('-d', '--derivative', default='rois_aal', help='derivative 类型')
    parser.add_argument('-p', '--pipeline', default='cpac', help='pipeline')
    parser.add_argument('-s', '--strategy', default='filt_noglobal', help='strategy')
    parser.add_argument('-o', '--out_dir', default='./ABIDE_pcp', help='输出目录')
    parser.add_argument('--diagnosis', default='both', choices=['asd', 'tdc', 'both'])
    parser.add_argument('--sex', default=None, choices=['M', 'F'])
    parser.add_argument('--site', default=None, help='站点过滤')
    parser.add_argument('--max_fd', type=float, default=0.2, help='最大 mean FD')
    parser.add_argument('--dry-run', action='store_true', help='仅列出文件，不下载')
    parser.add_argument('--no-qc', action='store_true', help='不应用 QC rater 过滤（下载全部 1035 个）')
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)

    # ========================
    # Step 1: 下载并加载 phenotype (使用 pandas，与 nilearn 一致)
    # ========================
    print("=" * 60)
    print("Step 1: 获取受试者列表")
    print("=" * 60)

    pheno_local = os.path.join(out_dir, "Phenotypic_V1_0b_preprocessed1.csv")
    if not os.path.exists(pheno_local):
        print("下载 phenotype 文件...")
        if not download_file(PHENO_URL, pheno_local, retries=5):
            print("FATAL: 无法获取 phenotype 文件，请检查网络连接")
            sys.exit(1)
    else:
        print(f"phenotype 文件已存在: {pheno_local}")

    # 使用 pandas 加载 (与 nilearn 一致)
    pheno = pd.read_csv(pheno_local)
    total = len(pheno)
    print(f"  表型总条目: {total}")

    # ========================
    # Step 2: 过滤 (严格复刻 nilearn fetch_abide_pcp)
    # ========================
    print("\nStep 2: 筛选符合条件的数据 (nilearn quality_checked 标准)")
    print("=" * 60)

    # 2a: 排除无文件名
    pheno = pheno[pheno['FILE_ID'] != 'no_filename']
    print(f"  排除 no_filename 后: {len(pheno)}")

    # 2b: QC rater 过滤 (复刻 nilearn quality_checked=True)
    if not args.no_qc:
        from nilearn.datasets._utils import filter_columns
        qc_kwargs = {
            'qc_rater_1': 'OK',
            'qc_anat_rater_2': ['OK', 'maybe'],
            'qc_func_rater_2': ['OK', 'maybe'],
            'qc_anat_rater_3': 'OK',
            'qc_func_rater_3': 'OK',
        }
        qc_mask = filter_columns(pheno, qc_kwargs)
        pheno = pheno[qc_mask]
        print(f"  QC rater 过滤后: {len(pheno)}")
    else:
        print(f"  跳过 QC 过滤")

    # 2c: 用户指定过滤
    if args.diagnosis == 'asd':
        pheno = pheno[pheno['DX_GROUP'] == 1]
        print(f"  仅 ASD: {len(pheno)}")
    elif args.diagnosis == 'tdc':
        pheno = pheno[pheno['DX_GROUP'] == 2]
        print(f"  仅 TDC: {len(pheno)}")
    if args.sex == 'M':
        pheno = pheno[pheno['SEX'] == 1]
        print(f"  仅男性: {len(pheno)}")
    elif args.sex == 'F':
        pheno = pheno[pheno['SEX'] == 2]
        print(f"  仅女性: {len(pheno)}")

    dx_counts = pheno['DX_GROUP'].value_counts()
    print(f"  最终: {len(pheno)} 被试 (ASD={dx_counts.get(1,0)}, Control={dx_counts.get(2,0)})")

    # 构建下载列表
    extension = '.1D' if 'roi' in args.derivative else '.nii.gz'
    url_prefix = f"{S3_PREFIX}/Outputs/{args.pipeline}/{args.strategy}/{args.derivative}"
    download_dir = os.path.join(out_dir, args.pipeline, args.strategy)

    to_download = []
    for _, row in pheno.iterrows():
        file_id = row['FILE_ID']
        filename = f"{file_id}_{args.derivative}{extension}"
        url = f"{url_prefix}/{filename}"
        save_path = os.path.join(download_dir, filename)
        to_download.append((url, save_path, file_id))

    if args.dry_run:
        print("\n[Dry Run] 前 10 个:")
        for url, path, fid in to_download[:10]:
            print(f"  {fid} -> {path}")
        print(f"  ... 共 {len(to_download)}")
        return

    # ========================
    # Step 3: 下载
    # ========================
    print(f"\nStep 3: 开始下载 ({len(to_download)} 个文件)")
    print("=" * 60)

    existing = success = fail = 0
    total = len(to_download)

    for i, (url, save_path, file_id) in enumerate(to_download):
        pct = 100 * (existing + success + fail) / total
        print(f"[{i+1}/{total}] ({pct:.1f}%) {file_id}", end='')
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            size_kb = os.path.getsize(save_path) / 1024
            print(f" 已存在 ({size_kb:.0f} KB), 跳过")
            existing += 1
            continue
        elif os.path.exists(save_path):
            print(f" (空文件，重新下载)")

        print()
        if download_file(url, save_path):
            success += 1
        else:
            fail += 1

        # 每 10 个文件输出一次汇总（减少刷屏）
        if (success + fail) % 10 == 0:
            done = existing + success + fail
            print(f"  --- 进度: {done}/{total} ({100*done/total:.1f}%)  "
                  f"成功:{success}  已存在:{existing}  失败:{fail} ---")

    print("\n" + "=" * 60)
    print("下载完成!")
    print(f"  成功: {success}  已存在: {existing}  失败: {fail}  总计: {total}")
    if fail > 0:
        print(f"  提示: 重新运行此脚本以重试失败的 {fail} 个文件")
    print(f"  数据目录: {download_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
