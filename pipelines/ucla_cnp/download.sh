#!/bin/bash
#
# UCLA CNP (ds000030) 数据集统一下载脚本
#
# 用法:
#   bash download.sh              # 根据 participants.tsv 下载所有可用数据
#   bash download.sh --verify     # 仅验证已下载数据的完整性
#   bash download.sh --subjects hc_subjects.txt  # 仅下载指定受试者
#
# 数据来源: OpenNeuro ds000030 (UCLA Consortium for Neuropsychiatric Phenomics)
# BIDS 格式: sub-{id}/{func,anat}/sub-{id}_task-rest_bold.nii.gz
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE="$REPO_ROOT/dataset/ucla_cnp"
S3="https://s3.amazonaws.com/openneuro.org/ds000030"
GITHUB_API="https://api.github.com/repos/OpenNeuroDatasets/ds000030/contents"
GITHUB_RAW="https://raw.githubusercontent.com/OpenNeuroDatasets/ds000030/master"

# ============================================================
# 工具函数
# ============================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()  { echo -e "${GREEN}  ✓${NC} $*"; }
warn(){ echo -e "${YELLOW}  ⚠${NC} $*"; }
fail(){ echo -e "${RED}  ✗${NC} $*"; }

download_metadata() {
    echo "[元数据] 下载 BIDS 元数据文件..."
    curl -sfL "$GITHUB_RAW/dataset_description.json" -o "$BASE/dataset_description.json"
    curl -sfL "$GITHUB_RAW/participants.tsv"        -o "$BASE/participants.tsv"
    curl -sfL "$GITHUB_RAW/task-rest_bold.json"      -o "$BASE/task-rest_bold.json"
    ok "元数据就绪"
}

# 下载 phenotype 表型数据 — 全部 52 组 (共 104 文件: .tsv + .json)
# 来源: https://github.com/OpenNeuroDatasets/ds000030/tree/master/phenotype
#
# 使用 GitHub Contents API 获取完整文件列表，无需手动维护白名单。
# 下载清单 (52 组):
#   人口学/临床基线: demographics scid medication health handedness
#   BD 相关:          hamilton ymrs bipolar_ii
#   SZ 相关:          bprs sans saps chapinf chapper chaphyp chapphy chapsoc
#   ADHD 相关:        adhd asrs acds_adult
#   认知/人格:         wais tci barratt dickman eysenck
#   行为任务:          ant bart colortrails cpt cvlt discounting dkefs
#                     dkefs_spanish golden hopkins stopsignal stroop
#                     taskswitch wms
#   筛查/其他:         chronotype colorvision language mpq rk scap smnm sr
#                     spanish_vocab tbi vcap visualacuity vmnm admin
#
download_phenotype() {
    echo "[表型] 下载 phenotype 数据..."
    local pheno_dir="$BASE/phenotype"
    mkdir -p "$pheno_dir"

    # 通过 GitHub API 获取 phenotype 目录下的全部文件名
    echo "  获取文件列表: $GITHUB_API/phenotype"
    local file_list
    file_list=$(curl -sf "$GITHUB_API/phenotype" \
        | python3 -c "import json,sys; [print(item['name']) for item in json.load(sys.stdin)]" 2>/dev/null)

    if [ -z "$file_list" ]; then
        warn "无法获取 phenotype 文件列表，请检查网络或 GitHub API 限额"
        return
    fi

    local count=0 total=0
    while IFS= read -r fname; do
        [ -z "$fname" ] && continue
        total=$((total + 1))
        if [ -s "$pheno_dir/$fname" ]; then
            continue
        fi
        curl -sfL "$GITHUB_RAW/phenotype/$fname" -o "$pheno_dir/$fname" \
            && count=$((count + 1)) || warn "下载失败: $fname"
    done <<< "$file_list"
    ok "表型数据就绪 ($count 个新文件, 共 $total 个)"
}

# 生成单个文件的 aria2c 条目
# 用法: add_aria2_entry <target_dir> <remote_subpath>
add_aria2_entry() {
    local dir="$1" remote="$2" fname="$3"
    mkdir -p "$dir"
    if [ -s "$dir/$fname" ]; then return 1; fi  # 已存在则跳过
    echo "$S3/$remote"       >> "$ARIA2_LIST"
    echo "  dir=$dir"        >> "$ARIA2_LIST"
    echo "  out=$fname"      >> "$ARIA2_LIST"
    echo "  continue=true"   >> "$ARIA2_LIST"
    return 0
}

# 批量下载
run_aria2() {
    local total=$(grep -c "^https://" "$ARIA2_LIST" 2>/dev/null || echo 0)
    if [ "$total" -eq 0 ]; then
        ok "无需下载，全部文件已就绪"
        return
    fi
    echo "  待下载: $total 个文件"
    aria2c \
        --input-file="$ARIA2_LIST" \
        --max-concurrent-downloads=5 \
        --split=16 \
        --min-split-size=1M \
        --max-connection-per-server=16 \
        --retry-wait=10 \
        --max-tries=10 \
        --timeout=60 \
        --connect-timeout=30 \
        --allow-overwrite=false \
        --auto-file-renaming=false \
        --console-log-level=notice \
        --summary-interval=30 \
        --human-readable=true || true
}

# 下载一个 subject 的所有数据
# 用法: download_subject <pid> <has_rest> <has_t1w>
download_subject() {
    local pid="$1" has_rest="${2:-0}" has_t1w="${3:-0}"

    if [ "$has_rest" = "1" ]; then
        add_aria2_entry "$BASE/$pid/func" "$pid/func/${pid}_task-rest_bold.nii.gz" "${pid}_task-rest_bold.nii.gz"
        add_aria2_entry "$BASE/$pid/func" "$pid/func/${pid}_task-rest_bold.json" "${pid}_task-rest_bold.json"
    fi
    if [ "$has_t1w" = "1" ]; then
        add_aria2_entry "$BASE/$pid/anat" "$pid/anat/${pid}_T1w.nii.gz" "${pid}_T1w.nii.gz"
    fi
}

# 验证单个 subject
# 用法: verify_subject <pid> <expect_rest> <expect_t1w>
verify_subject() {
    local pid="$1" expect_rest="${2:-0}" expect_t1w="${3:-0}"
    local status=""

    if [ "$expect_rest" = "1" ]; then
        local nii="$BASE/$pid/func/${pid}_task-rest_bold.nii.gz"
        local json="$BASE/$pid/func/${pid}_task-rest_bold.json"
        if [ -s "$nii" ]; then
            status="${status}F"
            [ -f "$json" ] || warn "$pid: func nii.gz OK, 缺少 JSON"
        else
            warn "$pid: 缺少 func nii.gz"
        fi
    fi
    if [ "$expect_t1w" = "1" ]; then
        local t1="$BASE/$pid/anat/${pid}_T1w.nii.gz"
        if [ -s "$t1" ]; then
            status="${status}T"
        else
            warn "$pid: 缺少 T1w"
        fi
    fi
    [ -n "$status" ] && ok "$pid ($status)"
}

# ============================================================
# 主流程
# ============================================================

MODE="download"
SUBJECT_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verify)  MODE="verify"; shift ;;
        --subjects) SUBJECT_FILE="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

ARIA2_LIST="/tmp/ucla_cnp_download.txt"
> "$ARIA2_LIST"

echo "============================================"
echo "  UCLA CNP (ds000030) 数据下载"
echo "============================================"
echo ""

# --- 下载元数据 ---
download_metadata

# --- 下载表型 ---
download_phenotype

# --- 纯验证模式 ---
if [ "$MODE" = "verify" ]; then
    echo "[验证] 扫描已下载数据..."
    n_rest=$(find "$BASE" -name "*task-rest_bold.nii.gz" -size +1M 2>/dev/null | wc -l)
    n_t1w=$(find "$BASE" -name "*_T1w.nii.gz" -size +1M 2>/dev/null | wc -l)
    n_subj=$(ls -d "$BASE"/sub-* 2>/dev/null | wc -l)
    echo "  受试者目录: $n_subj"
    echo "  rest fMRI:  $n_rest"
    echo "  T1w:        $n_t1w"

    # 逐受试者验证
    ok_count=0; fail_count=0
    while IFS=$'\t' read -r pid rest t1w; do
        if [ -s "$BASE/$pid/func/${pid}_task-rest_bold.nii.gz" ]; then
            ok_count=$((ok_count+1))
        elif [ "$rest" = "1" ]; then
            warn "$pid 缺失 rest fMRI"; fail_count=$((fail_count+1))
        fi
    done < <(awk -F'\t' -v OFS='\t' 'NR>1 {print $1, $9, $13}' "$BASE/participants.tsv")
    echo ""
    echo "完整: $ok_count, 缺失: $fail_count"
    exit 0
fi

# --- 下载模式 ---
echo "[下载] 构建下载列表..."

if [ -n "$SUBJECT_FILE" ]; then
    # 从指定文件读取 subject 列表（假定均有 rest fMRI）
    echo "  来源: $SUBJECT_FILE"
    while read -r pid _; do
        download_subject "$pid" 1 0
    done < "$SUBJECT_FILE"
else
    # 从 participants.tsv 读取完整列表 (列1=id, 列9=rest, 列13=T1w)
    while IFS=$'\t' read -r pid rest t1w; do
        download_subject "$pid" "$rest" "$t1w"
    done < <(awk -F'\t' -v OFS='\t' 'NR>1 {print $1, $9, $13}' "$BASE/participants.tsv")
fi

# --- 执行下载 ---
echo ""
echo "[下载] 开始传输..."
run_aria2

# --- 验证 ---
echo ""
echo "[验证] 检查完整性..."
rest_count=$(find "$BASE" -name "*task-rest_bold.nii.gz" -size +1M 2>/dev/null | wc -l)
t1w_count=$(find "$BASE" -name "*_T1w.nii.gz" -size +1M 2>/dev/null | wc -l)
subj_count=$(ls -d "$BASE"/sub-* 2>/dev/null | wc -l)
echo "  受试者目录: $subj_count"
echo "  rest fMRI:  $rest_count (预期 ≈206)"
echo "  T1w:        $t1w_count (预期 ≈264)"
echo ""
echo "完成。数据目录: $BASE"
