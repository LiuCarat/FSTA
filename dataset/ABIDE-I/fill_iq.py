"""
ABIDE-I IQ 缺失值填补(合并版:原 fill_fiq.py + fill_viq_piq.py)
============================================================

Stage 1 (fill_fiq.py): Phenotypic_V1_0b_preprocessed1.csv
    FIQ 缺失 → 用 VIQ、PIQ、SEX、AGE_AT_SCAN 做线性回归预测;
    VIQ/PIQ 也缺失时 → 用训练集 FIQ 均值填补。
    → Phenotypic_V1_0b_preprocessed1_fiq_filled.csv

Stage 2 (fill_viq_piq.py): Phenotypic_Processing.csv
    1. VIQ 缺但 PIQ 有 → 回归: VIQ ~ FIQ + PIQ
    2. PIQ 缺但 VIQ 有 → 回归: PIQ ~ FIQ + VIQ
    3. 两者都缺       → 回归: VIQ/PIQ ~ FIQ only
    → Phenotypic_Processing_filled.csv

无 MRI 数据泄露,仅用表型字段。
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Stage 1: 填补 FIQ(原始表型文件)
# ============================================================
def fill_fiq(input_path, output_path):
    df = pd.read_csv(input_path)
    n_total = len(df)
    print(f"[Stage 1] 总受试者数: {n_total}")

    # ---------- 1. 把 -9999 替换为 NaN ----------
    for col in ["FIQ", "VIQ", "PIQ"]:
        df[col] = df[col].replace(-9999, np.nan)

    # ---------- 2. 标记 FIQ 缺失 ----------
    missing_fiq = df["FIQ"].isna()
    missing_viq = df["VIQ"].isna()
    missing_piq = df["PIQ"].isna()
    print(f"[Stage 1] FIQ 缺失: {missing_fiq.sum()} 人")

    # ---------- 3. 构建回归特征 ----------
    features = ["VIQ", "PIQ", "SEX", "AGE_AT_SCAN"]

    # 训练集:FIQ 已知 且 VIQ/PIQ 都有
    train_mask = ~missing_fiq & ~missing_viq & ~missing_piq
    X_train = df.loc[train_mask, features].copy()
    y_train = df.loc[train_mask, "FIQ"].copy()
    print(f"[Stage 1] 训练集 (FIQ+VIQ+PIQ 齐全): {len(X_train)} 人")

    # 缺失 VIQ 的,用 VIQ 均值填充
    for col in features:
        if col in ["VIQ", "PIQ", "AGE_AT_SCAN"]:  # 数值列填均值
            train_mean = X_train[col].mean()
            X_train[col] = X_train[col].fillna(train_mean)

    # ---------- 4. 训练线性回归 ----------
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 打印回归方程
    print(f"\n[Stage 1] 回归方程: FIQ = {model.intercept_:.2f}"
          + "".join(f" + {c:.3f}*{n}" for c, n in zip(model.coef_, features)))
    print(f"[Stage 1] R² = {model.score(X_train, y_train):.3f}")

    # ---------- 5. 预测缺失的 FIQ ----------
    # 5a. 能回归填补的:FIQ 缺失 但 VIQ+PIQ 都有
    predict_mask = missing_fiq & ~missing_viq & ~missing_piq
    X_predict = df.loc[predict_mask, features].copy()
    for col in features:
        if X_predict[col].isna().any():
            X_predict[col] = X_predict[col].fillna(df.loc[train_mask, col].mean())

    predicted_fiq = model.predict(X_predict)
    df.loc[predict_mask, "FIQ"] = np.round(predicted_fiq).astype(int)
    print(f"[Stage 1] 回归填补: {predict_mask.sum()} 人 (FIQ 缺失但 VIQ+PIQ 齐全)")

    # 5b. FIQ 缺失 且 VIQ 或 PIQ 也缺失的,用均值填补
    cant_predict = missing_fiq & ~predict_mask
    if cant_predict.sum() > 0:
        mean_fiq = int(round(y_train.mean()))
        df.loc[cant_predict, "FIQ"] = mean_fiq
        print(f"[Stage 1] 均值填补: {cant_predict.sum()} 人 (VIQ/PIQ 也缺失), 填补值={mean_fiq}")

    # ---------- 6. 验证 ----------
    remaining_missing = df["FIQ"].isna().sum()
    print(f"\n[Stage 1] 填补后 FIQ 仍缺失: {remaining_missing} 人")
    print(f"[Stage 1] 最终可用受试者: {n_total - remaining_missing} 人")

    # 保存
    df.to_csv(output_path, index=False)
    print(f"[Stage 1] 已保存到: {output_path}\n")


# ============================================================
# Stage 2: 填补 VIQ 和 PIQ(实验用表型文件)
# ============================================================
def fill_viq_piq(input_path, output_path):
    df = pd.read_csv(input_path)
    n_total = len(df)
    print(f"[Stage 2] 总受试者数: {n_total}")

    # ---------- 1. 标记缺失 ----------
    missing_viq = df["VIQ"].isna() | (df["VIQ"] == -9999)
    missing_piq = df["PIQ"].isna() | (df["PIQ"] == -9999)
    print(f"[Stage 2] VIQ 缺失: {missing_viq.sum()}")
    print(f"[Stage 2] PIQ 缺失: {missing_piq.sum()}")

    # 完整样本(训练回归模型用)
    complete = ~missing_viq & ~missing_piq
    df_complete = df[complete].copy()
    print(f"[Stage 2] 三个 IQ 完整: {len(df_complete)} 人")

    # ---------- 2. 回归模型 ----------
    # VIQ ~ FIQ + PIQ
    X_viq = df_complete[["FIQ", "PIQ"]].values
    y_viq = df_complete["VIQ"].values
    model_viq_full = LinearRegression().fit(X_viq, y_viq)
    print(f"[Stage 2] VIQ ~ FIQ + PIQ: R²={model_viq_full.score(X_viq, y_viq):.3f}")

    # PIQ ~ FIQ + VIQ
    X_piq = df_complete[["FIQ", "VIQ"]].values
    y_piq = df_complete["PIQ"].values
    model_piq_full = LinearRegression().fit(X_piq, y_piq)
    print(f"[Stage 2] PIQ ~ FIQ + VIQ: R²={model_piq_full.score(X_piq, y_piq):.3f}")

    # VIQ ~ FIQ only (fallback)
    X_fiq = df_complete[["FIQ"]].values
    model_viq_fiq = LinearRegression().fit(X_fiq, y_viq)
    print(f"[Stage 2] VIQ ~ FIQ only: R²={model_viq_fiq.score(X_fiq, y_viq):.3f}")

    # PIQ ~ FIQ only (fallback)
    model_piq_fiq = LinearRegression().fit(X_fiq, y_piq)
    print(f"[Stage 2] PIQ ~ FIQ only: R²={model_piq_fiq.score(X_fiq, y_piq):.3f}")

    # ---------- 3. 分层填补 ----------
    # 3a. VIQ 缺但 PIQ 有
    mask = missing_viq & ~missing_piq
    n_filled = mask.sum()
    if n_filled > 0:
        df.loc[mask, "VIQ"] = np.round(
            model_viq_full.predict(df.loc[mask, ["FIQ", "PIQ"]].values)
        ).astype(int)
        print(f"[Stage 2] 回归填补 VIQ (FIQ+PIQ): {n_filled} 人")

    # 3b. PIQ 缺但 VIQ 有
    mask = missing_piq & ~missing_viq
    n_filled = mask.sum()
    if n_filled > 0:
        df.loc[mask, "PIQ"] = np.round(
            model_piq_full.predict(df.loc[mask, ["FIQ", "VIQ"]].values)
        ).astype(int)
        print(f"[Stage 2] 回归填补 PIQ (FIQ+VIQ): {n_filled} 人")

    # 3c. 两者都缺 → 只用 FIQ
    mask = missing_viq & missing_piq
    n_filled = mask.sum()
    if n_filled > 0:
        df.loc[mask, "VIQ"] = np.round(
            model_viq_fiq.predict(df.loc[mask, ["FIQ"]].values)
        ).astype(int)
        df.loc[mask, "PIQ"] = np.round(
            model_piq_fiq.predict(df.loc[mask, ["FIQ"]].values)
        ).astype(int)
        print(f"[Stage 2] 回归填补 VIQ+PIQ (FIQ only): {n_filled} 人")

    # ---------- 4. 验证 ----------
    final_missing_viq = df["VIQ"].isna().sum()
    final_missing_piq = df["PIQ"].isna().sum()
    print(f"\n[Stage 2] 填补后 VIQ 仍缺失: {final_missing_viq}")
    print(f"[Stage 2] 填补后 PIQ 仍缺失: {final_missing_piq}")
    print(f"[Stage 2] SEX + FIQ + VIQ + PIQ 全部完整: {n_total - max(final_missing_viq, final_missing_piq)} 人")

    df.to_csv(output_path, index=False)
    print(f"[Stage 2] 已保存到: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 1/2: FIQ 填补")
    print("=" * 60)
    fill_fiq(
        os.path.join(HERE, "Phenotypic_V1_0b_preprocessed1.csv"),
        os.path.join(HERE, "Phenotypic_V1_0b_preprocessed1_fiq_filled.csv"),
    )

    print("=" * 60)
    print("Stage 2/2: VIQ + PIQ 填补")
    print("=" * 60)
    fill_viq_piq(
        os.path.join(HERE, "Phenotypic_Processing.csv"),
        os.path.join(HERE, "Phenotypic_Processing_filled.csv"),
    )
