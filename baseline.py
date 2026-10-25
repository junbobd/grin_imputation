#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传统方法基线：线性插值 + KNN 插补
======================================================================
与 GRIN-lite 使用完全相同的测试集、完全相同的缺失掩码
（同种子、同生成顺序重建），统一在物理单位下计算 MAE / RMSE / R²，
并合并进 metrics 文件（如 metrics.final.json）生成对比表。

【本版改动】
1. 数据入口改用 resolve_data()：下载失败即终止程序，不再静默切合成数据。
   若主脚本以 --allow_synthetic 跑出合成结果，本脚本会自动读到同一份
   SYNTHETIC 缓存，并把输出文件名加上 .SYNTHETIC 后缀，避免混淆。
2. evaluate() 已升级为返回 MAE / RMSE / R² 三项，无需改调用处。
3. 明确不做随机森林、不做均值填充（均值填充已在主脚本以 mean-fill 名义输出）。
   本脚本只负责 docx《目标定位》里点名的传统基线：线性插值、KNN。
======================================================================
"""
import os, json, argparse
import numpy as np

# 复用主脚本的全部数据/掩码/图逻辑，保证口径完全一致
from grin_imputation import (VARIABLES, STATIONS, resolve_data, make_windows,
                             make_eval_mask, evaluate, DataSourceError)


def baseline_linear(X, M_in):
    """时间维线性插值：对每个 (窗口, 站点, 变量) 单独沿时间轴插值。

    这是传统方法里最常被拿来当对照的，也是最容易在"连续缺测段"崩掉的：
    插值点离两端观测越远，误差越大 —— 这一点在结果里应该能看到。
    """
    S, L, N, C = X.shape
    out = X.copy()
    for s in range(S):
        for n in range(N):
            for c in range(C):
                y = X[s, :, n, c]
                m = M_in[s, :, n, c] > 0.5
                if m.sum() > 1:
                    out[s, :, n, c] = np.interp(np.arange(L), np.where(m)[0], y[m])
                elif m.sum() == 1:
                    out[s, :, n, c] = y[m][0]      # 只有 1 个观测点：常数填充
                # m.sum()==0 时保持原值（真实数据下极少出现，出现说明窗口全缺）
    return out


def baseline_knn(X_in_nan, X_train_nan, n_neighbors=5):
    """KNN 插补：按"时间步=样本、变量=特征"展平，用训练集拟合后变换测试集。

    注意这个基线的一个内在局限（报告里值得写）：
    它把每个时间步当作独立样本，完全丢掉了时间顺序信息，
    因此在连续缺测段上只能靠变量间关系外推，时间维度上是瞎猜的。
    """
    from sklearn.impute import KNNImputer
    imp = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    C = X_in_nan.shape[-1]
    imp.fit(X_train_nan.reshape(-1, C))
    pred = imp.transform(X_in_nan.reshape(-1, C)).reshape(X_in_nan.shape)
    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=168)
    p.add_argument("--stride", type=int, default=72)
    p.add_argument("--mcar_rate", type=float, default=0.15)
    p.add_argument("--block_prob", type=float, default=0.30)
    p.add_argument("--block_maxlen", type=int, default=24)
    p.add_argument("--knn_neighbors", type=int, default=5)
    p.add_argument("--source", choices=["merra2", "power_hourly"], default="merra2")
    p.add_argument("--start", type=int, default=20180101)   # 须与主脚本一致，否则掩码对不上
    p.add_argument("--end", type=int, default=20231231)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--allow_synthetic", action="store_true")
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    # 与主脚本同种子、同顺序 -> 掩码完全一致
    rng = np.random.default_rng(args.seed)
    X_raw, M_obs, meta = resolve_data(args)
    suf = ".SYNTHETIC" if meta.get("synthetic") else ""
    tag = f".{args.tag}" if args.tag else ""

    C = X_raw.shape[-1]
    Ttr = int(X_raw.shape[0] * 0.7)
    mean = X_raw[:Ttr].reshape(-1, C).mean(0)
    std = X_raw[:Ttr].reshape(-1, C).std(0)
    std[std < 1e-8] = 1.0
    Xn = (X_raw - mean) / std
    Xw, Mw, _ = make_windows(Xn, M_obs, args.window, args.stride)
    n_tr, n_va = int(len(Xw) * 0.7), int(len(Xw) * 0.85)

    splits = {}
    for nm, sl in [("train", slice(0, n_tr)), ("val", slice(n_tr, n_va)),
                   ("test", slice(n_va, len(Xw)))]:
        M_eval = make_eval_mask(Mw[sl], rng, args.mcar_rate, args.block_prob, args.block_maxlen)
        splits[nm] = (Xw[sl], Mw[sl] * (1 - M_eval), M_eval)

    Xte, M_in_te, M_eval_te = splits["test"]
    Xtr, M_in_tr = splits["train"][0], splits["train"][1]
    Xte_phys = Xte * std + mean
    print(f">> 测试窗口 {len(Xte)} 个 | 评估缺口占比 {M_eval_te.mean():.3f}")

    results = {}
    # ---- 基线1：线性插值 ----
    pred_lin = baseline_linear(Xte, M_in_te)
    results["linear-interp"] = evaluate(pred_lin * std + mean, Xte_phys, M_eval_te, "linear-interp")

    # ---- 基线2：KNN ----
    X_in_te = np.where(M_in_te > 0.5, Xte, np.nan)
    X_in_tr = np.where(M_in_tr > 0.5, Xtr, np.nan)
    pred_knn = baseline_knn(X_in_te, X_in_tr, args.knn_neighbors)
    results["KNN"] = evaluate(pred_knn * std + mean, Xte_phys, M_eval_te, "KNN")

    # ---- 合并已有结果（GRIN-lite / mean-fill / LSTM）----
    mpath = f"metrics{tag}{suf}.json"
    if os.path.exists(mpath):
        old = json.load(open(mpath))
        old.update(results)
        results = old
    json.dump(results, open(mpath, "w"), indent=2)

    print(f"\n========== 统一对比表（物理单位）==========")
    print(f"{'model':>16}  {'MAE':>9}  {'RMSE':>9}  {'R2':>9}")
    for k, v in results.items():
        print(f"{k:>16}  {v['MAE']:>9.4f}  {v['RMSE']:>9.4f}  {v['R2']:>+9.4f}")
    print(f">> 已更新 {mpath}")
    if suf:
        print("\n!! 提醒：数据为合成数据，以上指标不得写入报告  !!")


if __name__ == "__main__":
    main()
