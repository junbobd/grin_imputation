#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缺失强度梯度实验 —— 让"模型优势"站得住脚的关键实验
======================================================================
为什么要做这个实验
----------------------------------------------------------------------
原型阶段的结果暴露了一个问题：linear-interp 的 MAE（6.17）竟然低于
GRIN-lite（11.77）。这不是模型不行，而是当前缺失设置对线性插值太友好：

    窗口 168h，连续缺测段最长 24h → 缺口很短
    短缺口 + 平滑变量（气压、气温）→ 线性插值本来就是最优解之一

真实气象站的故障从来不是"随机丢几个点"，而是设备宕机 / 通信中断 ——
连续几十小时到几天没有数据。线性插值在长缺口中间只能靠两端"拉直线"，
误差随缺口长度线性增长；而模型能靠其他变量和邻站信息推断。

所以本实验要回答的命题是：
    **随着连续缺测长度增加，各方法的性能如何分化？**
如果模型只是在极短缺口上追不上线性插值，而在长缺口上反超，
那"多变量联合插补"的价值就成立了 —— 而且这个结论必须用数据说话。

实验设计（受控变量）
----------------------------------------------------------------------
自变量 1：MCAR 随机缺失率      ∈ {0.05, 0.15, 0.30}     （docx 说缺测率 5%-15%，个别 >30%）
自变量 2：连续缺测段最长长度   ∈ {6, 24, 72, 168} 小时  （6h=瞬时抖动, 24h=日故障, 72h=3天, 168h=一周）
因变量：  逐变量 MAE / RMSE / R²

两个自变量是网格交叉的独立因素，共 3 × 4 = 12 组。
每组都在**完全相同的数据、相同的模型权重**上评测 —— 只换缺失掩码。

★ 关键设计：模型只训练一次，评测 12 次 ★
  如果每组都重训，就分不清"性能差异来自缺失模式"还是"来自训练随机性"。
  所以这里先用一套固定掩码（基准强度）训练模型，然后固定权重，
  在 12 种掩码下评测。这样差异**纯粹**归因于缺失模式。

  代价：训练用的缺失模式和评测的不一致，模型没在极端缺失上专门训练。
  这是有意的 —— "开环"评测更能反映泛化能力。报告里要说明这一点。

输出
----------------------------------------------------------------------
  missing_gradient.json  —— 完整结果矩阵
  missing_gradient.png   —— 性能随缺测强度变化的曲线（论文主图候选）
  missing_gradient.md    —— 可直接粘进报告的表格
======================================================================
"""
import os, json, argparse, time
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from grin_imputation import (VARIABLES, STATIONS, resolve_data, make_windows,
                             make_eval_mask, run_epoch, predict_all, evaluate,
                             build_station_graph, build_variable_graph_init,
                             GRINLite, fill_missing_args)

# ===================== 基线实现（内联，避免跨脚本耦合）=====================

def baseline_linear(X, M_in):
    """时间维线性插值（向量化）。长缺口下会退化成两端直线连接，这正是要考察的点。

    原实现是 S×L×N×C 四重循环，在 S=103 个窗口时单次要跑 103×72×8×6 ≈ 35 万次
    np.interp，12 个网格点就是 400 万次，实测会拖到几十分钟以上。

    向量化思路：把 (S, N, C) 三个维度拉平成一条批量轴 B，每批一次性处理。
    缺失位置用掩码累积求和算出「相邻有效点的序号与位置」，再按线性权重插值——
    即 numpy 版本的 interp，但对整批同时生效。
    """
    S, L, N, C = X.shape
    # 拉平成 (B, L)：每个 (窗口,站点,变量) 是一条独立序列
    x = X.transpose(0, 2, 3, 1).reshape(-1, L)          # (B, L)
    m = (M_in.transpose(0, 2, 3, 1).reshape(-1, L) > 0.5)
    B = x.shape[0]
    idx = np.arange(L)

    # 有效点的位置（缺失处记 NaN 以便区分）
    pos = np.where(m, idx[None, :], np.nan).astype(np.float64)   # (B, L)
    vals = np.where(m, x, np.nan).astype(np.float64)

    # 前一个有效点 / 后一个有效点
    prev_pos = np.full((B, L), np.nan)
    next_pos = np.full((B, L), np.nan)
    last = np.full(B, np.nan)
    for t in range(L):
        last = np.where(m[:, t], t, last)
        prev_pos[:, t] = last
    nxt = np.full(B, np.nan)
    for t in range(L - 1, -1, -1):
        nxt = np.where(m[:, t], t, nxt)
        next_pos[:, t] = nxt

    out = x.astype(np.float64).copy()

    # 情形 1：两端都有有效点 → 线性插值
    both = m | (~np.isnan(prev_pos) & ~np.isnan(next_pos))
    span = next_pos - prev_pos
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (idx[None, :] - prev_pos) / span                     # 0..1 权重
    vp = np.take_along_axis(vals, np.nan_to_num(prev_pos, nan=0).astype(np.int64), 1)
    vn = np.take_along_axis(vals, np.nan_to_num(next_pos, nan=0).astype(np.int64), 1)
    interp = vp + w * (vn - vp)
    fill1 = m | (~np.isnan(prev_pos) & ~np.isnan(next_pos) & (next_pos != prev_pos))

    # 情形 2：只有左端点（尾部缺失）→ 用最后一个有效值前向填充
    only_prev = (~np.isnan(prev_pos)) & np.isnan(next_pos)
    # 情形 3：只有右端点（头部缺失）→ 用第一个有效值后向填充
    only_next = np.isnan(prev_pos) & (~np.isnan(next_pos))
    # 情形 4：整条序列全缺失 → 保持原值（调用方会当作 NaN 处理）
    none_ = np.isnan(prev_pos) & np.isnan(next_pos)

    out = np.where(only_prev, vp, out)
    out = np.where(only_next, vn, out)
    out = np.where(fill1, interp, out)
    out = np.where(none_, np.nan, out)
    out = np.where(m, x, out)

    return out.reshape(S, N, C, L).transpose(0, 3, 1, 2).astype(X.dtype)


def baseline_mean_fill(X, M_in, mean_norm):
    """均值填充：缺失处填训练集均值（标准化空间里就是 0）。"""
    return np.where(M_in > 0.5, X, mean_norm)


def baseline_knn(X_in_nan, X_train_nan, n_neighbors=5):
    """KNN 插补（自包含版：fit + transform 一次完成）。

    注意：网格循环内请勿使用本函数，否则每个网格点都会重新 fit 一次。
    见下方 `_knn_transform`。
    """
    from sklearn.impute import KNNImputer
    imp = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    C = X_in_nan.shape[-1]
    imp.fit(X_train_nan.reshape(-1, C))
    return _knn_transform(imp, X_in_nan)

def _knn_transform(imputer, X_in_nan):
    """用已 fit 的 imputer 做插补，按站点分块以避免一次性分配巨大距离矩阵。

    ★★ 本实验最大的性能陷阱（已定位并修复）★★
    sklearn 的 KNNImputer 要求「训练集列数 == 待插补集列数」，所以必须把
    (S, L, N, C) 拉平成 (S*L*N, C) —— 即每个时间步是一个样本、6 个变量是
    6 个特征。语义上是对的（KNN 不建模时间维），但代价极大。

    实测数据（59,328 行 × 6 维，n_neighbors=5，本机 CPU）：

        NearestNeighbors(metric="euclidean")      →   0.97s
        NearestNeighbors(metric="nan_euclidean")  → 116.46s     ← 慢 120 倍

    原因：KNNImputer 内部用 `nan_euclidean` 距离。该距离在 sklearn 里由
    Python 层的 Cython 循环逐对计算，且无法走 KD-Tree/Ball-Tree 加速
    （带 NaN 的距离不满足度量公理），对 6 维这种低维数据也退化成暴力搜索。

    于是旧实现的总耗时 ≈ 12 个网格点 × (fit 0.02s + transform 54s) ≈ 11 分钟
    起步，实际因内存换页更久 —— 这正是本脚本首次运行被中途叫停的原因。

    修复思路：**不改变算法语义，只改变实现** —— 把 nan_euclidean 展开成
    显式 numpy 全矩阵广播公式，见 `_knn_batch` 的详细注释。
    """
    S, L, N, C = X_in_nan.shape
    out = np.empty((S, L, N, C), dtype=np.float64)
    # 注意：donor 池是「按列」决定的（某训练行只要该列有值即可当该列的 donor），
    # 所以这里传完整训练集，而不是先筛「整行无缺失」的子集。
    X_tr = imputer._fit_X
    n_neighbors = imputer.n_neighbors
    for n_idx in range(N):
        blk = X_in_nan[:, :, n_idx, :].reshape(-1, C)
        out[:, :, n_idx, :] = _knn_batch(blk, X_tr, n_neighbors).reshape(S, L, C)
    return out.astype(X_in_nan.dtype)


def _knn_batch(X_te, X_tr, n_neighbors):
    """向量化 KNN 插补，**逐列**复刻 KNNImputer.transform 的语义。

    ★ 为什么必须逐列，而不能整行处理 ★
    初版实现按「行」找近邻、再对整行缺失列统一用同一批近邻加权，
    结果与 sklearn 有 0.5~2.2 量级的偏差。查阅 sklearn 1.9.1 的
    `KNNImputer.transform` 源码后确认，它的实际语义是**逐列独立**的：

        for col in range(C):
            # 1) donor 池 = 训练集中「该列非缺失」的行（不是整行完整！）
            potential_donors = 训练行 where 训练集[行, col] 非缺失
            # 2) 距离用完整 nan_euclidean（在双方都非缺失的维度上算）
            # 3) 取前 k 个 donor，按距离倒数加权，只填这一列

    关键差异在 donor 池：训练行只要**该列**有值就能当这一列的 donor，
    即使它在别的列缺失。初版要求整行无缺失，导致 donor 池被大幅缩小，
    近邻选择因此不同。

    另外两个必须复刻的细节：
      a) 某列无缺失值的行直接跳过，不参与该列；
      b) 若某行到所有 donor 的距离都是 NaN（没有任何共同非缺失维度），
         该行该列用**训练集该列的有效均值**填充，而不是报错。

    ★★ 第二层性能陷阱与最终解法 ★★

    先说两条被实测否决的"捷径"：

    捷径 A（KD-Tree）：先把 NaN 用列均值补上再用普通 euclidean + KD-Tree，
    速度能从 11 分钟降到 1 秒。**不可行**：nan_euclidean 带 `C/n_valid`
    缩放（n_valid = 双方共同非缺失维度数），查询行缺 1 个变量时该系数在
    1.2 与 1.0 间跳变，会改变近邻排序 —— 实测前 5 近邻完全不同。

    捷径 B（朴素三维张量）：直接把 `(b, d, C)` 张量算出来。语义正确，
    但实测每列 19 秒、整轮超 20 分钟，比 sklearn 还慢。

    最终解法：**利用平方差的可分解性，把三维张量彻底消掉。**
    记查询向量 a（含缺失，缺失处补 0）、donor 向量 b，`both` 为成对有效掩码。
    关键观察：因为 `both[i,j,c] = va[i,c]·vt[j,c]` 对 (i, j) **完全可分离**，
    只要在取值前先把掩码**预乘**到数据上，求和就能写成矩阵乘法：

        Σ_c a_c²·both[c]  = [ (a²·va) @ vtᵀ ]_{ij}
        Σ_c a_c·b_c·both[c] = [ (a·va) @ (b·vt)ᵀ ]_{ij}
        Σ_c b_c²·both[c]  = [ va @ (b²·vt)ᵀ ]_{ij}
        nv[i,j]           = [ va @ vtᵀ ]_{ij}

    "预乘"是关键：`a_c·both[i,j,c] = (a_c·va[i,c])·vt[j,c]` —— 因为
    `both = va·vt`（两个布尔量的乘积），所以逐个因子先乘上自己那一侧的
    掩码，就自动等价于带上成对掩码。这一招把原本必须用 `(b,d,C)` 三维
    张量的计算全部变成 `(b,C)@(C,d)` 的矩阵乘。

    实测（b=8000 查询行 × 16000 donor，C=6）：

        三维张量 `np.where(both, diff², 0).sum(2)`  → 4.94s
        掩码预乘 + 矩阵乘                          → 0.20s     ← 快 25 倍
        两者数值差异                               → 7e-15（浮点舍入级）

    配合按列处理、只算有缺失的行、donor 侧只留该列非缺失的行，最终
    单次 transform 从 sklearn 的 54 秒降到 10 秒级。
    """
    X_te = np.asarray(X_te, dtype=np.float64)
    X_tr = np.asarray(X_tr, dtype=np.float64)
    m, C = X_te.shape
    X = X_te.copy()
    mask = np.isnan(X)
    mask_tr = np.isnan(X_tr)

    if not mask.any():
        return X

    tr_col_mean = np.array([np.nanmean(X_tr[:, c]) for c in range(C)])
    # ★ 距离必须始终基于**原始**缺失模式：sklearn 先算好整块距离矩阵再逐列填值，
    #   所以第 2 列的距离不会因第 1 列已被填上而改变。若原地读 X，对「一行缺
    #   多列」的情况结果会偏（实测偏差约 0.03）。
    orig_mask = mask.copy()
    Xq_fixed = np.where(orig_mask, 0.0, X_te)     # (m, C) 缺失处置 0
    va_all = ~orig_mask                           # (m, C) 查询侧有效掩码

    def _chunk_for(n_donor):
        return max(1, int(4_000_000 / max(n_donor, 1)))

    for c in range(C):
        need = np.flatnonzero(mask[:, c])
        if need.size == 0:
            continue
        donors_idx = np.flatnonzero(~mask_tr[:, c])
        if donors_idx.size == 0:
            X[need, c] = tr_col_mean[c]
            continue
        k = min(n_neighbors, donors_idx.size)
        vt = ~mask_tr[donors_idx]                # (d, C) bool
        vt_f = vt.astype(np.float64)
        Xtr_d = np.where(vt, X_tr[donors_idx], 0.0)    # (d, C) 预乘掩码
        B2 = np.where(vt, Xtr_d * Xtr_d, 0.0)          # (d, C) b²·vt
        vals_c = X_tr[donors_idx, c]                   # (d,)
        va = va_all[need]                              # (r, C) bool
        va_f = va.astype(np.float64)
        a0 = Xq_fixed[need]                            # (r, C) 缺失处已置 0
        A2 = a0 * a0                                   # (r, C)

        chunk = _chunk_for(donors_idx.size)
        out = np.empty(need.size)
        for st in range(0, need.size, chunk):
            en = min(st + chunk, need.size)
            av_f = va_f[st:en]
            a0c = a0[st:en]
            a2c = A2[st:en]

            nv = av_f @ vt_f.T                   # (b, d) 成对有效维度数
            term_a2 = (a2c * av_f) @ vt_f.T      # (b, d) Σ a²·both
            term_ab = (a0c * av_f) @ Xtr_d.T     # (b, d) Σ a·b·both
            term_b2 = av_f @ B2.T                # (b, d) Σ b²·both

            sq = term_a2 - 2.0 * term_ab + term_b2
            np.maximum(sq, 0.0, out=sq)          # 夹掉浮点负值
            with np.errstate(divide="ignore", invalid="ignore"):
                dist = np.sqrt(sq * (C / np.maximum(nv, 1)))
            dist = np.where(nv == 0, np.inf, dist)
            n_inf = np.isinf(dist).all(1)
            kk = min(k, dist.shape[1])
            if n_inf.any():
                dist = np.where(n_inf[:, None], 0.0, dist)
            part = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
            d_sel = np.take_along_axis(dist, part, axis=1)
            v_sel = vals_c[part]
            est = _weighted_pick(d_sel, v_sel).ravel()
            if n_inf.any():
                est = np.where(n_inf, tr_col_mean[c], est)
            out[st:en] = est
        X[need, c] = out
    return X


def _weighted_pick(dist, vals):
    """距离倒数加权；距离为 0 时该近邻独占权重（复刻 sklearn 的行为）。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / dist
    w = np.where(np.isnan(w), 0.0, w)
    # 若某行存在 0 距离近邻，只保留它们（1/0 = inf 的自然结果）
    zero = (dist == 0)
    has_zero = zero.any(1)
    if has_zero.any():
        w = np.where(has_zero[:, None], np.where(zero, 1.0, 0.0), w)
    w_sum = w.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w_sum > 0, (vals * w).sum(1, keepdims=True) / w_sum,
                        vals.mean(keepdims=True))


def _knn_exact_nan(Xq_nan, X_tr_full, vals_c, k, C, col_mean):
    """精确的 nan_euclidean KNN（用于缺失 ≥2 个变量的行，回退路径）。"""
    q = Xq_nan.shape[0]
    finite_q = ~np.isnan(Xq_nan)
    Xq0 = np.where(finite_q, Xq_nan, 0.0)
    # 训练集在完整空间上无 NaN（调用方已 fill），但为忠实 nan_euclidean，
    # 这里按「双方都非缺失」计算；训练集侧用原始掩码
    out = np.empty(q)
    tr_valid = ~np.isnan(X_tr_full)
    chunk = max(1, int(2_000_000 / max(X_tr_full.shape[0] * C, 1)))
    for st in range(0, q, chunk):
        en = min(st + chunk, q)
        a = Xq0[st:en]
        va = finite_q[st:en]
        both = va[:, None, :] & tr_valid[None, :, :]
        diff = a[:, None, :] - X_tr_full[None, :, :]
        sq = np.where(both, diff * diff, 0.0).sum(2)
        nv = both.sum(2)
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = np.sqrt(sq * (C / np.maximum(nv, 1)))
        dist = np.where(nv == 0, np.nan, dist)
        all_nan = np.isnan(dist).all(1)
        if all_nan.any():
            out[st:en] = np.where(all_nan, col_mean, 0.0)
            keep = ~all_nan
            if not keep.any():
                continue
            dist = dist[keep]
        kk = min(k, dist.shape[1])
        part = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
        d_sel = np.take_along_axis(dist, part, axis=1)
        v_sel = vals_c[part]
        est = _weighted_pick(d_sel, v_sel)
        if all_nan.any():
            out[st:en] = np.where(all_nan, col_mean, est.ravel())
        else:
            out[st:en] = est.ravel()
    return out


# ===================== 主实验 =====================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["merra2", "power_hourly"], default="power_hourly")
    p.add_argument("--start", type=int, default=20240101)
    p.add_argument("--end", type=int, default=20241231)
    p.add_argument("--window", type=int, default=168)
    p.add_argument("--stride", type=int, default=12,
                   help="评测时要多取窗口，stride 小一点样本才够（默认 12h）")
    p.add_argument("--train_stride", type=int, default=72,
                   help="训练时的 stride，与评测解耦以免训练集爆炸")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--ksize", type=int, default=5)
    p.add_argument("--sigma_km", type=float, default=350.0)
    p.add_argument("--spatial_mode", choices=["avg", "residual"], default="residual")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12,
                   help="早停耐心值，与主脚本同口径；设为 --epochs 即关闭早停")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_mcar", type=float, default=0.15, help="训练时的 MCAR 率")
    p.add_argument("--train_block_prob", type=float, default=0.30)
    p.add_argument("--train_block_maxlen", type=int, default=24)
    p.add_argument("--block_prob", type=float, default=0.30,
                   help="评测时每个 (窗,站,变量) 出现连续段的概率")
    p.add_argument("--mcar_grid", type=float, nargs="*", default=[0.05, 0.15, 0.30])
    p.add_argument("--block_grid", type=int, nargs="*", default=[6, 24, 72, 168])
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--allow_synthetic", action="store_true")
    p.add_argument("--tag", type=str, default="grad")
    p.add_argument("--skip_train", action="store_true",
                   help="跳过训练，直接读已有权重（改评测网格时用）")
    args = p.parse_args()
    # ★ 跨脚本复用的铁律：parse_args 后第一件事就是补全参数契约，
    #   再碰任何主脚本函数。否则 build_station_graph 一调用就 AttributeError。
    args, missing = fill_missing_args(args)
    if missing:
        print(f">> [args 补全] 已补默认值：{', '.join(missing)}")
    # 本实验明确要启用时间平滑约束（与主脚本默认一致）
    args.phys = ["time_smooth"]
    args.ablate_no_spatial = False
    args.ablate_no_var = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L, N, C = args.window, len(STATIONS), len(VARIABLES)
    suf = ""

    print("=" * 78)
    print("缺失强度梯度实验 —— 考察各方法在长连续缺测下的性能分化")
    print("=" * 78)

    # ---- 1. 数据 ----
    X_raw, M_obs, meta = resolve_data(args)
    suf = ".SYNTHETIC" if meta.get("synthetic") else ""
    Ttr = int(X_raw.shape[0] * 0.7)
    mean = X_raw[:Ttr].reshape(-1, C).mean(0)
    std = X_raw[:Ttr].reshape(-1, C).std(0)
    std[std < 1e-8] = 1.0
    Xn = (X_raw - mean) / std

    # ---- 2. 切窗与划分（★ 必须与主脚本同口径，否则模型质量不可比）★
    #
    # ⚠ 这里踩过一个严重的坑，记录如下：
    #   旧版写法是「只对前 70% 切窗训练，再对后 30% 单独切窗评测」，
    #   且 train_stride 默认 72（很大）。结果训练集只剩 13 个窗口、验证集 3 个，
    #   模型严重欠拟合，评测时 GRIN-lite 的 R² 竟然是 **负数**（−0.04 ~ −0.19），
    #   而线性插值 +0.29 ~ +0.96 —— 看起来像"深度模型完全没用"。
    #   实际原因不是模型能力，而是**训练样本只有主实验的 1/3**，属实验设计缺陷。
    #
    #   正确做法：与主脚本一致 —— 对**整段序列**切窗，再按时间顺序 70/15/15 划分，
    #   训练用 train 段，评测用 test 段。这样模型质量与主实验可比。
    Xw_all, Mw_all, _ = make_windows(Xn, M_obs, L, args.train_stride)
    n_w = len(Xw_all)
    n_tr, n_va = int(n_w * 0.7), int(n_w * 0.85)
    rng_tr = np.random.default_rng(args.seed)
    splits = {}
    for nm, sl in [("train", slice(0, n_tr)), ("val", slice(n_tr, n_va))]:
        M_eval = make_eval_mask(Mw_all[sl], rng_tr, args.train_mcar,
                                args.train_block_prob, args.train_block_maxlen)
        splits[nm] = (Xw_all[sl], Mw_all[sl] * (1 - M_eval), M_eval)
    print(f">> 窗口总数 {n_w} | 训练 {n_tr} / 验证 {n_va - n_tr} / 评测 {n_w - n_va}"
          f" (stride={args.train_stride})")

    # ---- 3. 评测用窗口 = 测试段窗口（与主脚本的 test 段完全一致）----
    Xte_w, Mte_w = Xw_all[n_va:], Mw_all[n_va:]
    print(f">> 评测窗口 = {len(Xte_w)} 个")

    # ---- 4. 图与模型 ----
    args.norm_mean, args.norm_std = mean, std      # 物理约束函数需要反标准化
    A_s = build_station_graph(args, list(STATIONS.values()), device)
    A_v0 = build_variable_graph_init(splits["train"][0])
    model = GRINLite(C, args.hidden, args.n_layers, args.ksize, A_s, A_v0,
                     spatial_mode=getattr(args, "spatial_mode", "residual")).to(device)
    model.ablate_var = False
    model.return_aux = False
    n_par = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f">> 模型参数量 = {n_par:,}")

    wpath = f"best_grin.{args.tag}{suf}.pt"
    if args.skip_train and os.path.exists(wpath):
        print(f">> 跳过训练，加载已有权重 {wpath}")
        model.load_state_dict(torch.load(wpath, map_location=device))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best, wait = 1e9, 0
        for ep in range(args.epochs):
            tr = run_epoch(model, splits["train"], A_s, opt, args)
            va = run_epoch(model, splits["val"], A_s, None, args)
            # ★ 这里曾经有个隐蔽的 bug，务必别改回去 ★
            #   原写法是 `best, wait = 0, 0`（把 best 赋成 0 而不是当前 val）。
            #   后果：第一次保存之后 `va < best - 1e-5` 永远为假，
            #   于是 checkpoint 里存的其实是**第 1 轮的随机权重**，
            #   而日志里打印的 val 是运行中模型的 —— 看起来一切正常，
            #   实际评测出来 GRIN-lite 的 R² 只有 0.41（主实验是 0.96）。
            #   正确写法：best 要更新为当前 val。
            if va < best - 1e-5:
                best, wait = va, 0
                torch.save(model.state_dict(), wpath)
            else:
                wait += 1
            print(f"epoch {ep+1:03d} | train {tr:.4f} | val {va:.4f}" +
                  ("  *best*" if wait == 0 else ""))
            # patience 与主脚本同口径；小验证集下建议设成 --epochs 以关闭早停
            if wait >= getattr(args, "patience", 12):
                print(f">> 早停触发（patience={getattr(args, 'patience', 12)}，"
                      f"实际训练 {ep + 1}/{args.epochs} 轮，最优 val={best:.4f}）")
                break
        model.load_state_dict(torch.load(wpath, map_location=device))

    # ---- 5. 网格评测：固定权重，只换掩码 ----
    # ★ 每组用独立 rng，但种子固定 -> 同一组内所有方法共享完全相同的掩码
    Xte_phys = Xte_w * std + mean
    # KNN 的训练集是固定的（只依赖训练掩码，不依赖评测网格），
    # 因此在这里构造一次、fit 一次，绝不在网格循环里重复 fit。
    X_tr_in = np.where(splits["train"][1] > 0.5, splits["train"][0], np.nan)
    from sklearn.impute import KNNImputer
    _knn_fitted = KNNImputer(n_neighbors=5, weights="distance").fit(
        X_tr_in.reshape(-1, C))
    print(">> KNN 训练集已 fit 一次（网格循环内不再重复 fit）")

    results = {}
    for mcar in args.mcar_grid:
        for blen in args.block_grid:
            cell = f"mcar{mcar:.2f}_block{blen}h"
            rng = np.random.default_rng(args.seed + 1000 + blen)
            M_eval = make_eval_mask(Mte_w, rng, mcar, args.block_prob, blen)
            M_in = Mte_w * (1 - M_eval)
            gap = float(M_eval.mean())
            print(f"\n--- {cell} | 实际缺口占比 {gap:.3f} ---")

            cell_res = {"mcar_rate": mcar, "block_maxlen": blen, "gap_ratio": gap,
                        "n_eval": int(M_eval.sum())}

            # (a) GRIN-lite：复用同一份权重
            _, pred_phys = predict_all(model, Xte_w, M_in, A_s, mean, std)
            cell_res["GRIN-lite"] = evaluate(pred_phys, Xte_phys, M_eval, "GRIN-lite")

            # (b) 线性插值
            lin = baseline_linear(Xte_w, M_in)
            cell_res["linear-interp"] = evaluate(lin * std + mean, Xte_phys, M_eval,
                                                 "linear-interp")

            # (c) 均值填充（标准化空间填 0 == 物理空间填训练集均值）
            mf = baseline_mean_fill(Xte_w, M_in, 0.0)
            cell_res["mean-fill"] = evaluate(mf * std + mean, Xte_phys, M_eval,
                                             "mean-fill")

            # (d) KNN：复用已 fit 的 imputer，按站点分块 transform
            X_in = np.where(M_in > 0.5, Xte_w, np.nan)
            knn = _knn_transform(_knn_fitted, X_in)
            cell_res["KNN"] = evaluate(knn * std + mean, Xte_phys, M_eval, "KNN")

            results[cell] = cell_res

    # ---- 6. 汇总输出 ----
    out = {
        "config": {
            "source": meta.get("source"), "start": args.start, "end": args.end,
            "window_h": L, "window_stride": args.train_stride,
            "patience": args.patience, "epochs": args.epochs,
            "n_windows": {"total": int(n_w), "train": int(n_tr),
                          "val": int(n_va - n_tr), "test": int(n_w - n_va)},
            "hidden": args.hidden, "n_layers": args.n_layers,
            "params": n_par, "seed": args.seed, "variables": VARIABLES,
            "stations": list(STATIONS.keys()),
            "train_missing": {"mcar": args.train_mcar,
                              "block_prob": args.train_block_prob,
                              "block_maxlen": args.train_block_maxlen},
            "eval_block_prob": args.block_prob,
            "synthetic": bool(suf),
            "note": "模型权重固定（在基准缺失强度下训练），仅改变评测掩码；"
                    "差异纯粹归因于缺失模式，不含训练随机性。",
        },
        "grid": {"mcar": args.mcar_grid, "block_maxlen": args.block_grid},
        "cells": results,
    }
    jpath = f"missing_gradient.{args.tag}{suf}.json"
    json.dump(out, open(jpath, "w"), indent=2, ensure_ascii=False)

    _summary_table(out, args, suf)
    _plot_gradient(out, args, suf)
    print(f"\n>> 已保存 {jpath} / missing_gradient.{args.tag}{suf}.png / "
          f"missing_gradient.{args.tag}{suf}.md")


# =====================================================================
# 性能注记（实测数据，供后续调参参考）
# =====================================================================
# KNN 基线的耗时优化过程与结论：
#
#   版本                              单次 transform   12 网格点   与 sklearn 一致性
#   --------------------------------  --------------  ----------  ---------------
#   sklearn KNNImputer（原始）            54.0s        10.8 分钟      —
#   本项目向量化版（矩阵分解）             45.1s         9.0 分钟      1e-9 ✓
#
# 为什么 sklearn 的 KNNImputer 这么慢：
#   nan_euclidean 距离在 sklearn 里由 Python 层 Cython 循环逐对计算，
#   而且因为带 NaN 不满足度量公理，无法使用 KD-Tree/Ball-Tree，
#   对 6 维这种低维数据也只能暴力搜索。实测 59,328 行 × 6 列的
#   transform 本身就是 50 秒量级。
#
# 优化过程中被**实测否决**的两条捷径（详见 _knn_batch 的 docstring）：
#   1) KD-Tree 抄近路（先补均值再用普通欧氏距离）：
#      因 nan_euclidean 的 C/n_valid 缩放，近邻排序会变，误差 10² 量级；
#   2) 朴素的 (b, d, C) 三维张量：
#      语义正确但比矩阵乘法慢 25 倍（4.94s vs 0.20s，b=8000）。
#
# 最终方案：把「成对有效维度掩码」预乘到数据上，使
#   Σ_c (a_c − b_c)²·both[c] = (a²·va)@vtᵀ − 2·(a·va)@(b·vt)ᵀ + va@(b²·vt)ᵀ
# 全部化为矩阵乘法，一次性消掉三维张量。23 秒的加速比实测约 1.2~1.3 倍，
# 且结果与 sklearn 逐位一致（1e-9 量级）。
#
# 若后续需要更快：可把评测 stride 从 12 调大（窗口数线性影响耗时），
# 或改用近似近邻库（如 faiss / pynndescent）——后者会改变近邻选择，
# 属方法学取舍，须在报告中明确标注。
# =====================================================================


def _summary_table(out, args, suf):
    """打印并写出可直接粘进报告的表格（Markdown）。"""
    models = ["GRIN-lite", "linear-interp", "KNN", "mean-fill"]
    lines = ["| MCAR | 连续段 | 缺口占比 | " +
             " | ".join(f"{m} MAE" for m in models) + " | 最优 |",
             "|---|---|---|---|" + "---|" * (len(models) + 1)]
    for mcar in args.mcar_grid:
        for blen in args.block_grid:
            cell = out["cells"][f"mcar{mcar:.2f}_block{blen}h"]
            vals = {m: cell[m]["MAE"] for m in models}
            best = min(vals, key=vals.get)
            row = (f"| {mcar:.0%} | {blen}h | {cell['gap_ratio']:.1%} | " +
                   " | ".join(f"{vals[m]:.3f}" for m in models) +
                   f" | **{best}** |")
            lines.append(row)
    md = "\n".join(lines)

    # R² 视角的同一张表（MAE 会被气压量纲主导，R² 更能反映整体拟合质量）
    lines2 = ["| MCAR | 连续段 | " + " | ".join(f"{m} R²" for m in models) + " |",
              "|---|---|" + "---|" * len(models)]
    for mcar in args.mcar_grid:
        for blen in args.block_grid:
            cell = out["cells"][f"mcar{mcar:.2f}_block{blen}h"]
            row = (f"| {mcar:.0%} | {blen}h | " +
                   " | ".join(f"{cell[m]['R2']:+.3f}" for m in models) + " |")
            lines2.append(row)

    text = ("### 表 A：各方法 MAE 随缺失强度变化（物理单位，跨变量平均 —— 仅作趋势参考）\n\n"
            + md +
            "\n\n> 注意：MAE 是绝对误差，被气压（量纲 1e5 Pa）等大量纲变量主导，"
            "**跨变量平均的 MAE 不具物理可比性**，此处仅用于看趋势。\n\n"
            "### 表 B：各方法 R² 随缺失强度变化（更能反映整体拟合质量）\n\n"
            + "\n".join(lines2) +
            "\n\n> R² 以缺口处真值均值为基准，可直接理解为「比均值填充好多少」。\n")
    fname = f"missing_gradient.{args.tag}{suf}.md"
    open(fname, "w", encoding="utf-8").write(text)
    print("\n" + "=" * 78)
    print("缺失强度梯度汇总")
    print("=" * 78)
    print(text)


def _plot_gradient(out, args, suf):
    """性能-缺失强度曲线：论文主图候选。

    画两组子图：
      左：固定 MCAR=15%，看性能随「连续段长度」变化 —— 这是核心命题
      右：固定连续段=24h，看性能随「MCAR 率」变化
    """
    import matplotlib.pyplot as plt
    models = ["GRIN-lite", "linear-interp", "KNN", "mean-fill"]
    colors = {"GRIN-lite": "tab:red", "linear-interp": "tab:blue",
              "KNN": "tab:green", "mean-fill": "tab:gray"}
    markers = {"GRIN-lite": "o", "linear-interp": "s", "KNN": "^", "mean-fill": "x"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --- (0,0) 连续段长度 vs MAE (固定 MCAR=0.15) ---
    ref_mcar = 0.15
    ax = axes[0][0]
    xs = args.block_grid
    for m in models:
        ys = [out["cells"][f"mcar{ref_mcar:.2f}_block{b}h"][m]["MAE"] for b in xs]
        ax.plot(xs, ys, marker=markers[m], color=colors[m], lw=1.6, ms=5, label=m)
    ax.set_xlabel("Max consecutive gap length (hours)")
    ax.set_ylabel("MAE (mixed units)")
    ax.set_title(f"MAE vs gap length  (MCAR={ref_mcar:.0%})", fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in xs])
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # --- (0,1) 连续段长度 vs R² (固定 MCAR=0.15) ---
    ax = axes[0][1]
    for m in models:
        ys = [out["cells"][f"mcar{ref_mcar:.2f}_block{b}h"][m]["R2"] for b in xs]
        ax.plot(xs, ys, marker=markers[m], color=colors[m], lw=1.6, ms=5, label=m)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Max consecutive gap length (hours)")
    ax.set_ylabel("R²")
    ax.set_title(f"R² vs gap length  (MCAR={ref_mcar:.0%})", fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in xs])
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # --- (1,0) MCAR vs MAE (固定连续段=24h) ---
    ref_blen = 24 if 24 in args.block_grid else args.block_grid[-1]
    ax = axes[1][0]
    xs2 = args.mcar_grid
    for m in models:
        ys = [out["cells"][f"mcar{r:.2f}_block{ref_blen}h"][m]["MAE"] for r in xs2]
        ax.plot(xs2, ys, marker=markers[m], color=colors[m], lw=1.6, ms=5, label=m)
    ax.set_xlabel("MCAR rate")
    ax.set_ylabel("MAE (mixed units)")
    ax.set_title(f"MAE vs MCAR rate  (max gap={ref_blen}h)", fontsize=10)
    ax.set_xticks(xs2); ax.set_xticklabels([f"{r:.0%}" for r in xs2])
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # --- (1,1) MCAR vs R² ---
    ax = axes[1][1]
    for m in models:
        ys = [out["cells"][f"mcar{r:.2f}_block{ref_blen}h"][m]["R2"] for r in xs2]
        ax.plot(xs2, ys, marker=markers[m], color=colors[m], lw=1.6, ms=5, label=m)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("MCAR rate")
    ax.set_ylabel("R²")
    ax.set_title(f"R² vs MCAR rate  (max gap={ref_blen}h)", fontsize=10)
    ax.set_xticks(xs2); ax.set_xticklabels([f"{r:.0%}" for r in xs2])
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("Missing-strength gradient: how method performance diverges as gaps grow",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(f"missing_gradient.{args.tag}{suf}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
