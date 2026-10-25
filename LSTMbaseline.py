#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线 3：同参数规模 LSTM 插补器
======================================================================
对应 docx《创新点二》要验证的命题：

    "小规模 LSTM 的体积同样不大，因此本作品的优势不在于'LSTM 跑不动'，
     而在于在相同参数预算下，通过卷积结构获得更好的精度。"

所以本脚本的唯一使命就是做这个公平对照：
    控制器变量 = 时序建模结构（卷积 vs 循环）
    固定不变   = 参数量（±5% 以内）、优化器、损失、batch、早停、数据、掩码

===================== 参数量匹配的诚实说明 =====================
匹配是"尽量接近"，不是精确相等。因为 LSTM 参数量与隐藏维是离散跳变的
（H 从 84 加到 85，参数量跳几千），不可能刚好等于 GRIN-lite 的值。
这里取偏差最小的 H，并把实际偏差百分比打印出来，写进 metrics 文件（如
metrics.final.json）的 params_gap_pct 字段，报告里如实标注 —— 不能假装"参数量完全相同"。

【本版改动】
1. 数据入口改用 resolve_data()：下载失败即终止，不再静默切合成数据。
2. 评估指标补齐 R²（docx 第五步要求 MAE / RMSE / R² 三项）。
3. 输出文件名带 tag / .SYNTHETIC 后缀，避免与真实结果混淆。
======================================================================
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

from grin_imputation import (VARIABLES, STATIONS, resolve_data, make_windows,
                             make_eval_mask, run_epoch, predict_all, evaluate,
                             build_station_graph, build_variable_graph_init, GRINLite)


def _fill_missing_args(args):
    """补全主脚本函数可能读取、但本脚本 argparse 未定义的字段。

    为什么要这一步：主脚本的 build_station_graph / run_epoch 会直接访问
    args 的某些属性。跨脚本复用时少定义一个字段就抛 AttributeError。
    这里统一兜底，保证参数契约完整。
    注意补的都是"关闭/默认"值 —— LSTM 基线不应启用任何消融开关或物理约束，
    否则就不是与完整版 GRIN-lite 的公平对照了。
    """
    defaults = {
        "ablate_no_spatial": False,
        "ablate_no_var": False,
        "phys": [],                    # 关键：LSTM 基线不加物理约束
        "phys_weights": {},
        "phys_ablate": True,
        "diurnal_weight_per_var": None,
        "norm_mean": None,
        "norm_std": None,
        "return_aux": False,
    }
    filled = [k for k in defaults if not hasattr(args, k)]
    for k in filled:
        setattr(args, k, defaults[k])
    if filled:
        print(">> [args 补全] 主脚本会读取、本脚本已补默认值的字段：")
        print("   " + ", ".join(f"{k}={defaults[k]}" for k in filled))
    return args


class LSTMImputer(nn.Module):
    """单层 LSTM 插补网络：逐站点独立建模时序（无空间图、无变量图）。

    接口与 GRINLite 完全一致（forward(x, m) -> (B,L,N,C)），
    这样才能共用 run_epoch / predict_all，保证训练过程逐行可比。
    """

    def __init__(self, C, hidden):
        super().__init__()
        self.embed = nn.Linear(2 * C, hidden)      # [观测值, 掩码] -> H
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, C)
        self.return_aux = False

    def forward(self, x, m):                       # x,m: (B,L,N,C)
        B, L, N, C = x.shape
        inp = torch.cat([x * m, m], -1)            # (B,L,N,2C)
        h = self.embed(inp).permute(0, 2, 1, 3).reshape(B * N, L, -1)
        h, _ = self.lstm(h)                        # 循环结构核心
        out = self.head(h)                         # (B*N,L,C)
        return out.reshape(B, N, L, C).permute(0, 2, 1, 3).contiguous()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_lstm_hidden(C, target):
    """搜索隐藏维 H，使 LSTM 参数量最接近 target。

    返回 (H, 参数量, 相对偏差百分比)。
    """
    best_h, best_d = None, 1e18
    for H in range(8, 1024):
        n = (2 * C * H + H) + (4 * H * (H + H) + 8 * H) + (H * C + C)
        if abs(n - target) < best_d:
            best_h, best_d = H, abs(n - target)
    n = (2 * C * best_h + best_h) + (4 * best_h * 2 * best_h + 8 * best_h) + (best_h * C + C)
    gap = (n - target) / target * 100
    if abs(gap) >= 5:
        print(f">> [警告] 参数量匹配偏差 {gap:+.2f}%，超出 ±5% 目标。")
        print("   原因：LSTM 参数量随隐藏维离散跳变，无法精确命中。")
        print("   处理：已在 metrics 文件记录 params_gap_pct，报告需如实标注，不得声称'参数量相同'。")
    return best_h, n, gap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["merra2", "power_hourly"], default="merra2")
    p.add_argument("--start", type=int, default=20180101)
    p.add_argument("--end", type=int, default=20231231)
    p.add_argument("--window", type=int, default=168)
    p.add_argument("--stride", type=int, default=72)
    p.add_argument("--hidden", type=int, default=64)     # GRIN 的 hidden，仅用于数参数量
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--ksize", type=int, default=5)
    p.add_argument("--sigma_km", type=float, default=350.0)
    p.add_argument("--spatial_mode", choices=["avg", "residual"], default="residual")
    p.add_argument("--mcar_rate", type=float, default=0.15)
    p.add_argument("--block_prob", type=float, default=0.30)
    p.add_argument("--block_maxlen", type=int, default=24)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12,
                   help="早停耐心值。与主脚本 GRIN-lite 保持一致才能公平对比；"
                        "设为 --epochs 即关闭早停（验证集较小时推荐）")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--allow_synthetic", action="store_true")
    p.add_argument("--tag", type=str, default="")
    args = _fill_missing_args(p.parse_args())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    L, N, C = args.window, len(STATIONS), len(VARIABLES)

    # ---- 1. 与主脚本完全相同的数据 / 掩码 ----
    X_raw, M_obs, meta = resolve_data(args)
    suf = ".SYNTHETIC" if meta.get("synthetic") else ""
    tag = f".{args.tag}" if args.tag else ""

    Ttr = int(X_raw.shape[0] * 0.7)
    mean = X_raw[:Ttr].reshape(-1, C).mean(0)
    std = X_raw[:Ttr].reshape(-1, C).std(0)
    std[std < 1e-8] = 1.0
    args.norm_mean, args.norm_std = mean, std
    Xn = (X_raw - mean) / std
    Xw, Mw, _ = make_windows(Xn, M_obs, L, args.stride)
    n_tr, n_va = int(len(Xw) * 0.7), int(len(Xw) * 0.85)
    splits = {}
    for nm, sl in [("train", slice(0, n_tr)), ("val", slice(n_tr, n_va)),
                   ("test", slice(n_va, len(Xw)))]:
        M_eval = make_eval_mask(Mw[sl], rng, args.mcar_rate, args.block_prob, args.block_maxlen)
        splits[nm] = (Xw[sl], Mw[sl] * (1 - M_eval), M_eval)

    # ---- 2. 实例化 GRIN-lite 仅用于读出目标参数量（不训练）----
    A_s = build_station_graph(args, list(STATIONS.values()), device)
    A_v0 = build_variable_graph_init(splits["train"][0])
    ref = GRINLite(C, args.hidden, args.n_layers, args.ksize, A_s, A_v0,
                   spatial_mode=getattr(args, "spatial_mode", "residual")).to(device)
    target = count_params(ref)
    H, lstm_n, gap = match_lstm_hidden(C, target)
    print(f">> 参数量对照: GRIN-lite = {target:,}  |  LSTM(H={H}) = {lstm_n:,}  ({gap:+.2f}%)")

    # ---- 3. 训练（复用主脚本 run_epoch：同损失、同 batch、同早停策略）----
    # ★ patience 必须与主脚本一致 ★
    #   否则 LSTM 与 GRIN-lite 的"实际训练轮数"不同，参数量匹配做得再好，
    #   对比也不再公平。主脚本默认 12；做消融/正式对比时统一传 --patience <epochs>。
    model = LSTMImputer(C, H).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, wait, path = 1e9, 0, f"best_lstm{tag}{suf}.pt"
    for ep in range(args.epochs):
        tr = run_epoch(model, splits["train"], A_s, opt, args)
        va = run_epoch(model, splits["val"], A_s, None, args)
        if va < best - 1e-5:
            best, wait = va, 0
            torch.save(model.state_dict(), path)
        else:
            wait += 1
        print(f"epoch {ep+1:03d} | train {tr:.4f} | val {va:.4f}" +
              ("  *best*" if wait == 0 else ""))
        if wait >= getattr(args, "patience", 12):
            print(f">> 早停触发（patience={getattr(args, 'patience', 12)}，"
                  f"实际训练 {ep + 1}/{args.epochs} 轮，最优 val={best:.4f}）")
            break
    model.load_state_dict(torch.load(path, map_location=device))

    # ---- 4. 测试评估（相同掩码、物理单位）----
    Xte, M_in_te, M_eval_te = splits["test"]
    t0 = time.perf_counter()
    _, pred_phys = predict_all(model, Xte, M_in_te, A_s, mean, std)
    dt = (time.perf_counter() - t0) / len(Xte)
    key = "LSTM" + (f"[{args.tag}]" if args.tag else "")
    res = evaluate(pred_phys, Xte * std + mean, M_eval_te, key)
    res.update({"params": lstm_n, "params_gap_pct": gap, "ms_per_sample": dt * 1e3,
                "source": meta.get("source"), "window_h": L, "synthetic": bool(suf)})

    # ---- 5. 合并 metrics 文件输出对比 ----
    mpath = f"metrics{tag}{suf}.json"
    if os.path.exists(mpath):
        old = json.load(open(mpath)); old.update({key: res}); res_all = old
    else:
        res_all = {key: res}
    json.dump(res_all, open(mpath, "w"), indent=2)

    print("\n========== 创新点二对照表（同参数规模、同配置）==========")
    print(f"{'model':>16}  {'MAE':>9}  {'RMSE':>9}  {'R2':>9}   {'params':>9}")
    for k, v in res_all.items():
        pp = f"{v.get('params', 0):,}" if "params" in v else "-"
        print(f"{k:>16}  {v['MAE']:>9.4f}  {v['RMSE']:>9.4f}  {v['R2']:>+9.4f}   {pp:>9}")
    print(f">> 已更新 {mpath}，模型权重: {path}")
    if suf:
        print("\n!! 提醒：数据为合成数据，以上指标不得写入报告  !!")


if __name__ == "__main__":
    main()
