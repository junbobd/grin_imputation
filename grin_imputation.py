#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动气象站缺测数据的智能插补 —— 轻量化 GRIN 变体（主程序）
================================================================
结构：站点空间图(residual 聚合 GCN) + 变量物理图(可学习边权) + 深度可分离膨胀时序卷积
数据：NASA POWER 逐小时站点多要素数据（免鉴权通道）
      NASA MERRA-2 逐小时（GES DISC OPeNDAP，需 Earthdata 账号，变量集更全）

============================ 设计要点 ============================
1. 【数据失败即终止，禁止静默降级】★ 最重要的原则 ★
   下载失败 -> 打印明确错误 + 排查建议 -> sys.exit(2)，不产生任何结果文件。
   理由：报告里的每个数字都必须来自真实数据。用合成数据跑出的 MAE 写进报告
   是学术不端，答辩时也会被问穿。
   如仅需跑通流程，必须显式加 --allow_synthetic，届时输出文件名带 .SYNTHETIC
   后缀、图标题打警告横幅，绝不与真实结果混淆。

2. 【数据源双通道】
   --source power_hourly : NASA POWER 逐小时，免鉴权、格式稳定（当前默认，实测可用）
   --source merra2       : MERRA-2 原生，变量集更全（含比湿、气压），
                           但需 Earthdata 账号 + 首次网页授权。
                           ⚠ OPeNDAP 的 ascii 解析尚未实现（故意占位，抛异常说明），
                           需拿到真实返回样本后才能补完，不得凭猜测写。

3. 【缓存带元信息指纹】
   cache 内存 data_signature（源/时段/站点/变量/版本），参数一变自动失效重下。
   杜绝"改了配置却读到旧数据"。

4. 【空间图用 residual 聚合（关键修复）】
   30 天实验的消融结果：原版 avg 聚合（h = A_s @ h）下行归一化后自环仅 0.274，
   即 72.6% 信息来自邻站。而 8 站相距 110~1056km、站间气温相关中位数 0.86，
   邻站新信息有限，聚合反而稀释本站信号 —— 早期协议（avg + 早停）下曾出现
   「去掉空间图 R² 反而更高」的假象，统一训练预算后消失。
   修复：residual 聚合 h = h + beta * (A_s @ h - h)，本站为主体、邻站只做增量修正，
   beta 可学习（邻站无用则自动压到 0，退化为无空间图）。
   终态（744h / 400 轮 / 关早停）：avg R² 0.6837 → residual 0.9593，
   同参数量 LSTM 0.8528；去空间图 0.9534（空间图已转为正贡献）。
   用 --spatial_mode avg 可复现旧行为，用于消融对照。

5. 【物理规律约束 = 可插拔注册表】
   PHYS_REGISTRY + @register_physics 装饰器 + --phys / --phys_weight 开关。
   当前启用 time_smooth（时间平滑）。diurnal_cycle / humidity_consistency /
   wind_nonneg 为函数骨架（默认权重 0，启用即 NotImplementedError），
   待消融实验验证后再决定是否纳入。

6. 【可解释性工程钩子】
   - --dump_aux          : 导出中间张量（embed 特征、变量图边权）供归因分析
   - model.record_aux()  : 统一中间量记录口
   - model.get_variable_graph() : 取学习到的变量物理图边权
   - model.get_spatial_beta()   : 取各层邻站注入强度 beta（residual 模式）
   - _plot_diurnal()     : 日变化相位诊断图（调参主依据）

7. 【置信区间不实现】
   原规划的"每个插补值附带置信区间"经评估属锦上添花，已决定从说明稿删除。
   本版不实现、不预留该模块。

8. 【基线为 3 个】线性插值 / KNN / 同参数规模 LSTM
   均值填充由本脚本内的 "mean-fill" 输出（标准化空间填 0 == 填训练集均值，
   命名如实标注，不叫 zero-fill）。不实现随机森林（用户明确限定 3 个）。

9. 【指标含 R²】evaluate() 返回 MAE / RMSE / R² 三项，对接报告要求。

============================ 快速开始 ============================
    python grin_imputation.py --source power_hourly \\
        --start 20240701 --end 20240730 \\
        --window 72 --stride 12 --block_maxlen 24 \\
        --epochs 300 --hidden 64 --n_layers 3 --tag d30

    消融对照：
    python grin_imputation.py ... --ablate_no_spatial   # 去站点空间图
    python grin_imputation.py ... --ablate_no_var       # 去变量图
    python grin_imputation.py ... --phys_ablate         # 去物理约束
    python grin_imputation.py ... --spatial_mode avg    # 用旧版平均聚合

⚠ 参数约束：block_maxlen 不得超过 window 的 50%（否则任务退化为"无中生有"，
   已加硬校验拦截）。窗口数不足 10 个也会直接报错并给出修复建议。
====================================================================
"""

import os, sys, json, math, time, argparse, hashlib
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================== 全局配置 =====================

# ---- 站点清单 ----
# 说明：我选的是 MERRA-2 格点能较好覆盖的华北平原，站点彼此距离 150-600 km。
#       站间距离必须和 sigma_km（高斯核带宽）匹配，否则邻接矩阵会全 0 或全 1。
STATIONS = {
    "Beijing":      (39.90, 116.40),
    "Tianjin":      (39.13, 117.20),
    "Shijiazhuang": (38.04, 114.51),
    "Jinan":        (36.67, 117.00),
    "Zhengzhou":    (34.75, 113.62),
    "Taiyuan":      (37.87, 112.55),
    "Xian":         (34.34, 108.94),
    "Qingdao":      (36.07, 120.38),
}

# ---- 变量清单（内部列名 -> 物理含义）----
# 挑这 6 个的理由：它们两两之间存在明确的物理耦合（能量平衡 / 热力学 / 水循环），
# 正是"多变量联合插补"能发挥作用的地方。
MERRA2_VARS = {
    "T2M":        ("T2M",   "2m 气温 (K)"),
    "QV2M":       ("QV2M",  "2m 比湿 (kg/kg)"),
    "U10M":       ("U10M",  "10m 纬向风 (m/s)"),
    "V10M":       ("V10M",  "10m 经向风 (m/s)"),
    "PS":         ("PS",    "地表气压 (Pa)"),
    "PRECTOTCORR":("PRECTOTCORR", "降水 (mm/hr)"),
    "SWGDN":      ("SWGDN", "地表下行短波辐射 (W/m2)"),
}

# ============================================================================
# ★★★ 两条数据通道的「列名 ↔ 实际变量」并不一一对应，务必看清 ★★★
# ============================================================================
# 内部列名（VARIABLES）是一套**与数据源无关的槽位名**，为的是让下游代码、
# 缓存指纹、模型结构在切换数据源时保持不变。但两条通道填进这些槽位的
# **实际变量和物理单位是不同的**：
#
#   槽位名         MERRA-2 通道                  NASA POWER 通道（当前实际使用）
#   ------------   ---------------------------   ---------------------------------
#   T2M            T2M      2m 气温    (K)       T2M                 2m 气温    (°C)
#   QV2M           QV2M     2m 比湿    (kg/kg)   RH2M  ★相对湿度★     (%)   ← 注意！
#   WS10M          U10M/V10M 合成风速 (m/s)     WS10M               10m 风速   (m/s)
#   PS             PS       地表气压  (Pa)      PS                  地表气压   (kPa) ← 注意！
#   PRECTOTCORR    PRECTOTCORR 降水   (mm/hr)   PRECTOTCORR         降水       (mm/hr)
#   SWGDN          SWGDN    下行短波辐射(W/m2)  ALLSKY_SFC_SW_DWN   全天空短波 (W/m2)
#
# 也就是说：**QV2M 这个槽位在 POWER 通道下装的是相对湿度，不是比湿**；
# PS 在 POWER 通道下是 kPa 不是 Pa；T2M 是 °C 不是 K。
# 报告、图例、指标表里必须按「实际变量」写，不能照抄槽位名。
#
# 为什么保留槽位名而不改名：改名会改动 _data_signature，导致全部缓存作废、
# 所有已完成的实验必须重跑，而数值本身不会变（数据完全相同）。
# 因此这里选择「元信息如实标注 + 文档按实际变量写」，而不是无意义地重跑。
# 若将来要彻底改名，需同步：VARIABLES / VARIABLES_POWER / VAR_LABEL /
# VAR_PHYS_UNIT_BY_SOURCE / 以及全部历史产物的说明。
# ============================================================================

VARIABLES     = ["T2M", "QV2M", "WS10M", "PS", "PRECTOTCORR", "SWGDN"]

# 槽位名 -> 各通道的实际变量名
VARIABLE_SOURCE_MAP = {
    "merra2":       ["T2M", "QV2M", "WS10M", "PS", "PRECTOTCORR", "SWGDN"],
    "power_hourly": ["T2M", "RH2M", "WS10M", "PS", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"],
}

# 各通道的实际物理单位（用于缓存元信息、图例、报告）
VAR_PHYS_UNIT_BY_SOURCE = {
    "merra2":       ["K",     "kg/kg", "m/s", "Pa",  "mm/hr", "W/m2"],
    "power_hourly": ["degC",  "%",     "m/s", "kPa", "mm/hr", "W/m2"],
}

# 各通道的实际变量标签（用于绘图图例）
VAR_LABEL_BY_SOURCE = {
    "merra2":       ["T2M (K)",    "QV2M (kg/kg)", "WS10M (m/s)", "PS (Pa)",
                     "PRECIP (mm/hr)", "SWGDN (W/m2)"],
    "power_hourly": ["T2M (degC)", "RH2M (%)",     "WS10M (m/s)", "PS (kPa)",
                     "PRECIP (mm/hr)", "SWGDN (W/m2)"],
}

# ---- 向后兼容别名：默认按 MERRA-2 通道，避免旧代码 import 报错 ----
VAR_LABEL     = VAR_LABEL_BY_SOURCE["merra2"]
VAR_PHYS_UNIT = VAR_PHYS_UNIT_BY_SOURCE["merra2"]

# ---- 缓存策略 ----
# CACHE_VERSION 一旦变动，所有旧缓存自动作废（因为物理含义可能已经改了）
CACHE_VERSION = "merra2_hourly_v1"
CACHE_PATH    = "raw_cache.npz"

# ===================== 数据获取：MERRA-2 逐小时 =====================

def _data_signature(start, end, stations, variables):
    """把数据请求的"身份证"算成指纹，用于判断缓存是否还能用。

    为什么要这个：原版只看 cache 文件在不在，参数改了照样读旧缓存，
    结果就是"我明明把时间范围改成 2024 了，跑出来的还是 2018-2023"。
    现在只要 start/end/站点/变量/缓存版本 任一变化，指纹就变，缓存自动失效。
    """
    key = "|".join([
        CACHE_VERSION,
        str(start), str(end),
        ",".join(sorted(stations.keys())),
        ",".join(variables),
    ])
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]


class DataSourceError(RuntimeError):
    """数据源不可用。这个异常是"终止程序"的信号，绝不允许被静默吞掉。

    设计意图：任何网络/鉴权/参数错误都抛这个异常，由调用方打印清晰指引后 exit。
    严禁在此异常被捕获后偷偷切到合成数据。
    """


def _load_earthdata_credentials():
    """读取 NASA Earthdata 登录凭据。

    GES DISC（MERRA-2 的托管方）要求所有请求带 Earthdata Login 认证。
    标准做法是在用户主目录放一个 .netrc 文件。这里同时支持环境变量写法，
    方便不想动 .netrc 的场合。

    返回 (username, password)，取不到返回 (None, None)。
    """
    u = os.environ.get("EARTHDATA_USERNAME")
    p = os.environ.get("EARTHDATA_PASSWORD")
    if u and p:
        return u, p
    try:
        from netrc import netrc
        auth = netrc().authenticators("urs.earthdata.nasa.gov")
        if auth:
            return auth[0], auth[2]
    except Exception:
        pass
    return None, None


# --------------------- MERRA-2 通道 A：GES DISC OPeNDAP（推荐） ---------------------
def fetch_merra2_opendap(lat, lon, start, end, credentials=None):
    """通过 NASA GES DISC 的 OPeNDAP 服务，按经纬度+时间范围取 MERRA-2 逐小时格点。

    调用方式参考 GES DISC 官方文档（subset 接口）：
        https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/...
    这里用的是服务端 subset 能力：只要给经纬度和时间范围，
    服务端就只返回目标格点的时序，不用下整个全球文件（全球文件单个 300MB+）。

    返回 (times: list[datetime], arr: np.ndarray[T, C])，任意失败抛 DataSourceError。
    """
    import requests
    from requests.auth import HTTPBasicAuth

    # MERRA-2 按变量拆在 4 个 collection 里，需要分别取再按时间对齐
    #  - tavg1_2d_slv_Nx : T2M, QV2M, U10M, V10M, PS  （单层诊断量）
    #  - tavg1_2d_lnd_Nx : PRECTOTCORR              （陆面诊断量）
    #  - tavg1_2d_rad_Nx : SWGDN                    （辐射诊断量）
    COLLECTION = {
        "tavg1_2d_slv_Nx": ["T2M", "QV2M", "U10M", "V10M", "PS"],
        "tavg1_2d_lnd_Nx": ["PRECTOTCORR"],
        "tavg1_2d_rad_Nx": ["SWGDN"],
    }
    BASE = "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2"

    # 时间范围格式化成 OPeNDAP 需要的 [start:end] 形式
    t0 = datetime.strptime(str(start), "%Y%m%d")
    t1 = datetime.strptime(str(end), "%Y%m%d") + timedelta(hours=23)

    auth = HTTPBasicAuth(*credentials) if credentials and credentials[0] else None
    if auth is None:
        raise DataSourceError(
            "缺少 NASA Earthdata 登录凭据，无法访问 GES DISC。\n"
            "  修复方式（三选一）：\n"
            "    1) 注册账号：https://urs.earthdata.nasa.gov/users/new\n"
            "    2) 在用户主目录创建 .netrc（推荐），内容：\n"
            "         machine urs.earthdata.nasa.gov\n"
            "         login  <你的用户名>\n"
            "         password  <你的密码>\n"
            "       Windows 放 C:\\Users\\<你>\\.netrc，注意文件不要带 .txt 后缀。\n"
            "    3) 设置环境变量 EARTHDATA_USERNAME / EARTHDATA_PASSWORD"
        )

    out_cols, ref_times = [], None
    for coll, vlist in COLLECTION.items():
        fname = _merra2_filename(coll, t0, t1)
        url = f"{BASE}/{coll}/{t1.year}/{t1.month:02d}/{fname}"
        # OPeNDAP 约束表达式：取最近格点 + 时间子集 + 需要的变量
        lat_i = _nearest_grid_index(lat, axis="lat")
        lon_i = _nearest_grid_index(lon, axis="lon")
        ce = (f"?{','.join(vlist)}"
              f"[{_hour_index(t0)}:1:{_hour_index(t1)}]"
              f"[0:0][{lat_i}:1:{lat_i}][{lon_i}:1:{lon_i}]")
        try:
            r = requests.get(url + ".ascii", auth=auth, timeout=120)
            if r.status_code == 401:
                raise DataSourceError(
                    "Earthdata 鉴权失败（HTTP 401）。\n"
                    "  排查：账号密码是否正确？是否已在 https://urs.earthdata.nasa.gov\n"
                    "        的 Applications 里授权了 'NASA GESDISC DATA ARCHIVE'？\n"
                    "        （首次使用必须在网页端手动同意一次，否则会一直 401）"
                )
            r.raise_for_status()
        except DataSourceError:
            raise
        except Exception as e:
            raise DataSourceError(
                f"下载 MERRA-2 集合 {coll} 失败：{type(e).__name__}: {e}\n"
                f"  请求 URL：{url}\n"
                "  排查：① 网络能否访问 gesdisc.eosdis.nasa.gov？\n"
                "        ② 是否需要走代理（国内直连较慢，建议设 HTTPS_PROXY）？\n"
                "        ③ 日期是否落在该集合的有效期内？"
            ) from e
        _parse_ascii_grid(r.text, out_cols, vlist, ref_times)  # 见下方解析函数

    # 组装：MERRA-2 的 WS10M 需要由 U10M/V10M 合成（矢量平均的标量风速）
    return _assemble_merra2_columns(out_cols, ref_times)


def _merra2_filename(coll, t0, t1):
    """拼 MERRA-2 的 OPeNDAP 文件名（不同集合文件名规则不同）。"""
    y, m, d = t0.year, t0.month, t0.day
    if coll == "tavg1_2d_slv_Nx":
        return f"MERRA2_{y}{m:02d}{d:02d}.tavg1_2d_slv_Nx.ana.nc4"
    if coll == "tavg1_2d_lnd_Nx":
        return f"MERRA2_{y}{m:02d}{d:02d}.tavg1_2d_lnd_Nx.ana.nc4"
    if coll == "tavg1_2d_rad_Nx":
        return f"MERRA2_{y}{m:02d}{d:02d}.tavg1_2d_rad_Nx.ana.nc4"
    raise DataSourceError(f"未知的 MERRA-2 集合：{coll}")


def _nearest_grid_index(value, axis):
    """经纬度 -> MERRA-2 格点索引。
    MERRA-2 网格：纬度 0.5°（-90 到 90，共 361 点，从北向南），
                  经度 0.625°（-180 到 179.375，共 576 点，从西向东）。
    """
    if axis == "lat":
        return int(round((90.0 - value) / 0.5))
    return int(round((value + 180.0) / 0.625))


def _hour_index(dt):
    """日期 -> 该年内的小时序号（0 起）。"""
    return int((dt - datetime(dt.year, 1, 1)).total_seconds() // 3600)


def _parse_ascii_grid(text, out_cols, vlist, ref_times):
    """解析 OPeNDAP .ascii 输出。

    OPeNDAP 的 ascii 格式对每个变量给一个二维数组块。这里做的是最朴素的解析：
    把每行按空白切开，收集数值，再按变量顺序切成 T 长度的一列列。
    真实运行时格式若与预期不符，会抛 DataSourceError 而不是猜。
    """
    raise DataSourceError(
        "MERRA-2 OPeNDAP ascii 解析未实现（占位）。\n"
        "  原因：GES DISC 的 ascii 输出格式随服务端版本变化，我不想在没实测的情况下\n"
        "        写一段'看起来能解析'的代码来骗你。\n"
        "  你需要做的：先手动跑一次 requests，把返回的文本贴给我（或看 --debug_dump），\n"
        "        我按真实格式把解析写完。\n"
        "  或者：改用通道 B（--source power_hourly，无需鉴权、格式稳定、立即可用）。"
    )


def _assemble_merra2_columns(out_cols, ref_times):
    """把各集合取到的列拼成 (T, C) 数组，并做物理量换算。

    换算说明（重要，报告里要写清楚）：
      - WS10M = sqrt(U10M^2 + V10M^2)  —— 风矢量合成标量风速
      - T2M 单位 K，QV2M 单位 kg/kg，PS 单位 Pa，均保持 MERRA-2 原生单位
    """
    raise DataSourceError("MERRA-2 列组装未实现（同 _parse_ascii_grid 原因）。")


# --------------------- MERRA-2 通道 B：NASA POWER hourly（免鉴权兜底） ---------------------
def fetch_power_hourly(lat, lon, start, end, retries=3):
    """NASA POWER API 的逐小时接口 —— 与 MERRA-2 同源，但无需注册、格式稳定。

    为什么保留这条通道：MERRA-2 的 GES DISC 需要账号 + 首次网页授权，
    在国内还可能遇到网络问题。POWER hourly 是同一套 NASA 数据体系的
    服务化封装，能立刻跑通，方便你先看原型效果。

    注意数据实质差异（报告里必须如实说明）：
      - POWER hourly 本质是 MERRA-2 的服务化产品，但只暴露部分变量；
      - 气压只给地表气压/海平面气压，湿度为相对湿度而非比湿；
      - 因此"多变量联合"的可用变量集合比 MERRA-2 原生略窄。

    返回 (dates: list[str], arr: np.ndarray[T, C])，失败抛 DataSourceError。
    """
    import requests
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = dict(
        parameters=",".join(VARIABLES_POWER),
        community="RE",
        longitude=lon, latitude=lat,
        start=f"{str(start)[:4]}{str(start)[4:6]}{str(start)[6:]}",
        end=f"{str(end)[:4]}{str(end)[4:6]}{str(end)[6:]}",
        format="JSON",
    )
    last = None
    for k in range(retries):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code != 200:
                raise DataSourceError(
                    f"NASA POWER 返回 HTTP {r.status_code}。\n"
                    f"  请求参数：{params}\n"
                    "  排查：① 参数名是否拼错（hourly 接口的 parameters 列表与 daily 不同）？\n"
                    "        ② 日期是否超出该接口范围（hourly 一般只覆盖 2001 年之后）？"
                )
            js = r.json()
            if "properties" not in js or "parameter" not in js["properties"]:
                raise DataSourceError(
                    "NASA POWER 返回体里没有 properties.parameter 字段，"
                    "很可能是参数不合法被服务端拒绝。\n"
                    f"  原始返回前 500 字符：{str(js)[:500]}"
                )
            par = js["properties"]["parameter"]
            times = sorted(par[VARIABLES_POWER[0]].keys())
            out = np.full((len(times), len(VARIABLES_POWER)), np.nan, np.float32)
            for ci, v in enumerate(VARIABLES_POWER):
                ser = par[v]
                out[:, ci] = [ser.get(t, np.nan) for t in times]
            # POWER 的填充值约定：<= -900 视为缺测
            out[out <= -900] = np.nan
            return times, out
        except DataSourceError:
            raise
        except Exception as e:
            last = e
            print(f"    [重试 {k+1}/{retries}] {type(e).__name__}: {e}")
            time.sleep(3)
    raise DataSourceError(
        f"NASA POWER 逐小时接口连续 {retries} 次请求失败，最后一次异常：{last}\n"
        "  排查：① 本机能否 ping 通 power.larc.nasa.gov？\n"
        "        ② 是否在公司/学校内网被限制，需要设置 HTTPS_PROXY 环境变量？\n"
        "        ③ 先浏览器直接打开 https://power.larc.nasa.gov/ 验证网络。"
    )


# POWER hourly 接口的变量名（与 MERRA-2 原名不同，需要映射）
VARIABLES_POWER = ["T2M", "RH2M", "WS10M", "PS", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"]


# --------------------- 统一入口 ---------------------
def load_raw_data(args):
    """加载原始数据。缓存命中就直接读，否则下载。

    ★核心原则（本函数最重要的契约）★
      下载失败 -> 抛 DataSourceError -> 调用方打印指引并 exit(1)。
      绝不允许"下载失败就换合成数据"。
      唯一例外：用户显式传了 --allow_synthetic，此时才走合成分支，
      且输出文件名会带 .SYNTHETIC 标记、图上会打警告横幅。
    """
    sig = _data_signature(args.start, args.end, STATIONS, VARIABLES)

    # ---- 1. 缓存命中判定：不仅看文件在不在，还要看指纹对不对 ----
    if os.path.exists(CACHE_PATH) and not args.refresh:
        try:
            d = np.load(CACHE_PATH, allow_pickle=True)
            if "meta" in d and str(d["meta"].item().get("data_signature", "")) == sig:
                meta = d["meta"].item()
                print(f">> 缓存命中 {CACHE_PATH}")
                print(f"   数据源: {meta.get('source')} | 分辨率: {meta.get('resolution')}")
                print(f"   时段: {meta.get('start')} ~ {meta.get('end')} | "
                      f"下载于 {meta.get('downloaded_at')}")
                print(f"   形状: {d['X'].shape}  (T, N, C)")
                return d["X"], d["M"], meta
            else:
                print(">> 缓存指纹不匹配（参数已变），将重新下载")
        except Exception as e:
            print(f">> 缓存读取失败（{type(e).__name__}: {e}），将重新下载")

    # ---- 2. 下载 ----
    print(f">> 开始下载 {args.source} 逐小时数据 ...")
    print(f"   站点数 {len(STATIONS)} | 变量 {len(VARIABLES)} | 时段 {args.start}~{args.end}")
    mats, all_times = [], None
    for name, (lat, lon) in STATIONS.items():
        if args.source == "merra2":
            times, arr = fetch_merra2_opendap(
                lat, lon, args.start, args.end, _load_earthdata_credentials())
        else:
            times, arr = fetch_power_hourly(lat, lon, args.start, args.end)
        if arr.shape[1] != len(VARIABLES):
            raise DataSourceError(
                f"站点 {name} 返回 {arr.shape[1]} 列，与 VARIABLES 的 {len(VARIABLES)} 列不符。\n"
                "  说明数据源变量清单与脚本配置不一致，需要在代码里同步修改，"
                "不允许用列数对齐的方式糊过去。"
            )
        print(f"   {name:<14} {arr.shape}  缺测率 {np.isnan(arr).mean():.3%}")
        mats.append(arr)
        all_times = times if all_times is None else all_times

    T = min(m.shape[0] for m in mats)
    X = np.stack([m[:T] for m in mats], 1)          # (T, N, C)
    M = (~np.isnan(X)).astype(np.float32)

    # ★ 真实性检查：真实数据必然有缺测，全 0 缺测说明数据有问题，要报出来
    miss = 1 - M.mean()
    print(f">> 下载完成：T={T} 小时 ({T/24:.1f} 天) | 原始缺测率 {miss:.3%}")
    if miss < 1e-6:
        print("   [警告] 原始缺测率恰为 0，真实观测数据极少出现这种情况，"
              "请确认数据源返回的填充值约定是否与本脚本一致。")
    if T < args.window * 3:
        raise DataSourceError(
            f"有效时间长度 T={T} 小时，不足窗口长度 {args.window} 的 3 倍，"
            "无法切出足够训练样本。\n"
            "  修复：放宽 --start/--end 时间范围，或调小 --window。"
        )

    # ---- 3. 写缓存（带完整元信息）----
    # ★ 注意：units / variable_labels 必须按**实际数据源**取，不能写死 MERRA-2 的那套。
    #   历史上这里写死过 VAR_PHYS_UNIT，导致 power_hourly 缓存把 RH2M 标成 kg/kg、
    #   把 kPa 标成 Pa、把 degC 标成 K —— 元信息与实际数值不符，属数据诚实性缺陷。
    _src = args.source if args.source in VAR_PHYS_UNIT_BY_SOURCE else "merra2"
    meta = {
        "source": args.source,
        "resolution": "hourly",
        "start": str(args.start), "end": str(args.end),
        "stations": list(STATIONS.keys()),
        "variables": VARIABLES,
        "actual_variables": VARIABLE_SOURCE_MAP[_src],
        "units": VAR_PHYS_UNIT_BY_SOURCE[_src],
        "variable_labels": VAR_LABEL_BY_SOURCE[_src],
        "slot_note": ("variables 为与数据源无关的槽位名；actual_variables 才是本缓存里"
                      "每个槽位实际装的变量，units 与之逐一对应。"),
        "data_signature": sig,
        "cache_version": CACHE_VERSION,
        "downloaded_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "shape": list(X.shape),
        "raw_missing_rate": float(miss),
    }
    np.savez(CACHE_PATH, X=np.nan_to_num(X, nan=0.0), M=M, meta=meta)
    print(f">> 已写入缓存 {CACHE_PATH}（含来源元信息）")
    return np.nan_to_num(X, nan=0.0), M, meta


def synthetic_data(args, reason: str):
    """合成数据 —— 仅在用户显式 --allow_synthetic 时被调用。

    这个函数里有两点刻意的设计，防止"假数据混进真结果"：
      1. 名字和调用点都写死 synthetic，全局搜索一眼能找到；
      2. 返回的 meta 里 synthetic=True，下游据此给文件名加 .SYNTHETIC 后缀。
    """
    banner = "!" * 78
    print(f"\n{banner}")
    print("!!  警告：正在使用【合成数据】—— 所有指标均不代表真实性能  !!")
    print("!!  原因：")
    for line in reason.strip().splitlines():
        print(f"!!    {line}")
    print("!!  本结果禁止写入任何比赛/论文/报告材料。")
    print(f"{banner}\n")
    rng = np.random.default_rng(args.seed)
    T = 24 * 365 * 5                                    # 5 年逐小时
    t = np.arange(T)
    hour = t % 24
    doy = (t // 24) % 365
    season = np.sin(2 * np.pi * (doy - 105) / 365)
    diurnal = np.sin(2 * np.pi * (hour - 9) / 24)       # 日变化相位：午后最热
    shared = rng.normal(0, 1, T)
    X = np.zeros((T, len(STATIONS), len(VARIABLES)), np.float32)
    for n, (name, (lat, lon)) in enumerate(STATIONS.items()):
        X[:, n, 0] = (288 + 14 * season - 0.15 * (lat - 35) * 0.1) + 6 * diurnal + rng.normal(0, 1.5, T)
        X[:, n, 1] = np.clip(0.006 * np.exp(-0.05 * (lat - 35)) * (1 - 0.3 * season) + rng.normal(0, 5e-4, T), 1e-5, None)
        X[:, n, 2] = np.clip(3 + 1.2 * np.abs(np.sin(2 * np.pi * doy / 180)) + rng.normal(0, 0.8, T), 0.1, None)
        X[:, n, 3] = 98000 + 800 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 60, T)
        X[:, n, 4] = np.clip(rng.gamma(0.6, 0.4, T) * np.clip(season + 0.6, 0, 1.4), 0, None)
        X[:, n, 5] = np.clip(220 * (0.1 + 0.9 * np.clip(season, 0, 1)) * np.clip(diurnal + 0.6, 0, 1.4) + rng.normal(0, 10, T), 0, 1400)
        X[:, n] += 0.1 * shared[:, None]
    M = np.ones_like(X, np.float32)
    meta = {
        "source": "SYNTHETIC", "resolution": "hourly",
        "start": str(args.start), "end": str(args.end),
        "stations": list(STATIONS.keys()), "variables": VARIABLES,
        "units": VAR_PHYS_UNIT, "data_signature": "synthetic",
        "cache_version": CACHE_VERSION,
        "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "shape": list(X.shape), "raw_missing_rate": 0.0,
        "synthetic": True, "synthetic_reason": reason,
    }
    return X, M, meta


def resolve_data(args):
    """数据入口：真实数据优先，失败就明确报错退出；只有显式开关才允许合成。"""
    try:
        return load_raw_data(args)
    except DataSourceError as e:
        if args.allow_synthetic:
            return synthetic_data(args, reason=str(e))
        # ★ 这里就是"直接输出网络异常，退出程序"
        print("\n" + "=" * 78, file=sys.stderr)
        print("【数据获取失败 · 程序终止】", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        print(str(e), file=sys.stderr)
        print("-" * 78, file=sys.stderr)
        print("程序已终止，未产生任何结果文件。", file=sys.stderr)
        print("本次运行没有输出合成数据、没有沿用旧缓存、没有降级。", file=sys.stderr)
        print("如仅需跑通流程，可显式加 --allow_synthetic（结果会带 .SYNTHETIC 标记）。",
              file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        sys.exit(2)


def fill_missing_args(args):
    """补全主脚本函数会读取、但调用方 argparse 可能未定义的字段。

    ★ 为什么需要这个 ★
      `build_station_graph` / `run_epoch` / `predict_all` 这些函数会直接访问
      args 的若干属性。跨脚本复用（实验脚本、基线脚本）时，只要调用方的
      argparse 少定义一个字段，就会抛 AttributeError。
      更坑的是：如果补全代码写在调用**之后**，错误依然会触发 —— 因为函数在
      调用时就已经读了。所以约定：**所有跨脚本复用的脚本，parse_args 之后
      第一件事就是调用 fill_missing_args(args)**，再碰任何主脚本函数。

    补的都是"关闭/默认"值。需要启用的脚本自己覆盖即可。
    """
    defaults = {
        "ablate_no_spatial": False,
        "ablate_no_var": False,
        "phys": [],
        "phys_weights": {},
        "phys_ablate": False,
        "diurnal_weight_per_var": None,
        "norm_mean": None,
        "norm_std": None,
        "return_aux": False,
        "dump_aux": False,
    }
    missing = [k for k in defaults if not hasattr(args, k)]
    for k in missing:
        setattr(args, k, defaults[k])
    return args, missing


# ===================== 邻接图构建 =====================
def haversine(lat1, lon1, lat2, lon2):
    """两点球面大圆距离（km）。"""
    R, p1, p2 = 6371.0, math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def build_station_graph(args, coords, device):
    """空间图：站点经纬度距离 -> 高斯核邻接矩阵。

    ★ 关键诊断：自环权重（self-loop weight）★
      本函数最后做了行归一化，使每行权重和为 1（等价于加权平均聚合）。
      归一化后对角线元素 = 该站点在聚合后特征里**自己占的比例**。

      这个比例太小时会出大问题（30 天实验实测踩过）：
        自环 0.274 → 每次聚合 72.6% 的信息来自邻站，本站信号被稀释。
        而 8 站相距 110~1056km、站间气温相关中位数 0.86，邻站新信息有限。
        在 avg 聚合 + 早停的旧协议下曾因此误判「去掉空间图反而更好」；
        改用 residual 并统一训练预算后，终态 R² 0.9593、去空间图 0.9534，
        空间图已转为正贡献。

      所以这里直接打印自环占比，并在过低时给出明确警告和修复方向。
      **自环 < 0.5 就该警惕**：说明聚合主导了特征，本站信息被淹没。
    """
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = haversine(*coords[i], *coords[j])
    A = np.exp(-D ** 2 / (2 * args.sigma_km ** 2))
    A[A < 0.15] = 0.0                      # 弱边截断，避免全连接噪声
    np.fill_diagonal(A, 1.0)
    A = A / A.sum(1, keepdims=True)        # 行归一化（相当于加权平均聚合）

    off = A[~np.eye(n, dtype=bool)]
    nz = (off > 0).mean()
    self_w = float(np.diag(A).mean())
    print(f">> 站点图: sigma={args.sigma_km:.0f}km | 非零边占比 {nz:.1%} | "
          f"站间距离 {D[D>0].min():.0f}~{D.max():.0f}km")
    print(f"   自环权重均值 {self_w:.3f} → 聚合后本站信息占 {self_w:.1%}，"
          f"邻站占 {1-self_w:.1%}")
    if nz < 0.10:
        print("   [警告] 非零边过少，空间图接近孤岛，sigma_km 建议调大")
    elif nz > 0.95:
        print("   [警告] 几乎全连接，空间图接近简单平均，sigma_km 建议调小")
    if self_w < 0.5:
        print(f"   [警告] 自环权重仅 {self_w:.3f}，本站信号会被邻站稀释。"
              "若站点间要素走势高度同步（如短窗口内的日变化信号），"
              "这会显著损害性能。\n"
              "     建议：改用 --spatial_mode residual（本站为主体 + 邻站增量，"
              "beta 可学习），或调小 --sigma_km。")

    if args.ablate_no_spatial:
        A = np.eye(n)
        print("   [消融] 已关闭空间图（退化为各站点独立建模）")
    return torch.tensor(A, dtype=torch.float32, device=device)


def build_variable_graph_init(Xtr_norm, signed=False):
    """变量图初始化：训练段变量间 Pearson 相关。

    signed=False（默认）：取绝对值，边权恒非负，配合 softplus + 行归一化
        构成凸组合，聚合是"加权平均"。
    signed=True：保留正负号，负相关以负权重进入聚合，归一化改按绝对值行和
        （见 GRINLite.norm_av），聚合是"加权和"。用于检验边权符号是否有增益。

    物理含义：这是"物理相关先验"——温度与比湿正相关、气压与温度弱相关等，
    由数据自己说出来。训练中边权可继续调整，即"数据驱动地修正物理先验"。

    ★ 数据量不足时会出问题 ★
      相关系数需要多个独立样本才能估计。当训练窗口很少（或某变量方差为 0）时
      np.corrcoef 会产生 NaN（numpy 会同时抛 divide-by-zero / invalid-value 警告）。
      这里做两件事：① 抑制无意义的 numpy 警告 ② 把 NaN 兜底为单位阵，
      即"没有先验信息时退化为各变量独立"，而不是让 NaN 污染整个模型。
    """
    flat = Xtr_norm.reshape(-1, Xtr_norm.shape[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(flat.T)
    corr = np.nan_to_num(corr if signed else np.abs(corr), nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return torch.tensor(corr + 1e-3, dtype=torch.float32)


# ===================== 物理规律约束：可插拔注册表 =====================
# ★ 设计意图（为后续实验预留）★
#   物理约束是"锦上添花"还是"真有用"，必须靠实验说话。所以这里做成注册表：
#   主干代码只认名字，不认具体公式。将来加新约束 = 写一个函数 + 注册一行，
#   不用动 run_epoch、不用动模型定义，也不会影响已有实验的可复现性。
PHYS_REGISTRY = {}


def register_physics(name, default_weight=0.0, note=""):
    """装饰器：把一个物理约束函数登记进注册表。"""
    def deco(fn):
        PHYS_REGISTRY[name] = {"fn": fn, "default_weight": default_weight, "note": note}
        return fn
    return deco


@register_physics("time_smooth", default_weight=0.05,
                  note="时间平滑：惩罚插补值在相邻时刻的剧烈跳变，抑制不合理毛刺")
def phys_time_smooth(pred, tgt, m_eval, ctx):
    """时间平滑约束（当前唯一启用的物理约束）。

    物理依据：除降水这类突变过程外，气温/气压/湿度在 1 小时尺度上的
    变化率有物理上限，不会出现分钟级的剧烈跳变。这里只对"缺口处"的
    相邻两点做惩罚，观测点不参与（观测点本来就以真实值为主）。
    """
    diff = (pred[:, 1:] - pred[:, :-1]).abs()
    pm = m_eval[:, 1:] * m_eval[:, :-1]
    return (diff * pm).sum() / (pm.sum() + 1e-8)


@register_physics("diurnal_cycle", default_weight=0.0,
                  note="TODO 日变化周期：约束插补序列的 24h 周期形状接近气候态正弦")
def phys_diurnal_cycle(pred, tgt, m_eval, ctx):
    """日变化周期约束 —— 【骨架，未做实验，默认权重 0】。

    后续实验思路（逐小时数据才成立，这也是换数据源的收益之一）：
      1. 用训练段统计每个 (站点, 变量, 小时) 的气候态均值，得到 C[n, c, 24]；
      2. 对插补结果按 24h 分箱求均值，与气候态做 L2；
      3. 该约束对气温、辐射有效，对降水很可能无效甚至有害 ——
         所以必须分变量给权重，不能一刀切。
    变量级权重的入口：ctx["diurnal_weight_per_var"]。
    """
    raise NotImplementedError(
        "日变化周期约束尚未实现（骨架）。\n"
        "  这是刻意的：该约束是否有效必须靠消融实验验证，不能先写死再找理由。\n"
        "  启用前提：① --phys diurnal_cycle ② 数据为逐小时 ③ 已做过时间平滑的消融作为对照。"
    )


@register_physics("humidity_consistency", default_weight=0.0,
                  note="TODO 干湿一致性：温度与比湿的耦合关系（Clausius-Clapeyron 方向）")
def phys_humidity_consistency(pred, tgt, m_eval, ctx):
    """干湿一致性约束 —— 【骨架，未做实验，默认权重 0】。

    后续实验思路：气温升高时饱和水汽压按 Clausius-Clapeyron 关系指数上升，
    比湿的气候态也应随温度单调变化。可用"插补点落在 (T, Q) 联合分布合理区域内"
    作为软约束。风险：真实大气受平流影响显著，(T,Q) 关系相当发散，
    硬约束容易把模型带偏，建议只做弱惩罚并单独消融。
    """
    raise NotImplementedError("干湿一致性约束尚未实现（骨架），启用前需先完成消融实验设计。")


@register_physics("wind_nonneg", default_weight=0.0,
                  note="TODO 风速非负：标量风速物理上不可能为负")
def phys_wind_nonneg(pred, tgt, m_eval, ctx):
    """风速非负约束 —— 【骨架，未做实验，默认权重 0】。

    注：本版模型输出的是标准化空间的值，非负约束必须放回物理单位才成立。
    正确做法是在 ctx 里传入 mean/std，先反标准化再惩罚。这个细节说明
    "物理约束不能随手加"，必须考虑模型内部的归一化约定。
    """
    raise NotImplementedError("风速非负约束尚未实现（骨架），注意需在物理单位下施加。")


def compute_physics_loss(pred, tgt, m_eval, names, weights, ctx):
    """按名字调用注册表里的物理约束，返回加权和 + 逐项明细（便于消融对比）。

    weights 支持两种写法（这是为"分变量给权重"预留的）：
      - 标量：  {"time_smooth": 0.05}
      - 按变量：{"diurnal_cycle": {"T2M": 0.1, "PRECTOTCORR": 0.0}}
    """
    total = pred.new_zeros(())
    detail = {}
    for nm in names:
        if nm not in PHYS_REGISTRY:
            raise KeyError(f"未知物理约束 '{nm}'，可选：{list(PHYS_REGISTRY)}")
        w = weights.get(nm, PHYS_REGISTRY[nm]["default_weight"])
        if isinstance(w, dict):
            w = float(np.mean(list(w.values()))) if w else 0.0
        if w == 0.0:
            detail[nm] = 0.0
            continue
        v = PHYS_REGISTRY[nm]["fn"](pred, tgt, m_eval, ctx)
        total = total + w * v
        detail[nm] = float(v.detach()) if torch.is_tensor(v) else float(v)
    return total, detail


# ===================== 模型：轻量 GRIN 变体 =====================
class DSC(nn.Module):
    """深度可分离膨胀时序卷积（替代原 GRIN 的循环模块）。

    深度可分离 = depthwise（每通道独立卷积，抓时序）+ pointwise（1x1 混合通道）。
    参数量约为普通卷积的 1/H，这正是"轻量化"的着力点。
    """

    def __init__(self, h, k, dil):
        super().__init__()
        self.dw = nn.Conv1d(h, h, k, padding=dil * (k - 1) // 2, dilation=dil, groups=h)
        self.pw = nn.Conv1d(h, h, 1)
        self.gn = nn.GroupNorm(4, h)

    def forward(self, x):                                   # (B*N, H, L)
        return F.gelu(self.gn(self.pw(self.dw(x))) + x)     # 残差，缓解深网梯度问题


class STBlock(nn.Module):
    """时空块：先空间图聚合邻站信息，再时序卷积提取本地演化。

    ★ 空间聚合的两种模式（这是消融实验定位出的关键设计点）★

    起因：30 天实验做了空间图消融，早期协议（avg + 早停）下曾出现
    **去掉空间图反而更好**的假象，诊断后确认根因：

        行归一化后的邻接矩阵，自环权重均值仅 0.274，
        即每次聚合 **72.6% 的信息来自其他站点**，本站在特征里只占 27.4%。

    而我们选的 8 个站相距 110~1056km，72h 窗口内各站温度由日变化主导、
    站间气温相关中位数 0.86。这种情况下邻站新信息有限，只会把本站的信号稀释掉。
    模型要补 Beijing 的缺口，本该主要看 Beijing 自己的历史，
    结果 73% 的注意力被拉到天津、石家庄 —— 这是纯粹的信息损失。
    统一训练预算（400 轮、关早停）后该假象消失：avg R² 0.6837、
    residual 0.9593、去空间图 0.9534，空间图已转为正贡献。

    mode="avg"（原版，默认保持兼容）：
        标准 GCN 加权平均，h = A_s @ h。适合邻站信息确有互补性的场景
        （如长缺口跨站推断、站点密集的气象观测网）。

    mode="residual"（推荐）：
        h = h + beta * (A_s @ h - h)，其中 beta 是可学习的标量（初始 0.1）。
        即"本站特征为主体，邻站只做增量修正"。beta 可学习意味着：
        **如果邻站真没用，模型会把 beta 压到 0，自动退化为无空间图**；
        如果邻站有用（长缺口场景），模型会把 beta 调大。
        这比手工选择"用/不用空间图"更稳健 —— 让数据决定。

    用法：--spatial_mode avg|residual
    """
    def __init__(self, h, k, dil, mode="avg"):
        super().__init__()
        self.dsc = DSC(h, k, dil)
        self.mode = mode
        if mode == "residual":
            # 可学习的邻站注入强度。初始 0.1 表示"先假设邻站作用很小"，
            # 避免训练初期被稀释；若邻站有用，梯度会把它推上去。
            self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, h, A_s):                              # h: (B,L,N,H)
        agg = torch.einsum("mn,blnh->blmh", A_s, h)          # 邻站聚合
        if self.mode == "residual":
            h = h + self.beta * (agg - h)                   # 残差式：以本站为主
        else:
            h = agg                                          # 原版：加权平均
        B, L, N, H = h.shape
        u = h.permute(0, 2, 3, 1).reshape(B * N, H, L)
        u = self.dsc(u)                                     # 深度可分离时序卷积
        return u.reshape(B, N, H, L).permute(0, 3, 1, 2).contiguous()


class GRINLite(nn.Module):
    """轻量 GRIN：变量物理图 + 站点空间图 + 膨胀卷积时序编码器。"""

    def __init__(self, C, hidden, n_layers, k, A_s, A_v_init, spatial_mode="avg",
                 signed_graph=False, var_out_mask=True):
        super().__init__()
        self.register_buffer("A_s", A_s)
        self.A_v = nn.Parameter(A_v_init.clone())           # 变量图边权：物理相关初始化、可学习
        self.C = C
        self.spatial_mode = spatial_mode
        self.signed_graph = signed_graph                    # True=保留正负号（负相关为负权重）
        # var_out_mask=True：缺口位置的变量图特征置 0（缺口位置三路输入全零，
        #   图信息只在观测位置起作用）。
        # var_out_mask=False：缺口位置保留"邻居观测值的加权和"，即跨要素推断值，
        #   这才是变量图对插补真正有贡献的通道（--no_var_out_mask 开启）。
        self.var_out_mask = var_out_mask
        self.embed = nn.Linear(3 * C, hidden)               # 输入 = [观测值, 掩码, 变量图聚合]
        self.blocks = nn.ModuleList(
            [STBlock(hidden, k, 2 ** i, spatial_mode) for i in range(n_layers)])
        self.head = nn.Linear(hidden, C)
        self.ablate_var = False
        # 可解释性钩子：开启后 forward 会把中间张量存进来
        self.return_aux = False
        self._aux = {}

    def get_spatial_beta(self):
        """取各层学到的邻站注入强度 beta（仅 residual 模式有）。

        可解释性价值：beta 趋近 0 说明"邻站对本任务无用"，
        beta 明显大于 0 说明"邻站确有信息贡献"。这是判断空间图是否该保留的
        直接证据，比只看指标更能说明问题。
        """
        return [float(b.beta.detach()) for b in self.blocks
                if getattr(b, "beta", None) is not None]

    # ---- 可解释性工程钩子 ----
    def get_variable_graph(self):
        """取出归一化后的变量图边权。出图 / 论文表格直接用这个。"""
        return self.norm_av().detach().cpu().numpy()

    def get_station_graph(self):
        return self.A_s.detach().cpu().numpy()

    def clear_aux(self):
        self._aux = {}

    def record_aux(self, key, value):
        """统一中间量记录口。后续做注意力可视化、逐变量归因都从这里取。"""
        self._aux[key] = value

    def get_aux(self, key=None):
        return self._aux if key is None else self._aux.get(key)

    def norm_av(self):
        if self.signed_graph:
            # 保留正负号：按绝对值行和归一化，聚合为加权和（可表达"反向关联"）
            Av = self.A_v
            return Av / (Av.abs().sum(-1, keepdim=True) + 1e-6)
        Av = F.softplus(self.A_v)                           # 保证边权非负
        return Av / (Av.sum(-1, keepdim=True) + 1e-6)

    def forward(self, x, m):                                # x,m: (B,L,N,C)
        Av = (torch.eye(self.C, device=x.device) if self.ablate_var
              else self.norm_av())
        if self.return_aux:
            self.record_aux("variable_graph", Av.detach())
        # ★ 变量物理图消息传递：只用【观测值】聚合 ★
        #   原写法 einsum(Av, x) 用完整真值聚合，缺口真值会经变量图流入可见要素，
        #   再经时序卷积/空间聚合流回缺口 —— 训练与评测时都"偷看"了标签。
        #   实测（已训练权重）：缺口真值 +3.0 → 观测位置输出平均变化 0.256。
        #   乘 m 后缺口位置以 0 参与加权，输出端的 xv * m 保持原语义不变。
        xv = torch.einsum("uv,blnu->blnv", Av, x * m)       # 只用观测值聚合，堵住标签泄漏
        if self.var_out_mask:
            xv = xv * m                                     # 缺口位置图特征置零
        h = self.embed(torch.cat([x * m, m, xv], -1))       # [观测值, 掩码, 图聚合特征]
        if self.return_aux:
            self.record_aux("embed", h.detach())
        for i, blk in enumerate(self.blocks):
            h = blk(h, self.A_s)
            if self.return_aux:
                self.record_aux(f"block{i}", h.detach())
        return self.head(h)                                 # (B,L,N,C)


# ===================== 缺失模拟与窗口切片 =====================
def make_eval_mask(m_obs, rng, point_rate, block_prob, max_len):
    """MCAR 随机缺失 + 连续缺测段（模拟真实观测中断场景）。

    两种缺失模式都要有：
      - MCAR：单点随机缺测，考察"插值"能力；
      - block：连续缺测段，考察"外推/上下文推断"能力 —— 后者才是真实
        设备故障/通信中断的形态，也是基线方法最容易崩的地方。

    ★ 关键约束：block_maxlen 必须由调用方按窗口长度的比例给出 ★
      如果 max_len 接近甚至超过窗口长度 L，会出现"整个窗口只锚定 1-2 个观测点"
      的退化样本，此时任务本质变成"从极少信息外推整条曲线"，
      任何模型都会退化成一条直线 —— 这不是模型能力问题，是任务设置失真。
      调用方须保证 max_len <= 0.5 * L（docx 的缺失上限 30% 也在此约束内）。
    """
    L = m_obs.shape[1]
    if max_len > 0.5 * L:
        raise ValueError(
            f"block_maxlen={max_len} 超过窗口长度 {L} 的 50%。\n"
            "  这样会造出「一个窗口里几乎全是缺口」的退化样本，任务从"
            "「插补」变成了「无中生有」，任何模型都会退化成直线。\n"
            f"  修复：把 --block_maxlen 降到 {int(0.5 * L)} 以内，或把 --window 调大。\n"
            f"  建议：缺口长度与窗口长度保持 1:3 以上（如 window=72h, block=24h）。"
        )
    M_eval = rng.random(m_obs.shape) < point_rate
    S, L, N, C = m_obs.shape
    for s in range(S):
        for n in range(N):
            for c in range(C):
                if rng.random() < block_prob:
                    l0 = rng.integers(0, L - 3)
                    ln = int(rng.integers(2, max_len + 1))
                    M_eval[s, l0:l0 + ln, n, c] = True
    return (M_eval & (m_obs > 0.5)).astype(np.float32)


def make_windows(X, M, window, stride):
    """把长时序切成滑窗样本。

    返回 (Xw, Mw, start_idx)：
      - Xw: (S, L, N, C) 窗口数据
      - Mw: (S, L, N, C) 观测掩码
      - start_idx: (S,) 每个窗口在原始时序里的起始时刻下标

    ★ start_idx 非常关键，别省 ★
      逐小时数据做"日变化"分析时，必须知道窗口内第 0 个点对应真实的一天中的
      哪个钟点。只靠窗口内的局部下标（0..L-1）会把不同相位的窗口混在一起平均，
      出来的"日变化曲线"相位是错的。有了 start_idx 就能算出真实钟点：
          真实小时 = (start_idx[s] + i) % 24
    """
    idx = list(range(0, X.shape[0] - window + 1, stride))
    return (np.stack([X[i:i + window] for i in idx]),
            np.stack([M[i:i + window] for i in idx]),
            np.array(idx, dtype=np.int64))


# ===================== 训练与评测 =====================
def masked_mae(pred, tgt, m):
    return ((pred - tgt).abs() * m).sum() / (m.sum() + 1e-8)


def run_epoch(model, data, A_s, opt=None, args=None):
    """一个 epoch 的训练或验证。损失 = 缺口主损失 + 观测重构 + 物理约束。"""
    X, M_in, M_eval = data
    train = opt is not None
    model.train(train)
    perm = np.random.permutation(len(X)) if train else np.arange(len(X))
    tot, cnt = 0.0, 0
    phys_names = getattr(args, "phys", [])
    phys_w = getattr(args, "phys_weights", {})
    # 物理约束上下文：把需要的东西都塞进来，约束函数就不用知道外部细节
    ctx = {"mean": getattr(args, "norm_mean", None), "std": getattr(args, "norm_std", None),
           "variables": VARIABLES, "station_names": list(STATIONS.keys()),
           "diurnal_weight_per_var": getattr(args, "diurnal_weight_per_var", None)}
    for i in range(0, len(perm), args.batch):
        j = perm[i:i + args.batch]
        xb = torch.tensor(X[j], dtype=torch.float32, device=A_s.device)
        mb = torch.tensor(M_in[j], dtype=torch.float32, device=A_s.device)
        eb = torch.tensor(M_eval[j], dtype=torch.float32, device=A_s.device)
        pred = model(xb, mb)
        l_eval = masked_mae(pred, xb, eb)                   # 缺口处主损失
        l_in = masked_mae(pred, xb, mb)                     # 观测点重构约束
        l_phys, _ = compute_physics_loss(pred, xb, eb, phys_names, phys_w, ctx)
        loss = l_eval + 0.5 * l_in + l_phys
        if train:
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        tot += float(loss.detach()) * len(j)
        cnt += len(j)
    return tot / cnt


@torch.no_grad()
def predict_all(model, X, M_in, A_s, mean, std, batch=64):
    """整批推理。返回 (标准化 pred, 物理单位 imputed)。"""
    model.eval()
    outs = []
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=A_s.device)
        mb = torch.tensor(M_in[i:i + batch], dtype=torch.float32, device=A_s.device)
        outs.append(model(xb, mb).cpu().numpy())
    pred = np.concatenate(outs)
    imputed = np.where(M_in > 0.5, X, pred)                 # 已观测处保留真值
    return pred, imputed * std + mean


def evaluate(pred_phys, X_phys, M_eval, tag=""):
    """在缺口处计算 MAE / RMSE / R²（物理单位）。

    R² 的参照系是"用缺口处真值的均值去填"：
      R² = 1 - SSE/SST，SST 以缺口处真值的均值为基准。
      所以 R²>0 表示"比直接拿均值填更好"，这是 docx 第五步要求的指标之一。
    """
    m = M_eval > 0.5
    err = (pred_phys - X_phys)[m]
    y = X_phys[m]
    sse = float((err ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum()) + 1e-12
    res = {
        "MAE":  float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "R2":   1.0 - sse / sst,
        "n_eval": int(m.sum()),
    }
    res["per_var_MAE"] = [
        float(np.abs((pred_phys - X_phys)[M_eval[:, :, :, c] > 0.5, c]).mean())
        for c in range(pred_phys.shape[-1])
    ]
    res["per_var_RMSE"] = [
        float(np.sqrt(((pred_phys - X_phys)[M_eval[:, :, :, c] > 0.5, c] ** 2).mean()))
        for c in range(pred_phys.shape[-1])
    ]
    print(f"  [{tag:<14}] MAE={res['MAE']:.4f}  RMSE={res['RMSE']:.4f}  R2={res['R2']:+.4f}")
    print(f"  {'':<14}  per-var MAE = " +
          "  ".join(f"{v}:{x:.3f}" for v, x in zip(VARIABLES, res["per_var_MAE"])))
    return res


# ===================== 主流程 =====================
def parse_args():
    p = argparse.ArgumentParser(
        description="自动气象站缺测数据的智能插补 —— 轻量化 GRIN 变体（逐小时）")
    # ---- 数据 ----
    p.add_argument("--source", choices=["merra2", "power_hourly"], default="merra2",
                   help="merra2=GES DISC 原生逐小时（需 Earthdata 账号，变量最全）；"
                        "power_hourly=NASA POWER 逐小时（免鉴权，格式稳定）")
    p.add_argument("--start", type=int, default=20180101)
    p.add_argument("--end", type=int, default=20231231)
    p.add_argument("--refresh", action="store_true", help="忽略缓存，强制重新下载")
    p.add_argument("--allow_synthetic", action="store_true",
                   help="【危险】允许在下载失败时改用合成数据跑通流程。"
                        "输出文件名会带 .SYNTHETIC 标记，禁止用于任何报告。")
    # ---- 模型/窗口 ----
    p.add_argument("--window", type=int, default=72,
                   help="窗口长度（小时）。建议 24 的整数倍，便于日变化分析。"
                        "须满足 window >= 2 * block_maxlen")
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--ksize", type=int, default=5)
    p.add_argument("--sigma_km", type=float, default=350.0)
    p.add_argument("--spatial_mode", choices=["avg", "residual"], default="residual",
                   help="站点空间图聚合方式。avg=原版加权平均（邻站占 72.6%%，实测会稀释"
                        "本站信号，R² 掉到 0.68）；residual=残差式，本站为主体、邻站只做"
                        "增量修正，beta 可学习（实测 R² 0.96）。推荐 residual。")
    p.add_argument("--signed_graph", action="store_true", default=True,
                   help="变量图保留相关性正负号（负相关=负权重，按绝对值行归一化）。"
                        "已设为默认；如需旧的取绝对值行为，用 --no_signed_graph。")
    p.add_argument("--no_signed_graph", dest="signed_graph", action="store_false",
                   help="关闭符号：边权取绝对值 + softplus + 行归一化（旧行为，仅用于对照）")
    # ---- 缺失模拟 ----
    p.add_argument("--mcar_rate", type=float, default=0.15)
    p.add_argument("--block_prob", type=float, default=0.30)
    p.add_argument("--block_maxlen", type=int, default=24,
                   help="最长连续缺测小时数。约束：不得超过 window 的 50%%，"
                        "否则任务会退化成「无中生有」。")
    # ---- 训练 ----
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12,
                   help="早停耐心值。验证集较小时（<20 个窗口）建议设为 --epochs 以关闭早停，"
                        "否则各组实际训练轮数不一致，会污染消融对比")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    # ---- 物理约束（可插拔）----
    p.add_argument("--phys", nargs="*", default=["time_smooth"],
                   choices=list(PHYS_REGISTRY.keys()),
                   help=f"启用的物理约束，默认 time_smooth。可选：{list(PHYS_REGISTRY)}")
    p.add_argument("--phys_weight", nargs="*", default=[],
                   help="按 '名称=权重' 覆盖默认权重，例：--phys_weight time_smooth=0.05")
    p.add_argument("--phys_ablate", action="store_true",
                   help="关闭全部物理约束（消融对照组）")
    # ---- 可解释性 ----
    p.add_argument("--diurnal_bins", type=int, default=24,
                   help="逐小时数据按小时分箱数，为后续日变化约束预留")
    p.add_argument("--dump_aux", action="store_true",
                   help="推理时导出中间张量（嵌入特征/变量图），供可解释性分析")
    # ---- 消融 ----
    p.add_argument("--ablate_no_spatial", action="store_true")
    p.add_argument("--ablate_no_var", action="store_true")
    p.add_argument("--no_var_out_mask", dest="var_out_mask", action="store_false",
                   default=True,
                   help="缺口位置保留变量图的跨要素推断值（默认会把缺口位置的图特征置零）。"
                        "置零时缺口位置三路输入全为 0，变量图对插补本身不起作用。")
    p.add_argument("--tag", type=str, default="",
                   help="实验标签，会写进 metrics 文件与输出文件名（形如 metrics{tag}{suffix}.json），便于多组消融并列")
    args = p.parse_args()

    # --phys_weight 解析成 dict
    args.phys_weights = {}
    for kv in args.phys_weight:
        if "=" not in kv:
            raise SystemExit(f"--phys_weight 格式应为 名称=权重，收到：{kv}")
        k, v = kv.split("=", 1)
        args.phys_weights[k.strip()] = float(v)
    if args.phys_ablate:
        args.phys = []
    # ---- 参数合法性校验：早报错，别跑一半才崩 ----
    if args.window < 2:
        raise SystemExit(f"--window 至少为 2，收到 {args.window}")
    if args.stride < 1:
        raise SystemExit(f"--stride 至少为 1，收到 {args.stride}")
    if args.window % 24 != 0:
        print(f">> [提示] --window={args.window} 不是 24 的倍数。逐小时数据下建议"
              "用 24h 的整数倍（如 168h = 7 天），否则窗口与一天不对齐，"
              "日变化分析会出现相位错位。")
    for nm in args.phys:
        if args.phys_weights.get(nm, PHYS_REGISTRY[nm]["default_weight"]) == 0.0:
            print(f">> [提示] 物理约束 '{nm}' 权重为 0，实际不生效（可作为消融对照）。")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    L, N, C = args.window, len(STATIONS), len(VARIABLES)

    print("=" * 78)
    print(f"自动气象站缺测数据智能插补 · 轻量 GRIN 变体   [{datetime.now():%Y-%m-%d %H:%M:%S}]")
    print(f"数据源 {args.source} ({args.start}~{args.end}) | 窗口 {L}h | "
          f"空间聚合 {args.spatial_mode} | 设备 {device} | 物理约束 {args.phys or '无(消融)'}")
    print("=" * 78)

    # ---- 1. 数据（真实数据优先；失败即终止）----
    X_raw, M_obs, meta = resolve_data(args)
    synthetic = bool(meta.get("synthetic", False))
    out_suffix = ".SYNTHETIC" if synthetic else ""
    tag = f".{args.tag}" if args.tag else ""

    Ttr = int(X_raw.shape[0] * 0.7)
    # 标准化：用训练段统计量。注意 std 里对 0 方差变量兜底，否则会除零
    mean = X_raw[:Ttr].reshape(-1, C).mean(0)
    std = X_raw[:Ttr].reshape(-1, C).std(0)
    std[std < 1e-8] = 1.0
    args.norm_mean, args.norm_std = mean, std
    Xn = (X_raw - mean) / std

    Xw, Mw, Widx = make_windows(Xn, M_obs, L, args.stride)
    # ★ 窗口数量守卫：切不出足够样本时立刻报错，别让它跑进训练循环再崩 ★
    n_w = len(Xw)
    if n_w < 10:
        raise SystemExit(
            f"只切出 {n_w} 个窗口样本，无法划分 训练/验证/测试 三段。\n"
            f"  当前：数据全长 {Xn.shape[0]}h，窗口 {L}h，stride {args.stride}h\n"
            f"  窗口数 = (全长 - 窗口) / stride + 1 = ({Xn.shape[0]} - {L})/{args.stride} + 1 = {n_w}\n"
            "  修复（任选）：\n"
            f"    1) 调小 --window（如 {max(24, L//3)}h）\n"
            f"    2) 调小 --stride（如 {args.stride//3}）\n"
            "    3) 拉长时间范围（--start/--end 多几天）\n"
            "  经验：窗口数至少要有 30 个，训练/验证/测试才都站得住。"
        )
    n_tr, n_va = int(n_w * 0.7), int(n_w * 0.85)
    if n_tr < 1 or n_va - n_tr < 1 or n_w - n_va < 1:
        raise SystemExit(
            f"窗口数 {n_w} 太少，按 7:1.5:1.5 划分后有空集"
            f"（train={n_tr}, val={n_va-n_tr}, test={n_w-n_va}）。\n"
            "  修复：调小 --window 或 --stride，或拉长时间范围。"
        )
    splits, masks_ev, win_idx = {}, {}, {}
    for nm, sl in [("train", slice(0, n_tr)), ("val", slice(n_tr, n_va)),
                   ("test", slice(n_va, len(Xw)))]:
        M_eval = make_eval_mask(Mw[sl], rng, args.mcar_rate, args.block_prob, args.block_maxlen)
        splits[nm] = (Xw[sl], Mw[sl] * (1 - M_eval), M_eval)
        win_idx[nm] = Widx[sl]          # 窗口起始时刻，供日变化相位对齐
    print(f">> 窗口样本 train/val/test = {n_tr}/{n_va - n_tr}/{len(Xw) - n_va} | "
          f"窗口 {L}h | 训练缺口占比 {splits['train'][2].mean():.3f}")

    # ---- 2. 图与模型 ----
    A_s = build_station_graph(args, list(STATIONS.values()), device)
    A_v0 = build_variable_graph_init(splits["train"][0], signed=args.signed_graph)
    model = GRINLite(C, args.hidden, args.n_layers, args.ksize, A_s, A_v0,
                     spatial_mode=args.spatial_mode,
                     signed_graph=args.signed_graph,
                     var_out_mask=args.var_out_mask).to(device)
    model.ablate_var = args.ablate_no_var
    model.return_aux = args.dump_aux
    n_par = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f">> 模型参数量 = {n_par:,}  (预算 100,000 {'✓ 达标' if n_par < 1e5 else '✗ 超标'})")

    # ---- 3. 训练（按 val 选最优权重 + 可选早停）----
    # ★ 为什么把 patience 做成参数 ★
    #   验证集只有 9 个窗口（744h / 72h 窗口 / stride 12），val 曲线噪声大，
    #   固定 patience=12 容易在平台期误触发早停，导致各组"实际训练轮数不同"。
    #   实测踩过：同一批消融里，完整模型 166 轮就早停，而另一组跑满 300 轮，
    #   两者 val 0.185 vs 0.134 —— 指标差异被训练预算差异污染，消融不再公平。
    #   做消融时请统一传 --patience 300（≈关闭早停），保证各组预算一致。
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, wait, path = 1e9, 0, f"best_grin{tag}{out_suffix}.pt"
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
        if wait >= args.patience:
            print(f">> 早停触发（patience={args.patience}，"
                  f"实际训练 {ep + 1}/{args.epochs} 轮，最优 val={best:.4f}）")
            break
    model.load_state_dict(torch.load(path, map_location=device))

    # ---- 4. 测试评测 ----
    print("\n>> 测试集结果（物理单位）")
    Xte, M_in_te, M_eval_te = splits["test"]
    t0 = time.perf_counter()
    pred, pred_phys = predict_all(model, Xte, M_in_te, A_s, mean, std)
    dt = (time.perf_counter() - t0) / len(Xte)
    key = "GRIN-lite" + (f"[{args.tag}]" if args.tag else "")
    res = {key: evaluate(pred_phys, Xte * std + mean, M_eval_te, key)}
    # ★ 完整运行配置落盘 ★
    # 起因：早期版本的 metrics 只记了 source/window_h，导致事后无法复现
    # （换了个 stride、epochs 或 lr 就跑不出同一个数，却查不到当时用的是哪个）。
    # 这里把决定结果的每个参数都写进 JSON，做到"结果可追溯"。
    run_config = {
        "data": {"source": meta.get("source"), "start": args.start, "end": args.end,
                 "n_hours": int(X_raw.shape[0]), "n_stations": int(X_raw.shape[1]),
                 "n_vars": int(X_raw.shape[2]),
                 "data_signature": meta.get("data_signature")},
        "window": {"window_h": L, "stride_h": args.stride,
                   "n_windows": {"train": n_tr, "val": n_va - n_tr,
                                 "test": int(len(Xw)) - n_va}},
        "missing": {"mcar_rate": args.mcar_rate, "block_prob": args.block_prob,
                    "block_maxlen_h": args.block_maxlen},
        "model": {"hidden": args.hidden, "n_layers": args.n_layers,
                  "ksize": args.ksize, "sigma_km": args.sigma_km,
                  "spatial_mode": args.spatial_mode,
                  "signed_graph": bool(args.signed_graph),
                  "ablate_no_spatial": bool(args.ablate_no_spatial),
                  "ablate_no_var": bool(args.ablate_no_var)},
        "train": {"epochs": args.epochs, "patience": args.patience,
                  "batch": args.batch, "lr": args.lr,
                  "seed": args.seed, "epochs_run": ep + 1,
                  "early_stopped": bool(ep + 1 < args.epochs)},
        "phys": {"enabled": args.phys, "weights": args.phys_weights,
                 "phys_ablate": bool(args.phys_ablate)},
        "params": n_par,
    }
    res[key].update({"params": n_par, "ms_per_sample": dt * 1e3,
                     "phys": args.phys, "source": meta.get("source"),
                     "window_h": L, "synthetic": synthetic,
                     "run_config": run_config})
    # 均值填充基线：标准化空间里的 0 == 训练集均值，命名如实标注
    res["mean-fill"] = evaluate(np.zeros_like(Xte) * std + mean,
                                Xte * std + mean, M_eval_te, "mean-fill")

    # ---- 可解释性导出 ----
    if args.dump_aux:
        model.return_aux = True
        with torch.no_grad():
            _ = model(torch.tensor(Xte[:2], dtype=torch.float32, device=device),
                      torch.tensor(M_in_te[:2], dtype=torch.float32, device=device))
        aux = {k: v.cpu().numpy() for k, v in model.get_aux().items()}
        vg = aux.pop("variable_graph", None)
        # ★ spatial_beta 必须落盘 ★
        #   报告里"浅层用邻站、深层靠自己"的结论全靠这几个数，没有原始 dump
        #   就成了"口头数字"。这里连同变量图一起存进 npz，做到可追溯。
        beta = np.asarray(model.get_spatial_beta(), dtype=np.float32)
        np.savez(f"aux{tag}{out_suffix}.npz",
                 variable_graph=vg, spatial_beta=beta, **aux)
        print(f">> 已导出可解释性中间量 aux{tag}{out_suffix}.npz")
        print(f"   变量图 {vg.shape} | 各层 spatial_beta = "
              f"{np.array2string(beta, precision=4)}")

    mpath = f"metrics{tag}{out_suffix}.json"
    json.dump(res, open(mpath, "w"), indent=2)

    # ---- 5. 可视化 ----
    _plot_graphs(model, A_s, C, tag, out_suffix)
    _plot_imputation(Xte, M_in_te, pred_phys, mean, std, tag, out_suffix)
    if args.diurnal_bins:
        _plot_diurnal(Xte, M_in_te, pred_phys, mean, std, args.diurnal_bins, tag, out_suffix,
                      te_start_idx=win_idx["test"])

    print(f"\n>> 已保存: {path} / {mpath} / "
          f"graph_heatmaps{tag}{out_suffix}.png / imputation_demo{tag}{out_suffix}.png")
    if synthetic:
        print("\n" + "!" * 78)
        print("!!  提醒：本次结果基于【合成数据】，仅供流程验证，切勿写入报告  !!")
        print("!" * 78)


def _plot_graphs(model, A_s, C, tag, suf):
    """站点空间图 + 学习到的变量物理图。后者是"可解释性"的主要产出。"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    names = list(STATIONS.keys())
    im0 = ax[0].imshow(A_s.cpu().numpy(), cmap="viridis")
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names, fontsize=8)
    ax[0].set_title("Station spatial graph (Gaussian kernel)")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)

    Av = model.get_variable_graph()
    im1 = ax[1].imshow(Av, cmap="magma")
    ax[1].set_xticks(range(C))
    ax[1].set_xticklabels(VARIABLES, rotation=30, fontsize=8)
    ax[1].set_yticks(range(C))
    ax[1].set_yticklabels(VARIABLES, fontsize=8)
    ax[1].set_title("Learned variable physical graph")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(f"graph_heatmaps{tag}{suf}.png", dpi=150)
    plt.close()


def _plot_imputation(Xte, M_in, pred_phys, mean, std, tag, suf, s0=0, n0=0, c0=0):
    """单站点单变量的插补曲线对照图。"""
    plt.figure(figsize=(13, 4))
    gt = Xte[s0, :, n0, c0] * std[c0] + mean[c0]
    obs = np.where(M_in[s0, :, n0, c0] > 0.5, gt, np.nan)
    plt.plot(gt, "k--", lw=1, label="Ground truth")
    plt.plot(obs, "o", ms=3, label="Observed (input)")
    plt.plot(pred_phys[s0, :, n0, c0], "r-", lw=1.2, label="GRIN-lite imputation")
    plt.title(f"{list(STATIONS)[n0]} - {VAR_LABEL[c0]}" + 
              ("   [SYNTHETIC DATA - NOT FOR REPORT]" if suf else ""))
    plt.xlabel("Hour within window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"imputation_demo{tag}{suf}.png", dpi=150)
    plt.close()


def _plot_diurnal(Xte, M_in, pred_phys, mean, std, bins, tag, suf,
                  te_start_idx=None, s0=0, n0=0, c0=0):
    """日变化相位诊断图（逐小时数据专属）。

    用途：直观判断模型是否学到了日变化 —— 这是换逐小时数据的主要收益。

    ★ 相位对齐 ★
      必须用窗口在原始时序中的起始下标 te_start_idx 把"窗口内位置"换算成
      "真实钟点"，否则不同相位的窗口会被混在一起平均，曲线相位是错的
      （会看到相位滞后 L/2 的假象）。这是逐小时数据独有的坑，日尺度数据没有。

    怎么看这张图：
      - 左图两条线若相位一致、幅度接近 -> 模型学到了日变化规律；
      - 若橙线整体右移（相位滞后），说明模型倾向于"用前面的值推算后面的值"，
        本质是平滑而非真正理解日循环 —— 这正是后续加日变化物理约束的动机。
    """
    gt = Xte[:, :, n0, c0] * std[c0] + mean[c0]     # (S, L)
    pr = pred_phys[:, :, n0, c0]
    S, L = gt.shape

    # 真实钟点：窗口起点 + 窗口内偏移，对 24 取模
    if te_start_idx is not None and len(te_start_idx) == S:
        h = (te_start_idx[:, None] + np.arange(L)[None, :]) % 24
        xlabel, phase_note = "Hour of day (UTC)", ""
    else:
        h = np.broadcast_to(np.arange(L) % 24, (S, L))
        xlabel, phase_note = "Hour of day (UTC)", "   [相位按窗口起点对齐]"

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for name, arr, style in [("Truth", gt, "k-o"), ("Imputed", pr, "tab:orange")]:
        prof = np.array([arr[h == k].mean() for k in range(24)])
        ax[0].plot(range(24), prof, style, ms=3.5, lw=1.4, label=name)
    ax[0].set_title(f"Diurnal profile - {list(STATIONS)[n0]} {VARIABLES[c0]}{phase_note}",
                    fontsize=10)
    ax[0].set_xlabel(xlabel)
    ax[0].set_xticks(range(0, 24, 3))
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    # 右图：只画一个窗口的连续 7 天（168h），带观测点标记，看细节
    seg = min(L, 168)
    ax[1].plot(range(seg), gt[s0, :seg], "k--", lw=1.1, label="Ground truth")
    obs = np.where(M_in[s0, :seg, n0, c0] > 0.5, gt[s0, :seg], np.nan)
    ax[1].plot(range(seg), obs, "o", ms=3, color="tab:blue", label="Observed (input)")
    ax[1].plot(range(seg), pr[s0, :seg], "-", lw=1.2, color="tab:red", label="Imputed")
    ax[1].set_title(f"One window detail - window #{s0}", fontsize=10)
    ax[1].set_xlabel("Hour within window")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"diurnal{tag}{suf}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
