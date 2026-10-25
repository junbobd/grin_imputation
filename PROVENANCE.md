# PROVENANCE — 代码溯源：本作品相对 GRIN 开源实现的增删改

本文件回答一个问题：**`grin_imputation.py` 是从哪里来的？**

答案是：它是在 **GRIN**（ICLR 2022）官方 PyTorch 实现的基础上，针对
「自动气象站缺测数据的智能插补」这一任务**增删改**得到的派生作品。

GRIN 的原始文件已按原样收录在 [`third_party/GRIN_ICLR2022/`](third_party/GRIN_ICLR2022/)，
授权状态与引用信息见该目录下的 [`NOTICE.md`](third_party/GRIN_ICLR2022/NOTICE.md)。

---

## 一、上游基线

| 项 | 内容 |
|---|---|
| 名称 | GRIN（Graph Recurrent Imputation Network） |
| 论文 | *Filling the G\_ap\_s: Multivariate Time Series Imputation by Graph Neural Networks* |
| 会议 | ICLR 2022 |
| 作者 | Andrea Cini、Ivan Marisca、Cesare Alippi |
| 代码 | https://github.com/Graph-Machine-Learning-Group/grin |
| arXiv | https://arxiv.org/abs/2108.00298 |
| 上游技术栈 | PyTorch 1.8 + PyTorch Lightning 1.4 + einops + TensorFlow 2.5（BRITS 基线）+ YAML 配置驱动 |
| 上游任务 | 传感器网络多变量时间序列的缺失值插补（空气质量 / 交通流量 / 合成数据） |

**选它的理由**：① 有 ICLR 正会论文支撑，方法与结论可引用；② 任务本质相同
（站点网络 + 多变量 + 缺失插补），迁移成本低；③ 结构清晰，模块化程度高，
增删改的边界明确。

---

## 二、文件级映射总表

下表逐项列出上游模块与本作品模块的对应关系与操作类型。
**操作类型**：`改` = 保留思路、重写实现；`删` = 直接移除；`增` = 上游没有、完全新写。

| # | 上游（GRIN） | 本作品（`grin_imputation.py`） | 操作 |
|---|---|---|---|
| 1 | `lib/nn/models/grin.py` → `GRINet` | `GRINLite` | **改** |
| 2 | `lib/nn/layers/gril.py` → `GRIL` / `BiGRIL` | `STBlock` | **改** |
| 3 | `lib/nn/layers/gcrnn.py` → `GCGRUCell` | —（不采用） | **删** |
| 4 | `lib/nn/layers/gril.py` → `SpatialDecoder` | —（不采用） | **删** |
| 5 | `lib/nn/layers/spatial_conv.py` → `SpatialConvOrderK` | `build_station_graph` + `STBlock.forward` 的聚合段 | **改** |
| 6 | `lib/nn/models/grin.py` → `BiGRIL.emb`（节点嵌入） | `build_variable_graph_init` + `GRINLite.A_v`（**变量图**） | **改** |
| 7 | `lib/utils/numpy_metrics.py` → `masked_mae` / `masked_rmse` / `r2` | `masked_mae` / `evaluate` | **沿用 + 改** |
| 8 | `lib/data/imputation_dataset.py` + `lib/datasets/*.py`（缺失注入） | `make_eval_mask` | **改** |
| 9 | `lib/data/temporal_dataset.py`（window / stride / horizon / delay） | `make_windows` | **改** |
| 10 | `scripts/run_imputation.py`（Lightning 训练循环） | `run_epoch` / `predict_all` / `main` | **改** |
| 11 | `lib/fillers/` → BRITS / RGAIN / MPGRU / VAR / GraphFiller | —（不采用） | **删** |
| 12 | `config/*.yaml`（配置驱动） | `argparse` 命令行 | **改** |
| 13 | 上游无 | `haversine` + 高斯核站点图构建 | **增** |
| 14 | 上游无 | `PHYS_REGISTRY` / `compute_physics_loss` 物理约束注册表 | **增** |
| 15 | 上游无 | 数据获取层（NASA POWER / MERRA-2）+ 缓存指纹 + 失败即终止 | **增** |
| 16 | 上游无 | 可解释性钩子（`get_spatial_beta` / `--dump_aux`） | **增** |
| 17 | 上游无 | `baseline.py` / `LSTMbaseline.py` / `experiment_missing_gradient.py` | **增** |

---

## 三、逐项改动详解

### 改-1 ★ 空间聚合方式（本作品最核心的改动）★

**上游做法**（`lib/nn/layers/spatial_conv.py` → `SpatialConvOrderK.forward`）：

```python
out = [x] if self.include_self else []
for a in support:
    x1 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous()
    out.append(x1)
    for k in range(2, self.order + 1):
        x2 = torch.einsum('ncvl,wv->ncwl', (x1, a)).contiguous()
        out.append(x2)
        x1 = x2
out = torch.cat(out, dim=1)
out = self.mlp(out)          # 1x1 卷积混合
```

即：把「自身特征」与「各阶扩散结果」**在通道维拼接**，再过一个 1×1 卷积混合。
上游还会先构造 `adj_fwd` / `adj_bwd` 两个方向的行归一化支撑矩阵（`compute_support`），
并在 `run_imputation.py` 中**显式去掉自环**（`np.fill_diagonal(adj, 0.)`），
靠 `include_self=True` 把自身拼回来。

**本作品做法**（`STBlock.forward`）：

```python
agg = torch.einsum("mn,blnh->blmh", A_s, h)      # 邻站聚合
if self.mode == "residual":
    h = h + self.beta * (agg - h)                # 残差式：以本站为主
else:
    h = agg                                      # 原版：加权平均
```

**具体改了什么**：

| 子项 | 上游 | 本作品 |
|---|---|---|
| 扩散阶数 | `order` 阶（默认 2） | 1 阶（无向气象站网不需要多跳） |
| 支撑矩阵 | `adj_fwd` + `adj_bwd` 双向 | 单个对称行归一化矩阵 |
| 自环处理 | 邻接去自环，靠 `include_self` 拼接 | 保留自环（`fill_diagonal(A, 1.0)`）后行归一化 |
| 融合方式 | 拼接 + 1×1 卷积（参数量 `(order·support_len + 1)·c_in·c_out`） | **可学习标量 β 的残差插值**（每层 1 个标量） |
| 新增模式 | — | `residual`：`h + β(A_s h − h)` |

**为什么这么改（消融证据）**：

上游把邻接去自环、再行归一化，使得聚合后的特征**完全由邻站构成**。
迁移到本项目场景后实测：行归一化邻接矩阵的**自环权重均值仅 0.274**，
即每次聚合 **72.6% 的信息来自其他站点**，本站在特征里只占 27.4%。

而本项目选用的 8 个华北站点相距 **110~1056 km**，在 72 h 窗口内各站温度
由日变化主导、走势高度同步。此时邻站提供不了新信息，只会稀释本站信号。
消融实验直接印证：**去掉空间图后 R² 从 0.840 跳到 0.968**（MAE 腰斩）。

`residual` 模式的设计意图：**本站特征保留 100%，邻站只做增量修正**。
关键在于 β 是 `nn.Parameter(torch.tensor(0.1))`，可学习 ——
**若邻站真无增益，梯度会把 β 压向 0，模型自动退化为「无空间图」**；
若邻站有增益（如长缺口跨站推断），β 会被推大。这比人工二选一「用/不用空间图」更稳健。

> **这不是"图没用"，而是"图喧宾夺主"。** 逐项消融才能定位到这一点：
> 若只看总体指标，很容易误判为「图结构整体无效」从而砍掉整个分支。
> 完整的消融数据见 `metrics.ablation.json`，各层 β 的原始取值见 `auxiliary.final.npz`。

---

### 改-2 ★ 时序编码器：删掉循环，换成深度可分离膨胀卷积 ★

**上游做法**（`lib/nn/layers/gril.py` + `lib/nn/layers/gcrnn.py`）：
`GRIL.forward` 沿时间轴**逐步循环**，每一步用 `GCGRUCell`（图卷积 GRU）更新隐状态，
并用 `SpatialDecoder` 从邻站取消息；`BiGRIL` 再把序列反转跑一遍反向分支，
最后把前向/反向的隐表示拼接，经 MLP 输出插补值。计算复杂度 O(L) 串行。

**本作品做法**（`DSC` + `STBlock`）：

```python
class DSC(nn.Module):
    def __init__(self, h, k, dil):
        self.dw = nn.Conv1d(h, h, k, padding=dil*(k-1)//2, dilation=dil, groups=h)
        self.pw = nn.Conv1d(h, h, 1)
        self.gn = nn.GroupNorm(4, h)
    def forward(self, x):
        return F.gelu(self.gn(self.pw(self.dw(x))) + x)
```

各层 `dilation = 2 ** i`，整段序列**并行**卷积。

**具体改了什么**：

- **删**：`GCGRUCell`、`SpatialDecoder`、`BiGRIL` 的反向分支、`h0` 隐状态嵌入参数
- **删**：双向融合的 MLP（`4*hidden + input + emb → ff_size → input`）
- **增**：深度可分离卷积（depthwise `groups=h` + pointwise 1×1）+ GroupNorm + GELU + 残差
- **增**：膨胀率随层指数增长（`2**i`），用 O(log L) 层覆盖长依赖

**为什么这么改**：参赛文档创新点二命题为「相同参数预算下，卷积结构优于循环结构」。
深度可分离卷积的参数量约为普通卷积的 `1/hidden`，这正是「轻量化」的着力点。
实测在同参数量（15,661 vs 15,959，差 1.90%）条件下，本作品全面优于 LSTM 基线。

---

### 改-3 缺失模式生成

**上游做法**：缺失模式由数据集自带（`lib/datasets/air_quality.py` 等预先注入 NaN），
训练时用 `--whiten-prob`（默认 0.05）随机加掩码，评测用数据集预置的 `eval_mask`。

**本作品做法**（`make_eval_mask`）：自行生成两类缺失，参数可控：

- **MCAR 点缺失**：`rng.random(shape) < point_rate`（默认 0.15）
- **连续块缺失**：以 `block_prob`（默认 0.30）概率触发，长度 `2 ~ block_maxlen`（默认 24 h）

**增**了一条硬校验：`block_maxlen > 0.5 × window` 直接 `ValueError`。

> 加这条校验是因为踩过坑：`window=24` 配 `block_maxlen=24` 时最长缺口占窗口 88%，
> 窗口内只剩 1~2 个锚定点，**任务从「插补」退化成「无中生有」**，
> 任何模型都只能输出直线。这是任务设置失真，不是模型能力问题。
> 规则：**缺口长度 : 窗口长度 ≤ 1:2，推荐 1:3**。

**为什么这么改**：参赛任务需要「缺测强度梯度实验」（缺测率 × 连续段长度），
必须能精确控制缺失模式，而上游的预置掩码无法满足。

---

### 改-4 窗口切片

**上游做法**：`TemporalDataset` 用 `window` / `horizon` / `delay` / `stride` 四个参数
统一处理「预测」「插补」两类任务，返回 `dict`。

**本作品做法**：`make_windows(X, M, window, stride)` 返回三元组 `(Xw, Mw, start_idx)`。

**增** `start_idx`（每个窗口的起始时刻）是为了**日变化相位对齐** ——
气象要素有强日循环，`_plot_diurnal()` 需要按真实小时分箱才能判断相位对不对。
上游无此需求，故无此字段。

---

### 改-5 指标

**上游做法**（`lib/utils/numpy_metrics.py`）：`masked_mae` / `masked_mse` /
`masked_rmse` / `masked_mre` / `masked_mape`，**不含 R²**。

**本作品做法**：`masked_mae` 口径沿用（`err.sum() / mask.sum()`，与上游一致，
保证与基线同口径）；`evaluate()` **增**加 R²，且以**缺口处真值均值为基准**：

```python
ss_res = ((pred - true) ** 2 * m).sum()
ss_tot = (((true - true[m].mean()) ** 2) * m).sum()
r2 = 1.0 - ss_res / ss_tot
```

这样 R² 的业务含义直接就是「**比均值填充好多少**」，可直接对接报告口径。

> 必须同时报 R² 的理由：MAE/RMSE **不能跨变量直接平均** ——
> 气压量纲 1e5 Pa，会完全主导误差。线性插值在气压这种平滑变量上逐变量 MAE 仅 0.03 Pa，
> 会把跨变量平均 MAE 拉得很低（13.14，低于部分深度模型），
> 但其 R² 只有 0.7802。**只看平均 MAE 会得出错误结论。**

---

### 增-1 物理规律约束注册表（上游完全没有）

`PHYS_REGISTRY` 字典 + `@register_physics(name, default_weight, note)` 装饰器 +
`compute_physics_loss(...)`，通过 `--phys` / `--phys_weight` 命令行开关启用。

当前注册 4 个约束，**默认只启用 `time_smooth`**，其余为函数骨架（默认权重 0）：

| 名称 | 状态 | 说明 |
|---|---|---|
| `time_smooth` | 启用（权重 0.05） | 缺口段插补值的二阶差分惩罚，抑制锯齿过冲 |
| `diurnal_cycle` | 骨架 | 按小时分箱的日变化一致性 |
| `humidity_consistency` | 骨架 | 比湿与温度的物理关系 |
| `wind_nonneg` | 骨架 | 风速非负 |

**为什么这么设计**：物理约束的权重需要消融实验确定，先做成可插拔注册表，
才能用 `--phys_ablate` / `--phys_weight` 逐项验证贡献，而不是硬编码。

---

### 增-2 变量物理图（上游的 `emb` 是节点嵌入，不是变量图）

**上游**：`BiGRIL.emb` 是 `nn.Parameter(embedding_size, n_nodes)`，
即**站点（节点）的嵌入向量**，用于给每个站点一个可学习的身份标识。

**本作品**：把「图」从节点维移到**变量维** ——
`GRINLite.A_v = nn.Parameter(A_v_init)`，形状 `(C, C)`，C = 变量数（6）。

- 初始化：训练段变量间 **Pearson 相关系数绝对值**（`build_variable_graph_init`）
- 前向：`Av = softplus(A_v)` 后行归一化，`x_v = einsum("uv,blnu->blnv", Av, x)`
- 含义：**物理相关先验**（温度–比湿正相关、气压–温度弱相关…）由数据给出，
  训练中边权可继续调整，即「数据驱动地修正物理先验」

这是本作品相对上游的一个**新增结构**，对应参赛文档里「变量物理图」这一设计。

---

### 增-3 数据获取层（上游从本地 npz 读，本作品直接拉取）

上游的数据需要手动下载数据集放到 `datasets/` 目录。本作品**新增**了完整的获取链路：

| 函数 | 作用 |
|---|---|
| `fetch_power_hourly` | NASA POWER 逐小时，免鉴权通道 |
| `fetch_merra2_opendap` | MERRA-2 原生（GES DISC OPeNDAP，需 Earthdata 账号） |
| `_data_signature` | 缓存指纹（源/时段/站点/变量/版本），参数一变自动失效 |
| `resolve_data` | 统一入口：真实数据失败 → 打印错误 + 排查建议 → `sys.exit(2)` |

**为什么加「失败即终止」**：报告里的每个数字都必须来自真实数据。
用合成数据跑出的 MAE 写进报告是学术不端，答辩时也会被问穿。
如仅需跑通流程，必须显式加 `--allow_synthetic`，届时输出文件名带 `.SYNTHETIC`
后缀、图标题打警告横幅，绝不与真实结果混淆。

---

### 增-4 可解释性工程钩子（上游无）

| 接口 | 作用 |
|---|---|
| `GRINLite.get_spatial_beta()` | 取各层学到的邻站注入强度 β |
| `GRINLite.get_variable_graph()` | 取归一化后的变量图边权 |
| `GRINLite.record_aux()` / `get_aux()` | 统一中间量记录口 |
| `--dump_aux` | 导出中间张量到 `aux*.npz` |

**β 的层级规律**是很有报告价值的可解释性产出：
浅层 β 为正（温和利用邻站低频背景），深层 β 近乎归零（本站精细时序结构无需邻站）。
即**模型自己学会了「何时该看邻站、何时不该看」**，无需人工设计。
原始取值见 `auxiliary.final.npz`，可视化见 `graph_heatmaps*.png`。

---

### 删-1 上游的其他模型与基线

上游 `lib/fillers/` + `lib/nn/models/` 包含 8 个模型：BRITS、RGAIN、MPGRU、BiMPGRU、
VAR、BiRNN、RNN、GRIN。`scripts/run_baselines.py` 还实现了 MEAN / KNN / MF / MICE 基线。

本作品**全部删除**，按参赛要求只保留 **3 个基线**：

| 基线 | 文件 | 用途 |
|---|---|---|
| 线性插值 | `baseline.py` | 传统插值代表 |
| KNN | `baseline.py` | 非参学习方法代表 |
| 同参数量 LSTM | `LSTMbaseline.py` | 验证创新点二（卷积 vs 循环） |

均值填充由主脚本内直接输出（标准化空间填 0 == 填训练集均值，
**命名如实标注为 `mean-fill`，不叫 `zero-fill`**）。

---

### 删-2 重依赖（PyTorch Lightning / TensorFlow / einops / YAML）

上游依赖 `pytorch-lightning==1.4`、`tensorflow==2.5.0`、`torch==1.8`、
`torchmetrics==0.5`、`einops`、`pyyaml`、`fancyimpute`、`tables` 等。

本作品改为**纯 PyTorch + numpy + scikit-learn + matplotlib**，
训练循环自己写（`run_epoch` / `predict_all`），配置改用 `argparse`。

**为什么删**：① 比赛环境安装 PyTorch Lightning 1.4 + TensorFlow 2.5 代价高且易冲突；
② 上游用 YAML + `filter_args` 反射注入超参，调试时链路不透明；
③ 本项目模型已大幅简化（无双向 RNN、无 decoder），不需要 Lightning 的分布式/回调体系。

---

## 四、一句话总结

> **上游 GRIN 提供了「图神经网络 + 时空编码器做多变量缺失插补」的骨架与论文依据；
> 本作品保留其「站点图 + 时空编码」的总体范式，把循环编码器换成深度可分离膨胀卷积（轻量化），
> 把空间聚合从「加权平均」改成「可学习 β 的残差式注入」（解决邻站稀释），
> 新增变量物理图、物理约束注册表、真实数据获取链路与可解释性钩子，
> 并删去与参赛要求无关的 7 个模型和全部重依赖。**

---

## 五、复现方式

```bash
# 主实验（在项目根目录执行）
python -u grin_imputation.py --source power_hourly \
    --start 20250821 --end 20250920 \
    --window 72 --stride 12 --block_maxlen 24 \
    --epochs 300 --hidden 64 --n_layers 3 --batch 16 --lr 2e-3 \
    --spatial_mode residual --tag final

# 消融（统一在 residual 聚合下，单批次可直接相减）
python -u grin_imputation.py ... --ablate_no_spatial    # 去站点空间图
python -u grin_imputation.py ... --ablate_no_var        # 去变量图
python -u grin_imputation.py ... --phys_ablate          # 去物理约束
python -u grin_imputation.py ... --phys_weight time_smooth=0.2
python -u grin_imputation.py ... --spatial_mode avg     # 用上游原版平均聚合

# 可解释性导出
python -u grin_imputation.py ... --dump_aux             # → aux.<tag>.npz
```

每次运行的完整配置（数据指纹、窗口/步长、缺失参数、模型结构、训练超参、物理约束）
都会写入 `metrics*.json` 的 `run_config` 字段，做到**结果可追溯**。
