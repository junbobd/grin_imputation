# NOTICE — 第三方参考源码归属说明

本目录收录的是 **GRIN** 官方开源实现的**部分原始文件**，仅作学术参考与代码溯源比对之用。
本项目的模型代码（`../../grin_imputation.py`）是在这些原始文件基础上**增删改**得到的派生作品，
并非直接调用或原样复用。逐项改动见项目根目录的 [`PROVENANCE.md`](../../PROVENANCE.md)。

---

## 上游来源

| 项 | 内容 |
|---|---|
| 仓库 | https://github.com/Graph-Machine-Learning-Group/grin |
| 论文标题 | *Filling the G\_ap\_s: Multivariate Time Series Imputation by Graph Neural Networks* |
| 会议 | ICLR 2022（International Conference on Learning Representations） |
| 作者 | Andrea Cini、Ivan Marisca、Cesare Alippi |
| arXiv | https://arxiv.org/abs/2108.00298 |
| OpenReview | https://openreview.net/forum?id=kOu3-S3wJ7 |
| 获取方式 | `https://codeload.github.com/Graph-Machine-Learning-Group/grin/zip/refs/heads/main`（main 分支快照） |
| 快照获取时间 | 2026-09-26 |

## BibTeX

```bibtex
@inproceedings{cini2022filling,
    title={Filling the G\_ap\_s: Multivariate Time Series Imputation by Graph Neural Networks},
    author={Andrea Cini and Ivan Marisca and Cesare Alippi},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=kOu3-S3wJ7}
}
```

## ⚠ 授权状态（如实说明，务必注意）

**上游仓库的 main 分支快照中不存在 `LICENSE` 文件**（已核对：仓库根目录仅有
`.gitignore`、`README.md`、`conda_env.yml`、`grin.png`、`requirements.txt`
以及 `config/`、`lib/`、`scripts/` 三个目录）。

因此本目录中的文件**没有明确的开源许可证授权**。收录行为基于：

1. 上游 `README.md` 明确表示该仓库是「for the reproducibility of the experiments
   presented in the paper」，属于面向学术复现的公开代码；
2. 本目录内容**仅用于学术溯源、算法比对与论文引用**，不参与本项目任何
   训练、推理或评测流程（项目运行不 import 本目录任何模块）；
3. 本目录所有文件**未作任何修改**，保留上游原始内容。

> **提交前建议**：若参赛/发表要求严格的许可证合规，可联系论文作者
> （Andrea Cini <andrea.cini@usi.ch>、Ivan Marisca <ivan.marisca@usi.ch>）
> 确认授权，或改为在正文中以引用 + 伪代码对照的形式说明血缘关系，
> 从提交包中移除本目录。**本目录的存在不影响项目能否运行。**

## 目录内容对照

| 本目录文件 | 上游路径 | 在本项目中的作用 |
|---|---|---|
| `lib/nn/models/grin.py` | 同名 | `GRINet` 顶层模型 → 本项目 `GRINLite` 的改造起点 |
| `lib/nn/layers/gril.py` | 同名 | `GRIL` / `BiGRIL` 时空循环编解码 → 本项目 `STBlock` 的改造起点 |
| `lib/nn/layers/gcrnn.py` | 同名 | `GCGRUCell` 图卷积 GRU → 本项目**删除**该循环单元，替换为膨胀卷积 |
| `lib/nn/layers/spatial_conv.py` | 同名 | `SpatialConvOrderK` 空间聚合 → 本项目 `build_station_graph` + `residual` 聚合的改造起点 |
| `lib/nn/layers/imputation.py` | 同名 | `ImputationLayer`（VAR 基线用） → 本项目未采用，仅作对照 |
| `lib/utils/numpy_metrics.py` | 同名 | `masked_mae` / `rmse` / `r2` → 本项目 `evaluate()` 的指标口径来源 |
| `lib/data/imputation_dataset.py` | 同名 | 窗口切分与掩码注入 → 本项目 `make_windows` / `make_eval_mask` 的改造起点 |
| `requirements.txt` | 同名 | 上游依赖清单，供环境比对 |
