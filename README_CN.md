<div align='center'>
<h1>ERPO: Environment-Regularized Policy Optimization</h1>
<h4>Breaking the Stability–Exploration Dilemma by Moving Regularization from the Action Side to the Input Side</h4>

[![Paper](https://img.shields.io/badge/Paper-5f16a8?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.23311)
[![PDF](https://img.shields.io/badge/PDF-red?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](./2608.23311v1.pdf)
[![Code](https://img.shields.io/badge/Code-2ea44f?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AlibabaResearch/ERPO)
[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-ff6f00?style=for-the-badge)](https://arxiv.org/abs/2608.23311)
</div>

<p align="center">
  <a href="README.md">English</a> | 简体中文
</p>

> [!IMPORTANT]
> **🔥 最新动态**
> - [2026/08] ERPO 被 **EMNLP 2026 主会**录用。
> - [2026/08] 我们基于 ROLL 发布了完整的训练方案、配置文件和评测数据。

面向大语言模型的策略优化（PO）长期面临**稳定性–探索权衡**：标准的 Policy-KL 正则项作用于*响应*（动作）分布，在限制探索的同时，却完全没有约束*查询*（输入）分布。随着训练推进，模型在训练查询上的似然会逐渐偏离强化学习前的参考模型，从而在不易察觉的情况下破坏优化稳定性。

**环境正则化策略优化（Environment-Regularized Policy Optimization，ERPO）** 将正则化移至输入侧，从而打破这一困境。ERPO 包含两个互补组件：

1. **Query-KL（QKL）**：对当前策略诱导的*查询*分布施加 KL 惩罚，限制其相对强化学习前参考分布的偏移。QKL 的梯度只经过查询似然，不会影响响应得分函数，因此能够完整保留探索能力。
2. **参考模型导出的逐查询权重**：一种数据集静态权重，使每个查询的更新更偏向参考分布中的典型查询，从而降低估计方差，并增强高解码温度下的鲁棒性。

这两个组件均与估计器无关，可以接入 GRPO、PPO 和 REINFORCE 等训练流程，且**不需要额外的前向计算**。

<p align="center">
  <img src="assets/overview.png" alt="ERPO 概览" width="95%">
</p>

**ERPO 概览。** (a) 策略模型与参考模型分别诱导当前查询分布（`ρ_θ`）和参考查询分布（`ρ_θ₀`）；ERPO 引入 Query-KL 来限制二者之间的偏移。(b) 标准 GRPO 流程对采样响应进行评分，并计算组内优势。(c) ERPO 使用 Query-KL 取代响应侧 KL，并通过参考模型导出的逐查询权重对优势进行重加权，从而实现环境感知的策略更新。

## 主要结果

在六个数学推理基准上（Qwen2.5-Math-7B，训练 240 步），ERPO 持续优于 GRPO 基线：

| 方法 | AIME24 | AIME25 | AMC | MATH500 | Minerva | Olympiad | **平均值** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GRPO | 0.174 | 0.072 | 0.398 | 0.528 | 0.207 | 0.266 | 0.274 |
| **ERPO** | **0.218** | **0.110** | **0.478** | **0.677** | **0.214** | **0.316** | **0.336** |

*Avg@32 为解码温度 0.1–1.5 上的均值。ERPO 的 Avg@32 总体平均提升 **6.2%**，Pass@32 提升 3.6%，Pass@1 提升 5.7%。包括 Pass@32、Pass@1 以及长时程（1K 步）训练动态在内的完整结果，请参阅[论文](https://arxiv.org/abs/2608.23311)。*

<p align="center">
  <img src="assets/result.png" alt="六个数学推理基准上的 Avg@32" width="95%">
</p>

### 训练动态

ERPO 使用输入侧正则化取代动作侧 Policy-KL，其效果贯穿整个训练过程：查询分布始终锚定在参考分布附近，而 GRPO 的查询分布会持续漂移；在长时程训练中，ERPO 也能保持稳定，避免性能崩溃。

<p align="center">
  <img src="assets/train.png" alt="训练动态：ERPO 与 GRPO 对比" width="95%">
</p>

**上图（240 步）：** GRPO 的 Policy-KL 虽然保持平稳，但 Query-KL 持续上升，且熵在训练后期出现突增；ERPO 能够同时稳定查询分布和熵。**下图（960 步）：** 在长时程强化学习中，GRPO 的 Query-KL 和 Policy-KL 在约 480 步后急剧增大，准确率随之崩溃；ERPO 则始终保持稳定和较高准确率。

## 仓库结构

本仓库提供基于 [ROLL](https://github.com/alibaba/ROLL) 强化学习框架的 **ERPO 增量补丁**，并非独立的训练系统。使用时需要将其合并到完整的 ROLL 安装中。

| 路径 | 内容 |
| --- | --- |
| `roll/pipeline/rlvr/actor_worker.py` | ERPO 核心实现：`prepare_backward_batch`、动态逐查询权重、Query-KL |
| `roll/pipeline/rlvr/rlvr_config.py` | `dynamic_prompt_logp_loss_weight` 配置项 |
| `examples/erpo/` | ERPO / GRPO 受控对比 YAML 配置（2 个文件） |
| `data/erpo/` | 训练与评测 JSONL 数据，以及包含校验和的 `manifest.json` |
| `assets/` | README 使用的图片（方法概览、实验结果和训练动态） |

## 基于 ROLL 快速开始

### 1. 集成到 ROLL

你需要一个包含 ERPO 扩展点（`dynamic_prompt_logp_loss_weight` 字段、`prepare_backward_batch` 钩子和 `op_data_parallel_sum`）的 [ROLL 安装](https://github.com/alibaba/ROLL/)。可选择以下方式之一：

**方案 A：合并到完整的 ROLL 源码目录（推荐）**

```bash
git clone <your-roll-repo-with-erpo>
cd ROLL
cp -r <path-to-erpo>/roll/* roll/
cp -r <path-to-erpo>/examples/erpo examples/
cp -r <path-to-erpo>/data/erpo data/
```

**方案 B：通过 pip 安装后覆盖相关文件**

```bash
pip install roll  # 需包含 ERPO 补丁
python -c "import roll.pipeline.rlvr, os; print(os.path.dirname(roll.pipeline.rlvr.__file__))"
# 在上述输出路径中覆盖 actor_worker.py 和 rlvr_config.py
```

### 2. 数据

处理好的训练与评测数据位于 `data/erpo/`，可以直接使用：

- 训练数据：`data/erpo/train/`
- 评测数据：`data/erpo/eval/`
- 校验和与元数据：`data/erpo/manifest.json`

### 3. 启动训练

在 ROLL 项目根目录下运行：

```bash
# ERPO
python examples/start_rlvr_pipeline.py \
  --config_path examples/erpo \
  --config_name erpo_base_qwen25_7b

# GRPO 基线（关闭逐查询权重并使用响应侧 KL）
python examples/start_rlvr_pipeline.py \
  --config_path examples/erpo \
  --config_name grpo_base_qwen25_7b
```

## ERPO 与 GRPO：配置开关

在受控对比中，仅有两个配置项不同；模型、数据、随机种子、rollout 规模（128×8）、优化器批次（1024 个响应）、序列打包方式和验证集均保持一致。

| 配置项 | ERPO（`erpo_base_qwen25_7b.yaml`） | GRPO（`grpo_base_qwen25_7b.yaml`） |
| --- | --- | --- |
| `dynamic_prompt_logp_loss_weight` | `true` | `false` |
| `kl_loss_mask_mode` | `prompt` | `response` |
| `use_kl_loss` / `kl_loss_coef` | `true` / `1.0e-2` | 相同 |

## 引用

```bibtex
@misc{zhou2026erpo,
      title={Beyond the Stability-Exploration Dilemma: Environmental Regularization for LLM Policy Optimization}, 
      author={Xianlei Zhou and Xiangdi Meng and Yu He and Tianyu Qi and Shuyan Guan and Xianli Zhang and Jian Zhang and Xin Li and Qika Lin and Jun Liu},
      year={2026},
      eprint={2608.23311},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.23311}, 
}
```

## 致谢

感谢 [ROLL](https://github.com/alibaba/ROLL) 团队提供高效、可扩展的强化学习基础设施，为本工作提供了重要支持。
