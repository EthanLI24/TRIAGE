<p align="center">
  <img src="assets/logo.svg" alt="TRIAGE logo" width="240">
</p>

# TRIAGE: Direction-Aware Mismatch Stabilization of Native NVFP4 Reinforcement Learning

**面向稳定原生 NVFP4 强化学习训练的 Slime 策略损失扩展。**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python ≥ 3.10](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)](pyproject.toml)

[English](README.md) · [文档](docs/TRAINING.md) · [Benchmark](benchmarks/README.md) · [失配刻画](docs/CHARACTERIZATION.md)

> **一句话概述：** TRIAGE 通过方向感知的 **Gate** 和有界的 **Repair** 控制 learner–sampler 失配，在保留端到端原生 W4A4 前向执行的同时，让原生 NVFP4 强化学习保持稳定。

## 亮点

- **稳定性** — 配合 TRIAGE，原生 NVFP4 RL 在 Qwen3-4B 上稳定训练 600 步、在 Qwen3-30B-A3B 上稳定训练 1,700 步；朴素 NVFP4 约在 300 步（4B）/ 700 步（30B-A3B）失稳；仅加截断重要性采样（TIS）在 4B 上仍会崩溃，在 30B-A3B 上虽能存活但付出约 9 个点的精度代价。
- **效率** — NVFP4 执行在 8×B300 上相对 BF16 取得 2.08–2.35× rollout 吞吐、1.21–1.33× 端到端迭代加速；TRIAGE 保留 2.27–2.30× 的 rollout 吞吐，learner 更新开销中位数仅 0.84%。
- **精度** — 达到 BF16 水平的 benchmark 平均分：4B 上 58.49 对 58.26（高于 BF16），30B-A3B 上 70.96 对 72.41（−1.45）；而 NVFP4+TIS 降至 50.76（4B，崩溃前的 checkpoint）/ 63.18（30B-A3B）。

![TRIAGE 总览：方向性 Gate、有界 Repair 与实测结果](assets/figures/triage-overview.png)

*TRIAGE 在短 response segment 中诊断 learner–sampler 失配，对方向上会放大 gap 的更新进行门控，并修复残余的负向失配——在不离开原生 W4A4 执行路径的前提下获得稳定性。*

## 动态

- **[2026/09] v1.0.0** — TRIAGE 首次公开发布：策略损失扩展、示例启动脚本、benchmark 结果与失配表征研究。

## 为什么需要 TRIAGE

仅看失配大小，无法判断下一次策略更新是在修复还是放大 learner–sampler 的概率差异：即使训练已经开始漂移，绝大多数 token 的 gap 仍然很小。失配的*方向*才是更早的信号——按优势加权的 token 质量早在边缘 gap 分布恶化之前就偏向了负向（放大）一侧。TRIAGE 正是基于这一信号：在短 segment 中诊断风险，选择性地门控放大型更新，并修复残余负向失配；它与 PPO/TIS 配合使用，支持 dense 和 MoE 模型。

![方向不对称性先于崩溃上升](assets/figures/characterization/directional-asymmetry.png)

*方向不对称比 ρ_asym 在 pre-terminal 窗口已从 1.02 升至 2.32（4B）、从 1.34 升至 1.85（30B-A3B），而此时仍有至少 83.0% / 91.9% 的 token 处于 |δ| < 0.05 的小幅区间——方向的恶化远早于幅度的恶化。完整研究见 [docs/CHARACTERIZATION.md](docs/CHARACTERIZATION.md)。*

## 安装

TRIAGE 分为两档安装。

**(a) 损失扩展包与 CPU 测试** — 任意装有 Python 3.10+、PyTorch、NumPy 的机器：

```bash
python -m pip install -e .
bash scripts/run_tests.sh
```

**(b) 完整训练栈** — 需要固定在指定 commit 的 Slime、SGLang、Megatron-LM 与 miniTransformer，NVFP4 执行还需要 Blackwell 代 GPU。miniTransformer 为内部发布版本、无公开 URL，仅 MoE（Qwen3-30B-A3B）的 NVFP4 执行路径需要它；dense 4B 路径不需要。环境配置见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)，固定版本与补丁内容见 [docs/STACK_AND_CHANGES.md](docs/STACK_AND_CHANGES.md)。

## 快速开始

### 在现有 Slime 启动命令上启用 TRIAGE

```bash
# 1. 应用暴露 TRIAGE loss 钩子的源码级补丁
#    （幂等且带哈希门控；将 SLIME_ROOT 指向你的 Slime 目录）。
export SLIME_ROOT=/path/to/slime
bash scripts/apply_patches.sh tokenmean

# 2. 让扩展可被 import。
export PYTHONPATH="$PWD/policy_extension:${PYTHONPATH:-}"

# 3. 启用 Gate 和 Repair，并采用参考的方向性作用范围。
export TRIAGE_GATE_ENABLE=1 TRIAGE_REPAIR_ENABLE=1
export TRIAGE_GATE_SCOPE=q3 TRIAGE_REPAIR_SCOPE=adv_positive TRIAGE_RECENTER_ENABLE=0

# 4. 在现有 Slime 启动参数中加入 TRIAGE loss。
#    --calculate-per-token-loss 为必需参数：缺少它 triage_custom_loss 会
#    直接报错。报告的 TRIAGE 配置与 TIS 一起运行，因此这里同时给出
#    --use-tis 及其 clip 边界。
python3 "$SLIME_ROOT/train.py" ... \
  --calculate-per-token-loss \
  --use-tis --tis-clip 2.0 --tis-clip-low 0.0 \
  --loss-type custom_loss \
  --custom-loss-function-path TRIAGE.triage_custom_loss \
  --rollout-data-postprocess-path TRIAGE.triage_rollout_postprocess
```

在远程 Ray actor 上，`TRIAGE_*` 变量必须通过 Slime 的 `--train-env-vars`
转发到训练 actor；下方示例启动脚本通过 `scripts/train_env.py` 自动完成
序列化与转发（见 [docs/TRAINING.md](docs/TRAINING.md)）。

### 端到端示例脚本

无需 GPU 或训练栈即可查看完整解析后的命令：

```bash
DRY_RUN=1 bash examples/train_triage.sh
```

设置公共路径后选择训练模式：

```bash
export SLIME_ROOT=/path/to/slime
export SGLANG_ROOT=/path/to/sglang
export MEGATRON_ROOT=/path/to/Megatron-LM
export MODEL_ROOT=/path/to/models
export PROMPT_DATA=/path/to/dapo-math-17k.jsonl

bash examples/train_bf16.sh      # 208.2 秒/迭代，rollout 2,977 tokens/s
bash examples/train_nvfp4.sh     # 169.4 秒/迭代，rollout 6,201 tokens/s
bash examples/train_triage.sh    # 172.1 秒/迭代，rollout 6,745 tokens/s
```

耗时数据（以 `MODEL=qwen3-30B-A3B R3=0` 测得；脚本默认为 Qwen3-4B）：
Qwen3-30B-A3B-Base，单节点 8×B300，global batch 256，自然 EOS、最长
20,480 token，预热 2 次后取 10 次均值，关闭 R3 capture。默认脚本会保存
checkpoint，并非计时配方；benchmark 口径见
[benchmarks/README.md](benchmarks/README.md)。三个脚本共用
[`common.sh`](examples/common.sh)，默认 Qwen3-4B-Base、单节点八卡、
batch 256、最大回答长度 16,384。`MODEL=qwen3-30B-A3B` 选择 MoE；
`USE_TIS=1` 配合 `examples/train_nvfp4.sh` 即为 NVFP4+TIS baseline
（`train_triage.sh` 已默认开启）；`START_LOCAL_RAY=1` 启动本地 Ray head。
checkpoint、续训、评估与运行时配置见 [docs/TRAINING.md](docs/TRAINING.md)。

### Gate 与 Repair 配置

算法默认值统一定义在 [`triage_config.py`](policy_extension/triage_config.py)；
启动脚本只保留启用开关，不在 shell 中重复维护调参值。

| 控制项 | 参数 | 默认值／可选值 |
|---|---|---|
| Gate 范围 | `TRIAGE_GATE_SCOPE` | 默认 `whole`；`q3` 仅选择负 gap token |
| Repair 范围 | `TRIAGE_REPAIR_SCOPE` | `whole`、`adv_positive`、`q2`，默认 `whole` |
| Repair 强度 | `TRIAGE_REPAIR_COEF` | `0.015` |
| Repair 起始步 | `TRIAGE_REPAIR_START_STEP` | `0`，包含边界 |

可选的 Gate 校准默认关闭：

```bash
python scripts/calibrate_gate.py --input /path/to/reference_jsonl \
  --output /shared/calibration/gate.json
TRIAGE_CALIBRATION_PATH=/shared/calibration/gate.json bash examples/train_triage.sh
```

参考窗采集与冻结阈值的使用方式见 [docs/CALIBRATION.md](docs/CALIBRATION.md)。

## 实验结果

![训练过程中的 entropy、reward 与 learner–sampler gap](assets/figures/training-dynamics.png)

*朴素 NVFP4 在两个模型上都崩溃（30B-A3B 约 700 步、4B 约 300 步）；NVFP4+TIS 推迟了失败，但在 4B 上仍然崩溃，在 30B-A3B 上虽能存活却以牺牲精度为代价；TRIAGE 在两个模型上都于完整的 1,700 / 600 步内保持 entropy、reward 与 learner–sampler gap 稳定。*

![Gate 与 Repair 消融](assets/figures/ablation.png)

*每移除一个组件，稳定性就进一步下降：从 step-300 的 TRIAGE checkpoint 分支，三个组件全部移除（朴素 NVFP4）在窗口内崩溃，而 Gate+TIS 与 TIS 虽能存活，但逐级低于完整 TRIAGE。*

吞吐量与完整 benchmark 表格：[benchmarks/README.md](benchmarks/README.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/TRAINING.md](docs/TRAINING.md) | 启动模式、checkpoint、续训、评估与运行时配置 |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | 完整训练栈的环境配置 |
| [docs/METHODS.md](docs/METHODS.md) | 代码中 Gate 与 Repair 的实现机制 |
| [docs/CHARACTERIZATION.md](docs/CHARACTERIZATION.md) | 设计背后的失配刻画研究 |
| [docs/CALIBRATION.md](docs/CALIBRATION.md) | 可选的 Gate 参考窗校准 |
| [docs/STACK_AND_CHANGES.md](docs/STACK_AND_CHANGES.md) | 固定的栈版本与补丁内容 |
| [docs/PAPER_AND_CODE.md](docs/PAPER_AND_CODE.md) | 论文与本代码库的对应关系 |
| [docs/VALIDATION.md](docs/VALIDATION.md) | 已执行的检查与待完成的硬件验证 |
| [benchmarks/README.md](benchmarks/README.md) | 测量口径与完整结果表 |

## 路线图与已知限制

按大致优先级规划：

- 逐项消融的数值汇总，以及独立的 final-checkpoint 评估入口。
- 按 response 加权均值归一化的对齐（本分支当前使用全局 weighted-token 归一化，见 [docs/PAPER_AND_CODE.md](docs/PAPER_AND_CODE.md)）。
- 将 Slime/SGLang 兼容性补丁上游化。
- 发布报告运行的参考 checkpoint。

已知限制：NVFP4 执行需要 Blackwell 代 GPU（上述吞吐量数据在单节点
8×B300 上测得）。BF16 训练可在更早的 NVIDIA 硬件上运行；损失扩展包及其
CPU 测试可在任意机器上运行。

## 常见问题

**需要什么硬件？** 损失扩展包、CPU 测试和 `DRY_RUN=1` 的命令预览可在任意机器上运行。NVFP4 训练需要 Blackwell 代 GPU；BF16 训练支持更早的 NVIDIA GPU。

**启动前如何检查环境？** 运行 `python scripts/preflight.py --mode triage --model qwen3-30B-A3B --r3 1`（启动脚本也会自动执行）。它会在占用任何 GPU 之前校验栈版本、补丁状态和 overlay 哈希。

**补丁可以重复应用吗？** 可以。补丁幂等且带哈希门控：preflight 能区分原始与已打补丁的源码，对已打补丁的目录重复应用是无操作。

**如何重新校准 Gate 阈值？** 按 [docs/CALIBRATION.md](docs/CALIBRATION.md) 操作：用 `scripts/calibrate_gate.py` 导出参考窗，然后将 `TRIAGE_CALIBRATION_PATH` 指向冻结阈值。

**支持哪些模型？** 启动脚本自带 Qwen3-4B-Base（dense）和 Qwen3-30B-A3B-Base（MoE）的配置；损失本身与模型无关。MoE 的 NVFP4 执行路径还需要 miniTransformer（内部发布版本，无公开 URL）；dense 4B 路径不需要。

## 引用

```bibtex
@misc{triage2026,
  title  = {TRIAGE: Direction-Aware Mismatch Stabilization of Native NVFP4 Reinforcement Learning},
  author = {TRIAGE Contributors},
  year   = {2026},
  note   = {Anonymous code release},
}
```

论文的 BibTeX 条目将在发表后补充于此。

## 致谢

TRIAGE 基于 [Slime](https://github.com/THUDM/slime)（RL 框架）、
[SGLang](https://github.com/sgl-project/sglang)（rollout 与服务）、
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)（训练后端）与
[Transformer Engine](https://github.com/NVIDIA/TransformerEngine)（NVFP4 执行）
构建。感谢这些项目的作者与贡献者。

## 许可证

Apache-2.0。见 [LICENSE](LICENSE)；代码再分发范围见 [NOTICE](NOTICE.md)。
