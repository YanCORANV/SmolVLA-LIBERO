> 2026-10-09 当前状态：最终 v4 评测已完成，合法初始状态为 33–49（153/153 有效 rollouts）。本文件下方的 33–52 计划和待办属于历史交接记录；请以 [最终结果](results.md) 与 [v4 冻结记录](../configs/final_confirmatory/freeze_final_protocol_v4.json) 为准。

# 新 Agent 交接：SmolVLA–LIBERO


> 2026-10-08 状态更正：paired_bc 已在开发集选为最终候选。首次最终评估启动时，State 33 曾在策略推理前因运行时 role 校验错误实例化并重置一次；未生成动作、轨迹或任务成功结果，也没有 complete.json。States 34–52 在该次失败启动中未被访问。历史冻结记录和失败目录保留，新修复另记 v2 冻结记录。
更新于 2026-09-30。此文档是该日期的历史交接摘要，当前执行以冻结的最终协议和项目负责人最新指示为准。

## 1. 立即要做的任务与当前状态

**States 23–32 开发集模型筛选评估（90 rollouts）已执行完毕，预注册门禁判定为 FAIL。**

该历史阶段尚未获准启动最终确认评测；当时的任务是汇报 anchored_language 开发集门禁失败并等待下一阶段指示。

## 2. 数据与指令严格划分状态

为防止测试集泄露与 post-hoc 偏差，项目实施严格的多阶段数据与提示词物理隔离：

### 状态切分（State Split）
* **历史早期探索状态**：States 0–2（已消费）。
* **第一轮确认评测测试状态**：States 3–22（共 20 个状态，已在 240-rollout 确认评测中消耗，永久退休，不得用于微调或筛选）。
* **当前模型筛选开发状态**：States 23–32（共 10 个状态，用于刚刚完成的 90-rollout 门禁评测）。
* **最终确认状态**：States 33–52（共 20 个状态）。State 33 在失败的基础设施启动中于策略推理前被实例化并重置一次；未观察到动作、轨迹或成功结果。States 34–52 在该次启动前未被访问。

### 提示词切分（Prompt Split）
* **训练指令（5 条；以 `configs/instructions/task0_train.jsonl` 为准）**：
  * `train_original`：`"pick up the black bowl between the plate and the ramekin and place it on the plate"`
  * `train_paraphrase_01`：`"pick up the black bowl that sits between the plate and the ramekin, and put it on the plate"`
  * `train_paraphrase_02`：`"take the black bowl between the plate and the ramekin and set it down on the plate"`
  * `train_paraphrase_03`：`"move the black bowl from between the plate and the ramekin onto the plate"`
  * `train_paraphrase_04`：`"find the black bowl between the plate and the ramekin, pick it up, and place it on the plate"`
* **第一轮确认消耗指令（2 条，永久退休）**：
  * `paraphrase_heldout_01`: `"transfer into the plate the black bowl that sits between the plate and the ramekin"`
  * `paraphrase_heldout_02`: `"take the black bowl located between the ramekin and the plate, and set it down on the plate"`
* **当前开发筛选指令（2 条）**：
  * `paraphrase_dev_01`: `"grasp the black bowl positioned between the plate and the ramekin and put it onto the plate"`
  * `paraphrase_dev_02`: `"set onto the plate the black bowl situated between the ramekin and the plate"`
* **当时保留、后来退役的最终指令（2 条；以下为历史记录）**：
  * 存储于 `configs/final_confirmatory_reserved/instructions/task0_final_reserved.jsonl`
  * `paraphrase_final_01`: SHA256 `8635d0fc3c89ac4ef1c606473a151cec56a331bd0b4baadece9fb0f9ab873fa7`
  * `paraphrase_final_02`: SHA256 `414a27e90ec3244636f1e456f999474e2288805549953d4c9794ab285f36701b`

## 3. 已完成的关键阶段与最新结果

### A. 第一轮冻结确认评测（States 3–22，240 rollouts）
* 运行目录：`outputs/held_out_evaluation/runs/task0_confirmatory_heldout_eval/`
* 核心发现（鲁棒性与保真度权衡）：
  * `original_language`: 原句成功率 0.95 (19/20)，未见释义泛化 0.25 (10/40)。
  * `augmented_language`: 原句成功率 0.80 (16/20)，未见释义泛化 0.375 (15/40)。
  * 结论：均匀语言增强提升了未见指令鲁棒性，但显著牺牲了原句规范表现。

### B. `anchored_language` 训练（1000 updates）
* 训练目录：`outputs/adaptation/runs/task0_anchored_training/anchored_language/step_001000/attempt_001/`
* 配置：与 `augmented_language` 优化超参完全相同（AdamW, lr 1e-5 cosine, batch 32, 1000 steps），但将原句采样比例锚定为 50%，其余 4 种训练改写各占 12.5%。

### C. 冻结模型筛选门禁（States 23–32，90 rollouts）
* 评测协议：3 模型 × 10 状态 × 3 指令（原句、dev_01、dev_02）= 90 rollouts。固定 280 步，单 worker 串行执行。
* 指标定义：
  * $O$: States 23–32 原句成功率（分母 10）。
  * $P$: dev_01 与 dev_02 合并泛化成功率（分母 20）。
  * $H = \frac{2 \cdot O \cdot P}{O + P}$（调和平均综合指标；若 $O+P=0$ 则 $H=0$）。
  * 门禁规则（须全部满足）：
    1. $O_{\text{anchor}} \ge O_{\text{augmented}}$
    2. $P_{\text{anchor}} \ge P_{\text{original\_language}}$
    3. $H_{\text{anchor}} \ge \max(H_{\text{original\_language}}, H_{\text{augmented\_language}})$
* 实测结果汇总（`goal_ever`）：

  | 模型 | 原句 $O$ (10) | dev_01 (10) | dev_02 (10) | 合并释义 $P$ (20) | 综合得分 $H$ | 鲁棒差距 $(P - O)$ |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **original_language** | **0.90** (9/10) | 0.70 (7/10) | 0.10 (1/10) | **0.40** (8/20) | **0.5538** | -0.50 |
  | **augmented_language** | **0.80** (8/10) | 0.30 (3/10) | 0.30 (3/10) | **0.30** (6/20) | **0.4364** | -0.50 |
  | **anchored_language** | **0.60** (6/10) | 0.30 (3/10) | 0.40 (4/10) | **0.35** (7/20) | **0.4421** | -0.25 |

* 门禁判定详情：
  1. $O_{\text{anchor}} \ge O_{\text{augmented}}$: $0.60 \ge 0.80$ $\to$ **不满足 (FAIL)**
  2. $P_{\text{anchor}} \ge P_{\text{original\_language}}$: $0.35 \ge 0.40$ $\to$ **不满足 (FAIL)**
  3. $H_{\text{anchor}} \ge \max(H_{\text{orig}}, H_{\text{aug}})$: $0.4421 \ge \max(0.5538, 0.4364) = 0.5538$ $\to$ **不满足 (FAIL)**
* 最终筛选结论：**FAIL（未通过）**。

## 4. 文件与记录位置

- 工作目录：`D:\SmolVLA-LIBERO`（WSL2 `/mnt/d/SmolVLA-LIBERO`）。
- 确认评测输出：`outputs/held_out_evaluation/runs/task0_confirmatory_heldout_eval/`。
- 开发集模型筛选输出：`outputs/development_evaluation/runs/task0_anchored_dev_eval/`
  - 汇总指标：`summary.json`
  - 逐状态记录：`per_state_raw_outcomes.csv`
  - 实验元数据：`experiment_meta.json`
- 最终确认保留配置：`configs/final_confirmatory_reserved/`。

## 5. 工作边界和纪律（遵守 AGENTS.md）

1. **历史阶段边界**：当时未获准在 States 33–52 上评测；此限制已由最新冻结最终协议和项目负责人后续指示取代。
2. **严禁事后修改规则**：门禁在观测前已冻结，FAIL 结论不可通过修改权重、调整指标定义或更换状态重新解释。
3. **如实汇报与等待指示**：将开发集真实观测与失败细节完整呈现给项目负责人，下一步动作需负责人明确指示。
