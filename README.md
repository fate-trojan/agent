# Agent RL（API 策略资产版 MVP）

在闭源 API 之上搭一圈 Agent 的 RL 训练闭环：**组内采样 → 可验证奖励 + Self-Judge → 组内相对优势 → 蒸馏成规则写回 θ → 训练前后对照验证**。

## 这是什么 / 不是什么

**是**：一个**无梯度的策略改进闭环**。θ 不是神经网络权重，而是一份可读可改的**策略资产**（`base_prompt` + 规则库）。GRPO 在这里贡献的是**信用分配**（哪些轨迹更好），而不是梯度。

**不是**：梯度 RL。没有 `learning_rate`、没有 `KL(πθ‖π_ref)`、没有反向传播。原始设计稿里的这两个超参在本范式下没有对应物，已替换为真实存在的边界参数（`MAX_RULES`、`MIN_RULE_GAIN`）。

| 原始设计稿的假设 | 用闭源 API 的实际情况 | 本实现的做法 |
|---|---|---|
| 取 logprob 算 ratio | API 只给 top-N logprobs，拿不到完整分布 | 放弃 ratio，用轨迹级奖励 |
| 算 KL(πθ‖π_ref) | 权重在别人服务器上，无第二个分布可算 | 删掉该参数 |
| 反向传播更新 θ | 物理上不存在梯度 | θ = 文本资产，用 LLM 蒸馏更新 |
| `grpo.py` 是优化器 | 写出来只能是不更新任何东西的空壳 | 保留 GRPO 的**优势公式**，换掉更新方式 |

## 快速开始

```bash
cd agent
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # 填入 DEEPSEEK_API_KEY
.venv/bin/python -m app.main  # http://127.0.0.1:8000/docs
```

跑一条轨迹看当前 πθ 的水平：

```bash
curl -s -X POST localhost:8000/agent/run \
  -H 'Content-Type: application/json' \
  -d '{"task":"","task_id":"two_sum"}' | python3 -m json.tool
```

启动一轮训练（后台任务），然后轮询进度与结果：

```bash
curl -s -X POST localhost:8000/training/start \
  -H 'Content-Type: application/json' \
  -d '{"num_rollouts":24,"epochs":1,"group_size":4}'
curl -s localhost:8000/training/status
curl -s localhost:8000/training/result | python3 -m json.tool   # baseline vs final
curl -s localhost:8000/training/asset  | python3 -m json.tool   # 直接读训练出来的 θ
```

## 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 探活 + 当前 θ 摘要 |
| GET | `/tasks` | 任务清单（含隐藏用例数量，不含内容） |
| POST | `/agent/run` | 跑一条轨迹。**给了 `task_id` 才有隐藏测试、奖励才可验证** |
| POST | `/training/start` | 启动训练，后台任务 |
| GET | `/training/status` | 进度 / 错误 / θ 版本 |
| POST | `/training/stop` | 在 group 边界安全停止 |
| GET | `/training/result` | 最近一次训练的 baseline vs final |
| GET | `/training/asset` | 当前 θ（规则库全文） |
| POST | `/training/asset/reset` | 清空规则库回到 θ₀，用于复现基线 |

## θ 长什么样

训练后 `assets/codegen.json` 里的规则会被渲染进 system prompt，可以直接阅读和手改：

```
## 历史经验（由训练自动写入，按重要性排序）
1. [经验] 提交最终答案前必须自测一遍所有边界输入
2. [反例] 不要重复提交同一版失败代码
```

## 怎么判定「训练真的有效」

只认 `baseline` vs `final`（eval 集，`temperature=0` 贪心采样）。eval 的 3 道题刻意与训练集不重叠，否则分不清泛化还是背题。

**训练 reward 上涨不算数** —— 策略可以学会讨好 Judge。这也是为什么：`success` 只由 `verify_score == 1.0` 决定，Judge 分无法把失败洗成成功。

## 关键机制

- **动作协议**：模型输出 `<attempt>` / `<final>` 包裹的代码块。解析是纯文本匹配，模型不守格式时降级为 `think`（`parse_failed=True`），不抛异常，用一步换一次重新对齐的机会。
- **自由任务**（不传 `task_id`）：没有隐藏用例可判定，环境就只做「真执行 + 返回 stdout/stderr」当作观测（`run_code()`），否则模型会在 N 步里反复收到同一条无用反馈，盲迭代什么都学不到。这类任务 `verify_score` 恒为 0、`success` 恒为 false、`verifiable=false`，reward 全部来自 Judge，**不能用于训练**——没有可验证奖励就没有组内方差，优势恒为 0。
- **稀疏终点奖励**：`rewards` 除最后一步外全为 0，最后一步承载 Judge 总分。
- **通过率而非 0/1**：奖励要**有方差**才可能更新 θ。组内奖励全相同 → 优势恒为 0 → 不更新（正确行为，`status.message` 会明确提示）。
- **未提交 `<final>` 的回退**：回退到分数最高的那次 attempt，而不是直接判 0。否则「不会用协议」和「写错代码」会被混成同一种失败，污染优势估计。
- **θ 的三道闸门**：非空/长度校验 → 归一化去重 → 超限按 `gain` 剪枝。没有这三道闸门，规则库会在一轮内膨胀到吃光上下文。

## 已知风险与未验证项

1. **代码执行不是沙箱**。`run_tests()` 的隔离手段只有「独立进程 + 超时」，没有网络 / 文件系统 / 内存隔离。**绝不可把本服务暴露到不受信任的网络**。生产环境需要换成容器或 gVisor / Firecracker。
2. **Self-Judge 可被 reward hacking**。Judge 与 policy 同厂商模型，存在自我偏好。缓解手段见 `app/judge.py` 顶部注释；生产环境应把 Judge 的输入输出落审计日志、并用不同厂商的模型交叉验证。
3. **无人类确认环节**。整条链路全自动，θ 的更新没有任何人工把关。上生产前建议在写回 θ 之前插一个人工审批（这与「AI 产出不应自我批准」的原则一致）。
4. **θ 的容量天花板**。规则写多了会稀释注意力，`MAX_RULES=24` 是硬上限，也意味着这个范式的能力上限受上下文长度约束，不是无限可扩展的。
5. **未验证项**：真实 DeepSeek 链路的**模型行为**未验证（开发环境没有 Key）。已验证的是：真实 uvicorn 服务器 + 真实 HTTP 请求下的完整链路（`/agent/run`、后台训练、θ 从 v0 升到 v1 并写出规则、`/training/result` 前后对照），其中 LLM 的 HTTP 层被替换成 stub，其余代码（`_chat`、`parse_action`、`_llm_score`、`run_tests`、`_distill`、训练循环、θ 三道闸门）全部是真实执行。仍需你用真 Key 确认的是：模型是否稳定遵守 `<attempt>`/`<final>` 协议，以及 Judge 打分是否合理。
6. **成本**。每轮训练 ≈ `n_groups × group_size × (步数 + 1 次 judge)` 次调用，再加每 group 一次蒸馏和 2 次 eval 集评测。`num_rollouts=24, group_size=4, MAX_STEPS=3` 时约 150~200 次调用。

## 排障

训练跑完但 `updates_applied=0`，看 `/training/result` 的 `updates_log`，每条都会给出原因：

| 日志形态 | 含义 | 处理 |
|---|---|---|
| `优势不足(max=+0.000)` | 组内奖励全相同，GRPO 优势恒为 0 | 提高 `GROUP_TEMPERATURE`；或题目太难/太易。也可检查是否 `group_size=1` |
| `+0规则 ｜蒸馏调用失败 ...` | 蒸馏这次 LLM 调用失败（Key/限流/网络） | 看错误码；θ 不受影响，可重跑 |
| `+0规则 ｜蒸馏输出解析不出规则` | 模型没按 JSON 数组格式回答 | 重跑；若持续出现，调 `_DISTILL_PROMPT` |
| `+0规则` 无附加说明 | 蒸馏出的规则与 θ 里已有的重复，被去重闸门拦下 | 正常，说明该经验已学到 |

`baseline` 与 `final` 完全相同且 delta=0，也可能是正常的——如果 θ 的变化对该评测集无效，框架会如实报告"没有提升"，不会粉饰。

## 目录

```
app/
├── main.py      FastAPI 入口（MVP：路由合并在此）
├── config.py    配置
├── models.py    数据模型（六元组 S,A,O,R + θ + API 契约）
├── asset.py     θ：规则库 + 版本管理 + 三道闸门
├── agent.py     πθ 采样 + 可验证环境（子进程执行）+ rollout 编排
├── judge.py     Self-Judge：verifier + LLM 双层奖励
├── grpo.py      组内相对优势 + 规则蒸馏 + 训练循环
└── tasks.py     9 道题（6 训练 / 3 评测），隐藏用例对 Agent 不可见
```
