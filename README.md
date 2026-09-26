# Agent

在闭源 API 之上搭一圈 Agent 的 RL 训练闭环：**组内采样 → 可验证奖励 + Self-Judge → 组内相对优势 → 蒸馏成规则写回 θ → 训练前后对照验证**。

## 这是什么 / 不是什么

**是**：一个**无梯度的策略改进闭环**。θ 不是神经网络权重，而是一份可读可改的**策略资产**（`base_prompt` + 规则库）。GRPO 在这里贡献的是**信用分配**（哪些轨迹更好），而不是梯度。

**不是**：梯度 RL。没有 `learning_rate`、没有 `KL(πθ‖π_ref)`、没有反向传播。原始设计稿里的这两个超参在本范式下没有对应物，已替换为真实存在的边界参数（`MAX_RULES`、`MIN_RULE_GAIN`）。

## 快速开始

```bash
cd agent
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # 填入 DEEPSEEK_API_KEY
.venv/bin/python -m app.main  # http://127.0.0.1:8000/docs
```

跑一条轨迹看当前 πθ 的水平：

```bash
curl -s -X POST localhost:8000/agent/run -H 'Content-Type: application/json' -d '{"task":"","task_id":"two_sum"}' | python3 -m json.tool --no-ensure-ascii
```

> `--no-ensure-ascii` 不能省：服务端返回的就是 UTF-8 中文，是 `json.tool` 默认把非 ASCII 转义成 `\u6a21\u5f0f` 的。

启动一轮训练（后台任务），然后轮询进度与结果：

```bash
curl -s -X POST localhost:8000/training/start -H 'Content-Type: application/json' -d '{"num_rollouts":24,"epochs":1,"group_size":4}'
curl -s localhost:8000/training/status
curl -s localhost:8000/training/result | python3 -m json.tool --no-ensure-ascii   # baseline vs final
curl -s localhost:8000/training/asset  | python3 -m json.tool --no-ensure-ascii   # 直接读训练出来的 θ
```

审计用法：

```bash
curl -s -X POST localhost:8000/agent/run -H 'Content-Type: application/json' -d '{"task":"","task_id":"two_sum"}' | python3 -m json.tool --no-ensure-ascii   # 摘要（几行）
curl -s 'localhost:8000/runs?limit=5' | python3 -m json.tool --no-ensure-ascii                    # 定位某条轨迹
curl -s localhost:8000/runs/<trajectory_id> | python3 -m json.tool --no-ensure-ascii              # 看它每一步发生了什么
ls runs/                                                                                          # traj-*.jsonl / run-*.jsonl / rule-*.jsonl
```

`/agent/run` 只回摘要、`/runs/{id}` 才回完整轨迹：一条轨迹每步都带观测与计量，直接塞进上一个接口的响应会变成一堵墙。

## 通用 Agent

```bash
curl -s -X POST localhost:8000/agent/run -H 'Content-Type: application/json' -d '{"task":"你好，你是谁？"}' | python3 -m json.tool --no-ensure-ascii
```

`/agent/run` 是一个**通用 Agent 入口**，按 `task` 里是否出现代码关键词，自动选一条路径：

| 路径 | 触发条件 | 干了什么 | 有判定吗 |
|---|---|---|---|
| `codegen` | `task` 命中代码关键词（`app/agent.py:CODEGEN_KEYWORDS`） | 进 `<attempt>` / `<final>` 闭环：写代码 → 子进程执行 → 拿测试反馈 → 修正 | 有。隐藏测试 + judge 双层奖励 |
| `chat` | 一个都没命中（如"你好"、"介绍一下 X"、"为什么…"） | 一次调用直接自然语言回答，不执行任何代码 | 无 |

`chat` 的 `success` / `verifiable` 恒为 false，是**没有可验证信号**，不是答错了 —— 没有隐藏测试就没有客观判定，所以不假装有一个。

## 多轮会话记忆

`/agent/run` 默认就是**有记忆**的：连续的 curl 属于同一个 `default` 会话，自动接上上一轮的「提问 + 产出」，拼成 4 条消息发出去：

摘要第一行会带上 `｜会话 default：续用 2 条历史`，可以看出记忆有没有生效。

**只记上一轮，不是全量历史** —— 所以对话再长，prompt 也不会越滚越大。

> 注意：`default` 会话是全服务共用的。多人共用一台服务时要各自传自己的 `session_id`，否则会互相串上下文。

## 目录

```
app/
├── main.py      FastAPI 入口（MVP：路由合并在此）+ 审计端点
├── config.py    配置
├── models.py    数据模型（六元组 S,A,O,R + θ + StepRecord/Escalation + API 契约）
├── safety.py    安全红线：AST 扫描禁止操作清单 + 子进程环境变量剥离
├── env.py       可验证环境 CodeEnv(S,A,T,O,G,terminate) + 子进程执行器
├── asset.py     θ：规则库 + 版本管理 + 三道闸门 + snapshot/restore 回滚
├── llm.py       LLM 客户端的唯一构造点（惰性单例 + Key 前置校验）
├── agent.py     πθ 采样 + rollout 驱动（不产生任何梯度）
├── judge.py     Self-Judge：verifier + LLM 双层奖励 + 安全硬失败闸门
├── grpo.py      组内相对优势 + 规则蒸馏 + 训练循环 + θ 回滚 + 升级人工
├── trace.py     轨迹/运行/规则的 append-only JSONL 落盘与检索
└── tasks.py     9 道题（6 训练 / 3 评测），隐藏用例对 Agent 不可见

runs/           审计日志（traj-*.jsonl / run-*.jsonl / rule-*.jsonl，gitignore）
assets/         θ 落盘（codegen.json，gitignore）
```

