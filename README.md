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
| GET | `/runs` | 最近的轨迹摘要 + 训练运行 + 规则写入记录（审计） |
| GET | `/runs/{trajectory_id}` | 按 id 取回一条完整轨迹（每步动作、状态、判定、计量） |

审计用法：

```bash
curl -s 'localhost:8000/runs?limit=5' | python3 -m json.tool     # 定位某条轨迹
curl -s localhost:8000/runs/<trajectory_id> | python3 -m json.tool  # 看它每一步发生了什么
ls runs/                                                        # traj-*.jsonl / run-*.jsonl / rule-*.jsonl
```

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

### 环境与边界

- **环境是显式形式化的**（`app/env.py`，`CodeEnv`）：`S` = attempts / best / denied / step，`A` = `execute_code(code)` 与提交最终答案，`T` = `dispatch()` 唯一的转移入口，`O` = 有隐藏用例时给逐用例的期望值/实际值、自由任务给真实 stdout/stderr，`G` = 隐藏测试全通过，`terminate` = 模型提交或达到 `MAX_STEPS`。状态在每次转移里**真实改变**，不是摆设。
- **每次转移先过校验与安全闸门**：工具名、参数结构、空代码、字符上限、安全红线，任一不过就 `denied`（记 `deny_reason`），不进入执行。
- **执行的四层边界**：独立进程（崩溃不带走服务）+ 超时（切断死循环）+ **干净环境变量**（剥离 `os.environ`，否则被测代码一句 `os.environ["DEEPSEEK_API_KEY"]` 就能把密钥读走）+ `-I` 隔离模式（忽略 `PYTHON*` 环境变量与用户 site-packages）。

### 安全红线（禁止操作清单，不是沙箱）

命中的处理方式是**判定失败、reward 归零**，而不是扣分——否则一次越界会被 0.3 权重的过程分洗白。命中后 `safety_passed` 永久置否（不因后续自愈而洗白）。

- 检查手段是 **AST 而不是正则**：正则会被 `__import__("o"+"s")` 这类拼接绕过，AST 看到的是真实的 import 与调用节点。
- 覆盖：禁止模块导入（`os`/`sys`/`socket`/`subprocess`/`shutil`/`pathlib`/`requests`…）、禁止调用（`open`/`input`/`eval`/`exec`/`compile`/`__import__`/`globals`…）、禁止访问的逃逸入口属性（`__globals__`/`__subclasses__`/`__builtins__`/`__mro__`…）。
- 它能挡住的是**模型在正常解题中顺手写出的越界操作**，挡不住真正的恶意代码。见下方风险清单第 1 条。

### 步级信用分配

轨迹级的优势下钻到「哪一步带来了提升」：每个 `StepRecord` 记 `verify_delta`（相对此前最好成绩的增量）与 `credited`（是否被最终采用）。蒸馏时把 `第1步 s=0.40 Δ=+0.40；第2步 s=1.00 Δ=+0.60｜最终采用 第2步` 喂进提示词，让 LLM 优先总结**带来提升的那一步**，而不是复述最终代码。

### θ 质量闸门：变差就回滚

一轮训练前对 θ 拍快照。训练后若 eval 变差（主判据 `avg_verify_score`，因为成功率常是 0%↔0% 没有分辨率），自动回滚到快照并**重新评测**，让 `final` 反映「线上现在真正生效的 θ」；变差的那份数据留在 `regressed_final` 供复盘。版本号回滚后**递增**而非回退——版本号的含义是「写入次数」，回滚本身也是一次写入。

### 自治边界：什么交给人工

框架自己处理重试、降级、回滚、停机；只有下面几类事件会写进 `escalations` 并把 `needs_human` 置真（同时自动停机，避免继续烧钱）：

| `code` | 触发条件 |
|---|---|
| `safety_violation` | 有轨迹命中安全红线 |
| `regression` | eval 变差（已自动回滚，但仍需人查蒸馏提示词） |
| `no_learning_signal` | 连续 `MAX_FLAT_GROUPS` 个 group 组内奖励无方差，优势恒为 0 |
| `budget_exceeded` | token 用量达到 `TOKEN_BUDGET`（0 表示不限） |
| `llm_failure` | 有轨迹因 LLM 调用失败被丢弃（单条失败不会炸整轮） |
| `unexpected_error` | 训练循环抛异常 |

### 可审计

三类 append-only JSONL（`runs/`），可以 `grep` / `diff` / `jq` 直接消费，不引入数据库依赖：

| 文件 | 内容 |
|---|---|
| `traj-YYYYMMDD.jsonl` | 一条完整轨迹：每步动作、环境状态、观测、耗时、token、是否被拒、安全违规 |
| `run-YYYYMMDD.jsonl` | 一次训练运行：baseline vs final、回滚、升级人工事件 |
| `rule-YYYYMMDD.jsonl` | 一次规则写入：规则来自哪一轮、哪几条轨迹、哪一步 |

轨迹里存的是隐藏用例的**指纹**（sha256 前 16 位）而不是用例正文——正文在 `tasks.py` 里由 git 版本化，指纹足以回答「这次判定用的是哪一版用例」，又不会让轨迹文件膨胀。写盘失败会被吞掉并返回空路径：审计是旁路，不能反噬主流程。

## 已知风险与未验证项

1. **代码执行不是沙箱**。`_run_python()` 的隔离手段只有「独立进程 + 超时 + 干净环境变量 + `-I` 模式」，没有网络 / 文件系统 / 内存 / 内核级隔离；`app/safety.py` 是一份**禁止操作清单**，不是沙箱。**绝不可把本服务暴露到不受信任的网络**。生产环境需要换成容器或 gVisor / Firecracker。
2. **Self-Judge 可被 reward hacking**。Judge 与 policy 同厂商模型，存在自我偏好。缓解手段见 `app/judge.py` 顶部注释；生产环境应把 Judge 的输入输出落审计日志（本实现已落 `runs/traj-*.jsonl`）、并用不同厂商的模型交叉验证。
3. **人类仅在异常时才被拉进来**。整条链路自动运行，θ 的更新没有逐步人工审批；只有安全红线 / 回归 / 无学习信号 / 预算超支 / 异常这五类事件会让训练**自动停机并把 `needs_human` 置真**（见 `escalations`）。日常的规则写入仍是全自动的——上生产前建议对规则写回加一道人工审批（这与「AI 产出不应自我批准」的原则一致）。
4. **θ 的容量天花板**。规则写多了会稀释注意力，`MAX_RULES=24` 是硬上限，也意味着这个范式的能力上限受上下文长度约束，不是无限可扩展的。
5. **未验证项**：真实 DeepSeek 链路的**模型行为**未验证（开发环境没有 Key）。已验证的是：真实 uvicorn 服务器 + 真实 HTTP 请求下的完整链路，以及上面这些机制的离线回归 —— 安全红线拦截与 reward 归零、`verify_delta` 步级信用、θ 变差自动回滚 + 回滚后复测、五类 escalation 触发、轨迹/规则 JSONL 落盘可读回、困在死循环被超时切断、自由任务拿到真实执行反馈。测试中 LLM 的 HTTP 层被替换成 stub，其余代码全部真实执行。仍需你用真 Key 确认的是：模型是否稳定遵守 `<attempt>`/`<final>` 协议，以及 Judge 打分是否合理。
6. **成本**。每轮训练 ≈ `n_groups × group_size × (步数 + 1 次 judge)` 次调用，再加每 group 一次蒸馏和 2 次 eval 集评测。`num_rollouts=24, group_size=4, MAX_STEPS=3` 时约 150~200 次调用。安全红线命中时**不会调用 judge**（直接判失败），这一条路径是省钱的。

## 排障

训练跑完但 `updates_applied=0`，看 `/training/result` 的 `updates_log`，每条都会给出原因：

| 日志形态 | 含义 | 处理 |
|---|---|---|
| `优势不足(max=+0.000)` | 组内奖励全相同，GRPO 优势恒为 0 | 提高 `GROUP_TEMPERATURE`；或题目太难/太易。也可检查是否 `group_size=1` |
| `连续 N 个 group 无方差，停机并升级人工` | 同上且已连续 N 次 → 自动停机（`no_learning_signal`） | 调题或调温度后重跑 |
| `+0规则 ｜蒸馏调用失败 ...` | 蒸馏这次 LLM 调用失败（Key/限流/网络） | 看错误码；θ 不受影响，可重跑 |
| `+0规则 ｜蒸馏输出解析不出规则` | 模型没按 JSON 数组格式回答 | 重跑；若持续出现，调 `_DISTILL_PROMPT` |
| `+0规则` 无附加说明 | 蒸馏出的规则与 θ 里已有的重复，被去重闸门拦下 | 正常，说明该经验已学到 |
| 有效样本 `<2` 条 | 该 group 内 LLM 调用失败太多 | 看 `escalations` 里的 `llm_failure` 明细 |

`baseline` 与 `final` 完全相同且 delta=0，也可能是正常的——如果 θ 的变化对该评测集无效，框架会如实报告"没有提升"，不会粉饰。

`/training/result` 里 `rolled_back=true` 表示本轮 θ 把 eval 做差了，已自动回滚：`regressed_final` 是变差的那份数据，`final` 是回滚后重测（即线上实际生效的 θ）。

## 目录

```
app/
├── main.py      FastAPI 入口（MVP：路由合并在此）+ 审计端点
├── config.py    配置
├── models.py    数据模型（六元组 S,A,O,R + θ + StepRecord/Escalation + API 契约）
├── safety.py    安全红线：AST 扫描禁止操作清单 + 子进程环境变量剥离
├── env.py       可验证环境 CodeEnv(S,A,T,O,G,terminate) + 子进程执行器
├── asset.py     θ：规则库 + 版本管理 + 三道闸门 + snapshot/restore 回滚
├── agent.py     πθ 采样 + rollout 驱动（不产生任何梯度）
├── judge.py     Self-Judge：verifier + LLM 双层奖励 + 安全硬失败闸门
├── grpo.py      组内相对优势 + 规则蒸馏 + 训练循环 + θ 回滚 + 升级人工
├── trace.py     轨迹/运行/规则的 append-only JSONL 落盘与检索
└── tasks.py     9 道题（6 训练 / 3 评测），隐藏用例对 Agent 不可见

runs/           审计日志（traj-*.jsonl / run-*.jsonl / rule-*.jsonl，gitignore）
assets/         θ 落盘（codegen.json，gitignore）
```

