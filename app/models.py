"""数据模型（MVP 裁剪版）。

保留原设计的 Agent 六元组形式化 (S, A, P, R, O, π)：
    S 状态空间 -> State
    A 动作空间 -> Action
    O 观测空间 -> Observation
    R 奖励     -> Trajectory.rewards / total_reward
    P 转移     -> app/agent.py 的 rollout 实现
    π 策略     -> app/agent.py + app/asset.py
"""

import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, computed_field


# ===================== 六元组 =====================


class State(BaseModel):
    """状态空间 S：某一决策点的环境状态快照。只存 env_state，不存对话历史。"""

    env_state: Dict[str, Any] = Field(default_factory=dict)
    step: int = 0


class Action(BaseModel):
    """动作空间 A。"""

    type: Literal["think", "tool_call", "answer"]
    content: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    #: 解析器未识别出合法动作时置位，用于统计协议失效率
    parse_failed: bool = False


class Observation(BaseModel):
    """观测空间 O：环境反馈。success 是环境的一阶判定。"""

    content: str
    success: bool


class SafetyViolation(BaseModel):
    """一条被触发的安全红线。"""

    rule: str
    detail: str
    line: int = 0


class StepRecord(BaseModel):
    """单步的审计与信用记录：#12 的动作/参数/耗时/token/错误，#10 的步级信用。"""

    step: int
    action_type: str
    parse_failed: bool = False
    tool_name: Optional[str] = None

    # ---- 环境判定 ----
    passed: int = 0
    total: int = 0
    verify_score: float = 0.0
    verify_delta: float = 0.0          # 相对此前最好成绩的增量
    credited: bool = False             # 该步的代码被作为最终提交

    # ---- 越界与安全 ----
    denied: bool = False               # 被拒绝执行（未进入环境）
    deny_reason: str = ""
    error: str = ""                    # 本步的失败说明（协议未识别 / 执行报错）

    # ---- 计量 ----
    llm_latency_ms: int = 0
    exec_latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: DeepSeek 前缀缓存命中/未命中的输入 token（计费差 10~70 倍，必须单独看）
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    started_at: str = ""
    finished_at: str = ""


class Trajectory(BaseModel):
    """完整轨迹 τ = (s₀, a₀, o₀, r₀, ..., s_T)。稀疏终点奖励：最后一步承载总奖励。"""

    task_id: str
    task_prompt: str = ""

    # ---- 可审计标识 ----
    trajectory_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    run_id: str = ""                   # 所属训练轮次；独立推理时为空
    session_id: str = ""               # 多轮会话标识；空 = 单轮无记忆
    started_at: str = ""
    finished_at: str = ""
    duration_ms: int = 0
    trace_path: str = ""               # 落盘位置，便于回溯

    states: List[State] = Field(default_factory=list)
    actions: List[Action] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)
    rewards: List[float] = Field(default_factory=list)
    steps: List[StepRecord] = Field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False

    # ---- 本轮走哪条路径 ----
    #: codegen = 代码生成闭环（有执行、有判定、有奖励）；chat = 通用对话（直接回答）
    mode: Literal["codegen", "chat"] = "codegen"
    #: chat 模式的回答正文；codegen 模式为空 —— 两者分开存，记忆回放时才不会
    #: 把一段闲聊当成代码塞进 ```python 围栏里
    reply: str = ""

    # ---- 训练拆解信息 ----
    final_code: str = ""
    verify_score: float = 0.0                      # 可验证奖励 = 隐藏测试通过率
    verify_detail: List[Dict[str, Any]] = Field(default_factory=list)
    judge_score: float = 0.0                       # LLM judge 过程分
    judge_rationale: str = ""
    advantage: float = 0.0                         # 组内相对优势 A_i
    asset_version: int = 0                         # 采样时使用的 θ 版本
    used_fallback_code: bool = False               # 未输出 <final>，回退到最优 attempt
    critical_step: int = -1                        # 最终代码由第几步产生（步级归因）

    # ---- 安全闭环 ----
    safety_passed: bool = True
    safety_violations: List[SafetyViolation] = Field(default_factory=list)

    # ---- 成本计量 ----
    #: 以下四项只统计策略自身的调用（judge / 蒸馏的 prompt 每次都不同，缓存命中率
    #: 恒接近 0，单独记在 judge_tokens 里，混进来只会稀释这里的命中率）
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    judge_tokens: int = 0                          # judge 自身的用量，与策略用量分开记

    @computed_field  # 派生的，不能当输入传；没有 computed_field 它不会出现在 JSON 里
    @property
    def cache_hit_rate(self) -> float:
        """输入 token 的缓存命中率。训练时 system prompt 固定，这个值应该很高。"""
        total = self.cache_hit_tokens + self.cache_miss_tokens
        return round(self.cache_hit_tokens / total, 4) if total else 0.0

    # ---- 环境终态（可审计环境状态）----
    final_env_state: Dict[str, Any] = Field(default_factory=dict)


# ===================== 策略资产 θ =====================


class Rule(BaseModel):
    """θ 的一个可训练单元。文本即参数。"""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    text: str
    kind: Literal["success", "antipattern"] = "success"
    origin_task: str = ""
    gain: float = 0.0          # 写回 θ 时的组内优势估计，用于剪枝
    asset_version: int = 0

    # ---- 经验溯源：这条规则是从哪一次运行、哪几条轨迹、哪一步蒸馏出来的 ----
    origin_run_id: str = ""
    origin_trajectory_ids: List[str] = Field(default_factory=list)
    origin_step: int = -1      # 证据来自该轨迹的第几步（步级信用）
    created_at: str = ""


class PolicyAsset(BaseModel):
    """策略资产 θ = base_prompt + rules。πθ 的输出分布完全由 render() 出来的 system prompt 决定。"""

    task_type: str = "codegen"
    version: int = 0
    base_prompt: str = ""
    rules: List[Rule] = Field(default_factory=list)
    updated_at: str = ""

    def render(self) -> str:
        if not self.rules:
            return self.base_prompt
        lines = ["", "## 历史经验（由训练自动写入，按重要性排序）"]
        for i, r in enumerate(self.rules, 1):
            tag = "反例" if r.kind == "antipattern" else "经验"
            lines.append(f"{i}. [{tag}] {r.text}")
        lines.append("")
        lines.append("以上经验来自你此前在同类任务上的失败与成功，请优先遵守。")
        return self.base_prompt + "\n".join(lines)


# ===================== 任务 =====================


class TaskSpec(BaseModel):
    """任务规格。tests 是"可验证"的来源，对 Agent 不可见。"""

    id: str
    prompt: str
    entry_point: str = "solution"
    #: (表达式字符串, 期望值)；表达式里可直接调用 solution
    tests: List[Tuple[str, Any]] = Field(default_factory=list)
    difficulty: str = "medium"
    split: Literal["train", "eval"] = "train"


# ===================== API 契约 =====================


class AgentRequest(BaseModel):
    task: str = Field(..., description="任务描述")
    task_id: Optional[str] = Field(
        default=None,
        description="任务 ID。给了才有隐藏测试与可验证奖励；"
        "不给则退化为纯 LLM-judge 评分，奖励不可验证。",
    )
    max_steps: int = 3
    temperature: float = 0.7
    session_id: str = Field(
        default="default",
        description="多轮会话标识。不给就是 default 会话 —— 连续的 curl 会自动接上"
        "上一轮的提问与代码，「显示一下刚才的结果」这类指代才成立。"
        "传空串 \"\" 表示这一轮不要上下文；传新 id 开一段互不干扰的对话。",
    )


class AgentResponse(BaseModel):
    """单条轨迹的结果摘要。完整轨迹用 GET /runs/{trajectory_id} 取 —— 每步都带观测
    与计量，直接塞进这里会让响应变成一堵墙。
    """

    task_id: Optional[str] = None
    result: str
    success: bool
    verifiable: bool
    #: 用 GET /runs/{trajectory_id} 取回这条轨迹的每一步
    trajectory_id: str = ""
    #: 轨迹落盘位置。审计时用它回溯这条轨迹的每一步、每个工具调用与判定依据
    trace_path: str = ""


class TrainingRequest(BaseModel):
    task_type: str = "codegen"
    num_rollouts: int = 24
    epochs: int = 1
    group_size: Optional[int] = None


class Escalation(BaseModel):
    """需要人工介入的事件。能自动判定的（验证、回滚、停机）都不走这里。"""

    code: Literal[
        "safety_violation",
        "regression",
        "no_learning_signal",
        "llm_failure",
        "budget_exceeded",
        "unexpected_error",
    ]
    detail: str
    at: str = ""


class TrainingStatus(BaseModel):
    running: bool = False
    task_type: str = "codegen"
    epoch: int = 0
    total_epochs: int = 0
    rollout: int = 0
    total_rollouts: int = 0
    avg_reward: float = 0.0
    success_rate: float = 0.0
    asset_version: int = 0
    rules_count: int = 0
    updates_applied: int = 0
    flat_groups: int = 0               # 组内无方差（无学习信号）的 group 数
    message: str = ""
    error: Optional[str] = None

    # ---- 自治边界 ----
    run_id: str = ""
    needs_human: bool = False
    escalations: List[Escalation] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    judge_tokens: int = 0               # judge 调用用量（prompt 每次不同，无缓存红利）
    distill_tokens: int = 0             # 规则蒸馏调用用量
    llm_calls: int = 0


class EvalResult(BaseModel):
    """θ 的外部验证结果。判断"训练是否真的有效"的唯一客观依据。"""

    asset_version: int
    rules_count: int
    num_tasks: int
    success_rate: float
    avg_reward: float
    avg_verify_score: float


class TrainResult(BaseModel):
    """一次训练运行的完整产出，含训练前后对照。"""

    task_type: str
    epochs: int
    total_rollouts: int
    updates_applied: int
    baseline: EvalResult
    final: EvalResult
    delta_success_rate: float
    delta_avg_reward: float
    updates_log: List[str] = Field(default_factory=list)

    # ---- 标识与落盘 ----
    run_id: str = ""
    trace_path: str = ""

    # ---- θ 质量闸门：变差就回滚 ----
    rolled_back: bool = False
    rollback_reason: str = ""
    #: 回滚前的（变差了的）eval 结果，保留下来供人工复盘
    regressed_final: Optional[EvalResult] = None

    # ---- 成本与自治边界 ----
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    judge_tokens: int = 0               # judge 调用用量（prompt 每次不同，无缓存红利）
    distill_tokens: int = 0             # 规则蒸馏调用用量
    llm_calls: int = 0
    needs_human: bool = False
    escalations: List[Escalation] = Field(default_factory=list)

    @computed_field
    @property
    def total_tokens(self) -> int:
        """本轮训练的全部 token 消耗（策略 + judge + 蒸馏）。"""
        return (
            self.prompt_tokens
            + self.completion_tokens
            + self.judge_tokens
            + self.distill_tokens
        )

    @computed_field
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hit_tokens + self.cache_miss_tokens
        return round(self.cache_hit_tokens / total, 4) if total else 0.0
