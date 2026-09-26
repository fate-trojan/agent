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

from pydantic import BaseModel, Field


# ===================== 六元组 =====================


class State(BaseModel):
    """状态空间 S：环境状态 + 观测历史。"""

    env_state: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, str]] = Field(default_factory=list)
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
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    """完整轨迹 τ = (s₀, a₀, o₀, r₀, ..., s_T)。

    奖励约定：稀疏终点奖励 —— 除最后一步外 rewards 全为 0，
    最后一步承载 Judge 给出的总奖励。
    """

    task_id: str
    task_prompt: str = ""
    states: List[State] = Field(default_factory=list)
    actions: List[Action] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)
    rewards: List[float] = Field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False

    # ---- 训练拆解信息 ----
    final_code: str = ""
    verify_score: float = 0.0                      # 可验证奖励 = 隐藏测试通过率
    verify_detail: List[Dict[str, Any]] = Field(default_factory=list)
    judge_score: float = 0.0                       # LLM judge 过程分
    judge_rationale: str = ""
    advantage: float = 0.0                         # 组内相对优势 A_i
    asset_version: int = 0                         # 采样时使用的 θ 版本
    used_fallback_code: bool = False               # 未输出 <final>，回退到最优 attempt


# ===================== 策略资产 θ =====================


class Rule(BaseModel):
    """θ 的一个可训练单元。文本即参数。"""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    text: str
    kind: Literal["success", "antipattern"] = "success"
    origin_task: str = ""
    gain: float = 0.0          # 写回 θ 时的组内优势估计，用于剪枝
    asset_version: int = 0


class PolicyAsset(BaseModel):
    """策略资产 θ = base_prompt + rules。

    πθ 的输出分布完全由 render() 出来的 system prompt 决定；
    对 θ 的"训练"就是改写这个对象再落盘。
    """

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


class AgentResponse(BaseModel):
    task_id: Optional[str] = None
    result: str
    trajectory: Trajectory
    success: bool
    verifiable: bool


class TrainingRequest(BaseModel):
    task_type: str = "codegen"
    num_rollouts: int = 24
    epochs: int = 1
    group_size: Optional[int] = None


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
    message: str = ""
    error: Optional[str] = None


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
