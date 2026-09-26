"""πθ 与 rollout 控制面。

    Agent._chat()  —— 一次 LLM 采样，返回文本 + 本次调用的计量（耗时/token）
    parse_action() —— 把模型输出解析成动作
    rollout()      —— 驱动 CodeEnv 循环：act → dispatch → observe → act ...

本文件不产生任何梯度。θ 是 app/asset.py 里的策略资产（prompt + 规则库），
LLM 只在给定 θ 的条件下采样动作。因此这里没有 learning_rate，也没有 backward。

环境状态、参数校验、安全闸门、执行全部在 app/env.py；本文件只负责"感知—决策"
这一半，以及把每一步的计量写进轨迹。
"""

import asyncio
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.asset import CHAT_SUFFIX
from app.config import settings
from app.env import CodeEnv, elapsed_ms, now_iso, run_tests
from app.llm import client
from app.models import Action, PolicyAsset, State, TaskSpec, Trajectory

_FINAL_RE = re.compile(r"<final>(.*?)</final>", re.S | re.I)
_ATTEMPT_RE = re.compile(r"<attempt>(.*?)</attempt>", re.S | re.I)
_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n?(.*?)```", re.S)


# ===================== 动作解析 =====================


def _extract_code(body: str) -> Optional[str]:
    """从协议块里取出代码。解析顺序：围栏代码块 → 裸 def solution。"""
    m = _FENCE_RE.search(body)
    if m and m.group(1).strip():
        return m.group(1).strip()
    if re.search(r"^\s*def\s+\w+\s*\(", body, re.M):
        return body.strip()
    return None


def parse_action(text: str) -> Action:
    """把 LLM 原始输出解析成动作。模型不遵守格式时不抛异常，而是降级为
    think（parse_failed=True），让 rollout 用一次步数换一次重新对齐的机会。"""
    text = text or ""

    m = _FINAL_RE.search(text)
    if m:
        body = m.group(1)
        code = _extract_code(body)
        if code:
            prose = _FINAL_RE.sub("", text).strip()
            return Action(type="answer", content=prose, tool_args={"code": code})
        return Action(type="think", content=text.strip(), parse_failed=True)

    m = _ATTEMPT_RE.search(text)
    if m:
        body = m.group(1)
        code = _extract_code(body)
        if code:
            prose = _ATTEMPT_RE.sub("", text).strip()
            return Action(
                type="tool_call",
                content=prose,
                tool_name="execute_code",
                tool_args={"code": code},
            )

    # 兜底：模型忘了写标签但给了代码块，按 attempt 处理
    code = _extract_code(text)
    if code:
        return Action(
            type="tool_call",
            content="",
            tool_name="execute_code",
            tool_args={"code": code},
            parse_failed=True,
        )

    return Action(type="think", content=text.strip(), parse_failed=True)


# ===================== 输入路由 =====================


#: 命中任一关键词走 codegen 闭环，否则走通用对话（README 有一张同样的表）。
#: ponytail: 关键词是脆的（「写首歌」会被判成闲聊）。要准就得让 LLM 先分类 ——
#: 多一次调用 + 一个失败点；或者干脆拆成 /agent/code 与 /agent/chat 由调用方指定。
CODEGEN_KEYWORDS: Tuple[str, ...] = (
    "代码", "脚本", "函数", "程序", "算法", "实现", "编写", "写一个", "写个",
    "调试", "报错", "修复", "重构", "优化", "单元测试", "正则", "爬虫", "接口",
    "数据库", "排序", "递归", "画", "打印", "计算", "求解", "解题",
    "code", "script", "function", "python", "bug", "debug", "refactor",
    "algorithm", "sql", "regex", "api", "def ", "class ",
)


def detect_mode(task: str) -> str:
    """按关键词决定这一轮走 codegen 闭环还是通用对话。"""
    text = (task or "").lower()
    return "codegen" if any(k in text for k in CODEGEN_KEYWORDS) else "chat"


def _meter(traj: Trajectory, meta: Dict[str, int]) -> None:
    """把一次采样的计量累加进轨迹。两条路径（闭环 / 对话）共用同一套记账。"""
    traj.llm_calls += 1
    traj.prompt_tokens += meta.get("prompt_tokens", 0)
    traj.completion_tokens += meta.get("completion_tokens", 0)
    traj.cache_hit_tokens += meta.get("cache_hit_tokens", 0)
    traj.cache_miss_tokens += meta.get("cache_miss_tokens", 0)


# ===================== 策略 πθ =====================


def _usage_of(resp: Any) -> Dict[str, int]:
    """从响应里取 token 用量。取不到就当 0（不影响主流程，但审计里要能看出缺失）。

    DeepSeek 的缓存命中/未命中是它自己的扩展字段，OpenAI 原生 SDK 的 Usage 模型里
    没有，靠 pydantic 的 extra 透传。透传到 model_extra 还是能直接取属性随 SDK 版本
    而定，两条路都走一遍 —— 取不到就静默返回 0，那会把"没统计到"伪装成"命中率 0%"。
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return dict.fromkeys(
            ("prompt_tokens", "completion_tokens", "cache_hit_tokens", "cache_miss_tokens"), 0
        )
    extra = getattr(usage, "model_extra", None) or {}

    def num(name: str) -> int:
        v = getattr(usage, name, None)
        if v is None:
            v = extra.get(name)
        return int(v or 0)

    return {
        "prompt_tokens": num("prompt_tokens"),
        "completion_tokens": num("completion_tokens"),
        "cache_hit_tokens": num("prompt_cache_hit_tokens"),
        "cache_miss_tokens": num("prompt_cache_miss_tokens"),
    }


class Agent:
    """πθ(a_t | s_t)：给定 θ（渲染为 system prompt）与观测历史，采样下一步动作。"""

    def __init__(self, asset: PolicyAsset):
        self.asset = asset
        self.client = client()

    async def _chat(
        self, messages: List[Dict[str, str]], temperature: float, max_tokens: int = 2048
    ) -> Tuple[str, Dict[str, int]]:
        """一次采样。返回 (文本, 计量)。计量跟着动作走，才能算清每个决策的成本。"""
        last: Optional[Exception] = None
        t0 = time.perf_counter()
        for _ in range(2):
            try:
                resp = await self.client.chat.completions.create(
                    model=settings.DEEPSEEK_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                meta = _usage_of(resp)
                meta["latency_ms"] = elapsed_ms(t0)
                return resp.choices[0].message.content or "", meta
            except Exception as e:  # 网络/限流类错误重试一次即可
                last = e
                await asyncio.sleep(1.0)
        raise RuntimeError(f"LLM 调用失败：{last}")

    async def rollout(
        self,
        task: TaskSpec,
        max_steps: Optional[int] = None,
        temperature: Optional[float] = None,
        run_id: str = "",
        prior: Optional[List[Dict[str, str]]] = None,
        session_id: str = "",
    ) -> Trajectory:
        """跑完一条完整轨迹：act → dispatch → observe → act ...

        循环与终止条件、状态变更全部由 CodeEnv 决定，这里只负责决策与记账。

        prior 是上一轮会话的对话前缀（提问 + 产出）。它让「显示一下刚才画的结果」
        这类指代有东西可指 —— 没有它，模型只能自己另编一个"刚才"。
        """
        max_steps = max_steps or settings.MAX_STEPS
        temperature = settings.ACT_TEMPERATURE if temperature is None else temperature

        t0 = time.perf_counter()
        env = CodeEnv(task, max_steps)
        traj = Trajectory(
            task_id=task.id,
            task_prompt=task.prompt,
            mode="codegen",
            asset_version=self.asset.version,
            run_id=run_id,
            session_id=session_id,
            started_at=now_iso(),
        )
        history: List[Dict[str, str]] = list(prior or [])
        history.append({"role": "user", "content": task.prompt})

        while not env.done:
            traj.states.append(State(env_state=env.snapshot(), step=env.step))

            text, meta = await self._chat(
                [{"role": "system", "content": self.asset.render()}] + history,
                temperature,
            )
            _meter(traj, meta)

            action = parse_action(text)
            traj.actions.append(action)
            history.append({"role": "assistant", "content": action.content or ""})

            obs, rec = env.dispatch(action, meta)
            traj.steps.append(rec)
            traj.rewards.append(0.0)
            if obs is None:  # 提交动作：没有环境反馈，循环结束
                break
            traj.observations.append(obs)
            history.append({"role": "user", "content": f"[执行反馈]\n{obs.content}"})

        # ---- 终态 ----
        if env.terminated_by == "submit":
            traj.final_code = env.best_code
        else:
            traj.final_code = env.resolve_final()
            traj.used_fallback_code = True

        traj.critical_step = env.best_step
        if 0 <= env.best_step < len(traj.steps):
            traj.steps[env.best_step].credited = True

        traj.safety_passed = env.safety_passed
        traj.safety_violations = list(env.violations)
        traj.final_env_state = env.final_state()

        if traj.final_code and env.has_verifier:
            # 自由任务没有隐藏测试，verify_score 保持 0，奖励全部来自 judge
            score, detail, _ = run_tests(
                traj.final_code, task.tests, task.entry_point
            )
            traj.verify_score = score
            traj.verify_detail = detail

        traj.finished_at = now_iso()
        traj.duration_ms = elapsed_ms(t0)
        return traj

    async def reply(
        self,
        task: TaskSpec,
        temperature: Optional[float] = None,
        prior: Optional[List[Dict[str, str]]] = None,
        session_id: str = "",
    ) -> Trajectory:
        """通用对话：一次调用直接回答，不进 CodeEnv 闭环。

        没有执行、没有隐藏测试，也就没有可验证奖励与 judge —— 产出就是那段文本。
        """
        temperature = settings.ACT_TEMPERATURE if temperature is None else temperature

        t0 = time.perf_counter()
        traj = Trajectory(
            task_id=task.id,
            task_prompt=task.prompt,
            mode="chat",
            asset_version=self.asset.version,
            session_id=session_id,
            started_at=now_iso(),
        )
        history: List[Dict[str, str]] = list(prior or [])
        history.append({"role": "user", "content": task.prompt})

        # CHAT_SUFFIX 追加在 base_prompt 之后，把其中的代码输出协议压掉
        text, meta = await self._chat(
            [{"role": "system", "content": self.asset.render() + CHAT_SUFFIX}] + history,
            temperature,
        )
        _meter(traj, meta)
        traj.reply = text.strip()

        traj.finished_at = now_iso()
        traj.duration_ms = elapsed_ms(t0)
        return traj
