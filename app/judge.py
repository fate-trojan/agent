"""Self-Judge：双层奖励。

    第一层 verifier  —— 隐藏测试通过率，客观，权重高。这是 RLVR 里的那个 "V"。
    第二层 LLM judge —— 过程质量（是否有效修复、是否处理边界），主观，权重低。

必须说清的风险：这一层没有任何人类确认。judge 与 policy 来自同一厂商的模型，
存在 self-preference 与 reward hacking 的风险 —— 策略完全可能学会写「看起来很
规范」的注释和结构来抬高 judge 分，而不是真的通过测试。缓解手段：

  1. VERIFY_WEIGHT 默认 0.7，且 verifier 是客观通过率，压住主观分的话语权。
  2. success 只由 verify_score == 1.0 决定，judge 分不能把失败洗成成功。
  3. judge 走独立的 DEEPSEEK_JUDGE_MODEL 配置，便于换成不同厂商的模型。
  4. 训练是否有效，只认 eval 集的训练前后对照，不认训练 reward 曲线。
"""

import json
import re
from typing import Tuple

from openai import AsyncOpenAI

from app.config import settings
from app.models import TaskSpec, Trajectory

_JUDGE_PROMPT = """你是一个严格的代码评审。请评估这次「解题过程」的质量，不要只盯最终对错。

【题目】
{prompt}

【最终提交的代码】
```python
{code}
```

【过程统计】尝试了 {steps} 步；是否在未提交最终答案时回退到最优尝试：{fallback}
【隐藏测试】{passed}/{total} 通过
【执行反馈摘要】
{history}

请就以下三个维度各占三分之一打分：
1. 是否针对失败反馈做了有效修正，而不是反复提交同一版代码；
2. 代码是否主动处理了边界情况（空输入、重复、负数、单元素等）；
3. 代码是否简洁、无冗余、无无关输出。

只输出 JSON，不要输出任何其他文字：
{{"score": 0.0, "reason": "一句话理由"}}
score 取值范围 0.0 ~ 1.0。"""


class Judge:
    def __init__(self) -> None:
        if not settings.llm_configured:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置，无法启用 Judge。")
        self.client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )

    async def _llm_score(
        self, traj: Trajectory, task: TaskSpec
    ) -> Tuple[float, str, bool, int]:
        total = len(task.tests)
        passed = int(round(traj.verify_score * total)) if total else 0
        history = "\n".join(
            f"- 第{i + 1}步: {o.content[:200]}" for i, o in enumerate(traj.observations)
        ) or "（模型未执行任何尝试，直接提交）"

        prompt = _JUDGE_PROMPT.format(
            prompt=task.prompt,
            code=traj.final_code or "（空）",
            steps=len(traj.actions),
            fallback=traj.used_fallback_code,
            passed=passed,
            total=total,
            history=history,
        )
        try:
            resp = await self.client.chat.completions.create(
                model=settings.DEEPSEEK_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            return 0.5, f"judge 调用失败：{e}", True, 0

        usage = getattr(resp, "usage", None)
        tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0

        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return 0.5, f"judge 输出非 JSON：{text[:120]}", True, tokens
        try:
            obj = json.loads(m.group(0))
            score = max(0.0, min(1.0, float(obj.get("score", 0.5))))
            return score, str(obj.get("reason", ""))[:200], False, tokens
        except Exception:
            return 0.5, f"judge JSON 解析失败：{text[:120]}", True, tokens

    async def score(self, traj: Trajectory, task: TaskSpec) -> None:
        """就地填好 traj 的 judge_score / total_reward / success。"""
        # 无论如何都保证 rewards 非空，让"最后一步承载总奖励"的约定成立
        if not traj.rewards:
            traj.rewards.append(0.0)

        # 安全红线优先于一切：命中即判失败，不调 judge、不留任何分数补偿空间。
        # 必须是"判定失败"而不是"扣分项" —— 否则一次越界会被 0.3 权重的过程分洗白。
        if settings.SAFETY_ENFORCE and not traj.safety_passed:
            rules = "；".join(
                f"{v.rule}@L{v.line}" for v in traj.safety_violations[:3]
            )
            traj.judge_score = 0.0
            traj.judge_rationale = (
                f"[安全红线] 命中 {rules}，直接判失败（未调用 judge，reward 归零）"
            )
            traj.total_reward = 0.0
            traj.success = False
            traj.rewards[-1] = 0.0
            return

        score, reason, failed, tokens = await self._llm_score(traj, task)
        traj.judge_tokens += tokens
        traj.judge_score, traj.judge_rationale = score, reason

        if not task.tests:
            # 无隐藏测试 -> 奖励不可验证，只能用 judge 分，权重无需混合
            traj.total_reward = score
        else:
            traj.total_reward = (
                settings.VERIFY_WEIGHT * traj.verify_score
                + settings.JUDGE_WEIGHT * score
            )

        # success 只认可验证信号，judge 分再高也不能把失败洗成成功
        traj.success = bool(task.tests) and traj.verify_score >= 1.0
        traj.rewards[-1] = round(traj.total_reward, 6)
        if failed:
            traj.judge_rationale = "[judge 降级] " + traj.judge_rationale
