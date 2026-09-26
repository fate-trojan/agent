"""GRPO 的无梯度实现 + RLVR 训练循环（MVP 合并版）。

GRPO 在本方案里贡献的是「信用分配」而不是「梯度」：

    1. 同一道题采样 G 条轨迹，构成一个 group
    2. 组内相对优势  A_i = (r_i - mean(r)) / (std(r) + eps)      ← 原论文公式
    3. 用 A_i 的正负挑出「正样本 / 负样本」，让 LLM 对照蒸馏出可复用规则
    4. 规则写回 θ（app/asset.py）

第 2 步可以精确复现原论文。第 3 步不是梯度下降 —— 它是一次启发式搜索，
数学上不保证策略单调提升，也不保证 θ 的改进方向正确。所以：

  - 训练是否有效，只认 eval 集的训练前后对照（TrainResult.baseline vs final）
  - 组内奖励若没有方差，优势恒为 0，训练什么都不会发生（这是正确行为，不是 bug）
"""

import asyncio
import json
import re
from typing import List, Optional, Tuple

import numpy as np
from openai import AsyncOpenAI

from app.agent import Agent
from app.asset import AssetStore
from app.config import settings
from app.judge import Judge
from app.models import (
    EvalResult,
    Rule,
    TaskSpec,
    TrainingStatus,
    TrainResult,
    Trajectory,
)
from app.tasks import eval_tasks, train_tasks


def group_advantages(rewards: List[float], eps: Optional[float] = None) -> List[float]:
    """GRPO 的组内相对优势：A_i = (r_i - mean(r)) / (std(r) + eps)。

    用样本标准差（numpy 默认 ddof=0）。方差为 0 时返回全 0 ——
    这正是 GRPO 的行为：组内没有差异就没有学习信号，不该人为造梯度。
    """
    eps = settings.ADVANTAGE_EPS if eps is None else eps
    arr = np.asarray(rewards, dtype=np.float64)
    if arr.size == 0:
        return []
    if float(arr.std()) < 1e-9:
        return [0.0] * int(arr.size)
    return ((arr - arr.mean()) / (arr.std() + eps)).tolist()


_DISTILL_PROMPT = """这是一次对照实验的结果。请从中总结「可复用的经验」，用于改进未来所有同类任务的解法。

【题目】
{prompt}
{blocks}
要求：
1. 最多输出 3 条经验，每条一句话、20~80 字，必须能直接指导写代码。
2. 每条都必须是跨题目可复用的方法论或陷阱提醒。
3. 严禁写成针对本题具体输入的答案（例如「输入 [1,2] 时返回 3」），那是过拟合单个用例。
4. kind 取 "success"（从成功中总结）或 "antipattern"（从失败中总结的陷阱）。

只输出 JSON 数组，不要任何其他文字：
[{{"text": "...", "kind": "success"}}]"""


def _feedback_of(traj: Trajectory) -> str:
    if not traj.observations:
        return "（模型未执行任何尝试）"
    return " / ".join(o.content[:120].replace("\n", " ") for o in traj.observations[-2:])


def _block(title: str, traj: Trajectory, total: int) -> str:
    return (
        f"\n【{title}】组内优势 {traj.advantage:+.2f}，"
        f"隐藏测试 {int(round(traj.verify_score * total))}/{total} 通过\n"
        f"```python\n{traj.final_code or '（空）'}\n```\n"
        f"执行反馈：{_feedback_of(traj)}\n"
    )


def _parse_drafts(text: str) -> List[Rule]:
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        raw = json.loads(m.group(0))
    except Exception:
        return []
    out: List[Rule] = []
    for item in raw[:3]:
        if not isinstance(item, dict):
            continue
        t = str(item.get("text", "")).strip()
        if not t or len(t) > 240:
            continue
        kind = item.get("kind", "success")
        out.append(Rule(text=t, kind=kind if kind in ("success", "antipattern") else "success"))
    return out


class GRPOTrainer:
    """一轮训练 = 若干 group × (采样 → 打分 → 优势 → 更新 θ) 后做外部验证。"""

    def __init__(self, task_type: str = "codegen") -> None:
        self.task_type = task_type
        self.store = AssetStore(task_type)
        self.judge = Judge()
        if not settings.llm_configured:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置。")
        # 蒸馏用独立 client：它的角色是「策略改进」，与 Judge 的「打分」不是一回事，
        # 未来换成更强的模型做蒸馏时不该影响打分链路。
        self.client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
        self.status = TrainingStatus(task_type=task_type)
        self.result: Optional[TrainResult] = None
        self._stop = False
        self._sem = asyncio.Semaphore(max(1, settings.MAX_CONCURRENCY))

    # ---------- 对外控制 ----------

    def stop(self) -> None:
        self._stop = True

    # ---------- 内部工具 ----------

    async def _one(
        self, agent: Agent, task: TaskSpec, temperature: float
    ) -> Trajectory:
        async with self._sem:
            traj = await agent.rollout(task, temperature=temperature)
            await self.judge.score(traj, task)
            return traj

    async def _evaluate(self, agent: Agent, tasks: List[TaskSpec]) -> EvalResult:
        """贪心采样（temperature=0）跑 eval 集，作为 θ 的外部体检。"""
        trajs = await asyncio.gather(*[self._one(agent, t, 0.0) for t in tasks])
        n = max(1, len(trajs))
        return EvalResult(
            asset_version=agent.asset.version,
            rules_count=len(agent.asset.rules),
            num_tasks=len(trajs),
            success_rate=sum(t.success for t in trajs) / n,
            avg_reward=sum(t.total_reward for t in trajs) / n,
            avg_verify_score=sum(t.verify_score for t in trajs) / n,
        )

    async def _distill(self, task: TaskSpec, group: List[Trajectory]) -> Tuple[List[Rule], str]:
        """找出组内正负样本，让 LLM 对照蒸馏出规则。

        返回 (规则, 失败原因)。失败原因必须向上传：
        「优势明明够大却没写进任何规则」是使用者最需要能自己归因的情况，
        如果只把原因写在 status.message 上，会被后续的状态更新覆盖掉。
        """
        winner = max(group, key=lambda t: t.advantage)
        loser = min(group, key=lambda t: t.advantage)
        total = len(task.tests)

        blocks = _block("本次优势最高的一次尝试", winner, total)
        if loser.advantage < 0 and loser is not winner:
            blocks += _block("本次优势最低的一次尝试", loser, total)

        prompt = _DISTILL_PROMPT.format(prompt=task.prompt, blocks=blocks)
        try:
            resp = await self.client.chat.completions.create(
                model=settings.DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            return [], f"蒸馏调用失败 {type(e).__name__}: {str(e)[:100]}"

        rules = _parse_drafts(text)
        if not rules:
            return [], f"蒸馏输出解析不出规则: {text[:100]!r}"
        for r in rules:
            r.origin_task = task.id
            r.gain = round(winner.advantage, 4)
        return rules, ""

    # ---------- 主循环 ----------

    async def train(
        self,
        num_rollouts: Optional[int] = None,
        epochs: int = 1,
        group_size: Optional[int] = None,
    ) -> TrainResult:
        num_rollouts = num_rollouts or settings.MAX_ROLLOUTS
        group_size = group_size or settings.GRPO_GROUP_SIZE
        if group_size < 2:
            raise ValueError("group_size 必须 >= 2，否则组内无方差、优势恒为 0")

        tasks, evals = train_tasks(), eval_tasks()
        if not tasks or not evals:
            raise RuntimeError("任务集为空：训练集与评测集都必须存在")

        self._stop = False
        self.status = TrainingStatus(
            running=True,
            task_type=self.task_type,
            total_epochs=epochs,
            total_rollouts=num_rollouts,
            asset_version=self.store.version,
            rules_count=len(self.store.asset.rules),
            message="基线评测中…",
        )

        try:
            agent = Agent(self.store.asset)
            baseline = await self._evaluate(agent, evals)

            n_groups = max(1, num_rollouts // group_size)
            done = 0
            updates = 0
            flat_groups = 0
            all_rewards: List[float] = []
            all_success = 0
            log: List[str] = []

            for epoch in range(1, epochs + 1):
                # 每个 epoch 重建 Agent，确保它持有的 asset 对象是最新的
                agent = Agent(self.store.asset)
                for g in range(n_groups):
                    if self._stop:
                        break

                    task = tasks[g % len(tasks)]
                    group = list(
                        await asyncio.gather(
                            *[
                                self._one(agent, task, settings.GROUP_TEMPERATURE)
                                for _ in range(group_size)
                            ]
                        )
                    )

                    rewards = [t.total_reward for t in group]
                    advs = group_advantages(rewards)
                    for t, a in zip(group, advs):
                        t.advantage = a
                    # 注意无法追溯到具体步骤

                    done += group_size
                    all_rewards.extend(rewards)
                    all_success += sum(1 for t in group if t.success)
                    if float(np.std(np.asarray(rewards))) < 1e-9:
                        flat_groups += 1

                    self.status.epoch = epoch
                    self.status.rollout = done
                    self.status.avg_reward = round(sum(all_rewards) / len(all_rewards), 4)
                    self.status.success_rate = round(all_success / done, 4)
                    self.status.message = f"epoch {epoch}/{epochs} · rollout {done}/{n_groups * group_size}"
                    self.status.asset_version = self.store.version
                    self.status.rules_count = len(self.store.asset.rules)

                    # ---- 策略改进：只在存在正优势时才动 θ ----
                    best = max(advs)
                    if best <= settings.MIN_RULE_GAIN:
                        log.append(
                            f"e{epoch} r{done} {task.id} v{self.store.version} "
                            f"优势不足(max={best:+.3f})，跳过更新"
                        )
                        continue

                    rules, note = await self._distill(task, group)
                    added, pruned = self.store.add_rules(rules)
                    if added:
                        updates += 1
                        self.status.updates_applied = updates
                        self.status.asset_version = self.store.version
                        self.status.rules_count = len(self.store.asset.rules)
                    log.append(
                        f"e{epoch} r{done} {task.id} v{self.store.version} "
                        f"reward={[round(r, 3) for r in rewards]} "
                        f"adv={[round(a, 2) for a in advs]} "
                        f"+{len(added)}规则 -{len(pruned)}规则"
                        + (f" ｜{note}" if note else "")
                    )

                if self._stop:
                    break

            # ---- 外部验证：唯一能判定「训练是否真的有效」的依据 ----
            agent = Agent(self.store.asset)
            self.status.message = "训练后评测中…"
            final = await self._evaluate(agent, evals)

            result = TrainResult(
                task_type=self.task_type,
                epochs=epochs,
                total_rollouts=done,
                updates_applied=updates,
                baseline=baseline,
                final=final,
                delta_success_rate=round(final.success_rate - baseline.success_rate, 4),
                delta_avg_reward=round(final.avg_reward - baseline.avg_reward, 4),
                updates_log=log,
            )
            self.result = result

            msg = (
                f"完成：{done} 次 rollout，θ 更新 {updates} 次，"
                f"v{baseline.asset_version}→v{final.asset_version}，"
                f"eval 成功率 {baseline.success_rate:.0%}→{final.success_rate:.0%}"
            )
            if self._stop:
                msg = "已手动停止。" + msg
            if updates == 0 and flat_groups == n_groups * epochs:
                msg += (
                    f"｜注意：{flat_groups} 个 group 组内奖励无方差，"
                    "GRPO 优势恒为 0，本次没有产生任何学习信号"
                )
            self.status.message = msg
            self.status.running = False
            return result

        except Exception as e:  # 服务化场景：错误走 status 通道，不炸后台任务
            self.status.running = False
            self.status.error = f"{type(e).__name__}: {e}"
            self.status.message = "训练失败，详见 error 字段"
            raise
