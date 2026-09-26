"""Agent 策略 πθ + 可验证环境 + rollout 编排。

    act()      —— πθ 单步动作采样
    run_tests  —— 可验证环境（子进程执行 + 隐藏测试）
    rollout()  —— 控制面：循环 act → 执行 → 反馈，产出完整轨迹

本文件不产生任何梯度。θ 是 app/asset.py 里的策略资产（prompt + 规则库），
LLM 只在给定 θ 的条件下采样动作。因此这里没有 learning_rate，也没有 backward。

安全边界：run_tests 是 HumanEval 式的评测执行，隔离手段仅有「独立进程 + 超时」，
没有网络 / 文件系统 / 内存隔离。绝不可把本服务暴露到不受信任的网络上。
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from app.config import settings
from app.models import Action, Observation, PolicyAsset, State, TaskSpec, Trajectory

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
    """把 LLM 原始输出解析成动作。

    协议解析是纯文本匹配，模型不遵守格式时不会抛异常，而是降级为
    think（parse_failed=True），让 rollout 用一次步数换一次重新对齐的机会。
    """
    text = text or ""

    m = _FINAL_RE.search(text)
    if m:
        body = m.group(1)
        code = _extract_code(body)
        if code:
            prose = _FINAL_RE.sub("", text).strip()
            return Action(type="answer", content=prose, tool_args={"code": code})
        return Action(
            type="think", content=text.strip(), parse_failed=True
        )

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


# ===================== 可验证环境 =====================

_RUNNER_SRC = '''
import json, sys, traceback


def main():
    with open("cases.json", encoding="utf-8") as f:
        cases = json.load(f)
    ns = {}
    try:
        with open("solver.py", encoding="utf-8") as f:
            src = f.read()
        exec(compile(src, "solver.py", "exec"), ns)
    except Exception:
        print(json.dumps({"error": "\u7f16\u8bd1\u6216\u6267\u884c\u5931\u8d25:\\n" + traceback.format_exc(limit=2)}, ensure_ascii=False))
        return
    fn = ns.get("__ENTRY__")
    if not callable(fn):
        print(json.dumps({"error": "\u672a\u5b9a\u4e49\u53ef\u8c03\u7528\u7684\u51fd\u6570 __ENTRY__"}, ensure_ascii=False))
        return
    passed = 0
    detail = []
    for c in cases:
        try:
            got = eval(c["expr"], ns)
            ok = got == c["expected"]
        except Exception as e:
            got = type(e).__name__ + ": " + str(e)
            ok = False
        passed += int(ok)
        detail.append({
            "expr": c["expr"],
            "expected": c["expected"],
            "got": repr(got)[:160],
            "ok": bool(ok),
        })
    print(json.dumps({"passed": passed, "total": len(cases), "detail": detail}, ensure_ascii=False))


main()
'''


def _run_python(files: Dict[str, str], entry: str) -> Tuple[Optional[int], str, str]:
    """在临时目录的独立进程里执行一个 Python 脚本。

    返回 (returncode, stdout, stderr)；超时返回 (None, "", 说明)。
    隔离手段只有「独立进程 + 超时」，没有网络 / 文件系统 / 内存隔离。
    """
    tmpdir = tempfile.mkdtemp(prefix="agentrl_")
    try:
        for name, content in files.items():
            with open(os.path.join(tmpdir, name), "w", encoding="utf-8") as f:
                f.write(content)
        try:
            proc = subprocess.run(
                [sys.executable, "-u", entry],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=settings.EXEC_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return None, "", f"执行超时（>{settings.EXEC_TIMEOUT}s），疑似死循环"
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_tests(
    code: str, tests: List[Tuple[str, Any]], entry_point: str = "solution"
) -> Tuple[float, List[Dict[str, Any]], str]:
    """在子进程里跑隐藏测试，返回 (通过率, 逐用例明细, 错误信息)。

    通过率而非 0/1 二值，是为了给 GRPO 提供有方差的奖励信号 ——
    如果组内奖励全相等，优势恒为 0，训练不会发生。

    注意：没有隐藏用例的自由任务不该走这里（拿不到任何有效判定），
    要用 run_code()。
    """
    if not code.strip():
        return 0.0, [], "代码为空"
    if not tests:
        raise ValueError("run_tests 需要隐藏用例；自由任务请改用 run_code()")

    rc, stdout, stderr = _run_python(
        {
            "solver.py": code,
            "cases.json": json.dumps(
                [{"expr": e, "expected": x} for e, x in tests], ensure_ascii=False
            ),
            "runner.py": _RUNNER_SRC.replace("__ENTRY__", entry_point),
        },
        "runner.py",
    )
    if rc is None:
        return 0.0, [], stderr

    payload = None
    for line in reversed(stdout.strip().splitlines()):
        try:
            payload = json.loads(line)
            break
        except Exception:
            continue

    if payload is None:
        tail = stderr.strip() or stdout.strip()
        return 0.0, [], f"无法解析执行输出：{tail[-400:]}"
    if "error" in payload:
        return 0.0, [], str(payload["error"])[:600]

    total = payload.get("total") or len(tests)
    passed = int(payload.get("passed", 0))
    return (passed / total if total else 0.0), payload.get("detail", []), ""


def run_code(code: str) -> Tuple[bool, str]:
    """只执行、不判定对错 —— 用于没有隐藏用例的自由任务。

    这类任务环境给不出可验证奖励（reward 全来自 judge），但模型依然需要真实的
    执行反馈才能自我修复。返回 (是否正常退出, 输出文本)。
    """
    if not code.strip():
        return False, "代码为空"
    rc, stdout, stderr = _run_python({"main.py": code}, "main.py")
    if rc is None:
        return False, stderr
    out = (stdout + stderr).strip()
    return rc == 0, (out[-800:] if out else "（运行成功，无任何输出）")


# ===================== 策略 πθ =====================


class Agent:
    """πθ(a_t | s_t)：给定 θ（渲染为 system prompt）与观测历史，采样下一步动作。"""

    def __init__(self, asset: PolicyAsset):
        if not settings.llm_configured:
            raise RuntimeError(
                "DEEPSEEK_API_KEY 未配置。请 `cp .env.example .env` 后填入真实 Key。"
            )
        self.asset = asset
        self.client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )

    async def _chat(
        self, messages: List[Dict[str, str]], temperature: float, max_tokens: int = 2048
    ) -> str:
        last: Optional[Exception] = None
        for _ in range(2):
            try:
                resp = await self.client.chat.completions.create(
                    model=settings.DEEPSEEK_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # 网络/限流类错误重试一次即可
                last = e
                await asyncio.sleep(1.0)
        raise RuntimeError(f"LLM 调用失败：{last}")

    async def act(self, state: State, temperature: float) -> Action:
        messages = [{"role": "system", "content": self.asset.render()}] + list(
            state.history
        )
        return parse_action(await self._chat(messages, temperature))

    async def rollout(
        self,
        task: TaskSpec,
        max_steps: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Trajectory:
        """跑完一条完整轨迹：act → 执行 → 反馈 → 再 act ...

        失败的尝试会把「哪些用例挂了、期望什么、实际得到什么」写回 history，
        这就是模型自我修复的信息来源 —— 也是本任务"长程"的体现。
        """
        max_steps = max_steps or settings.MAX_STEPS
        temperature = settings.ACT_TEMPERATURE if temperature is None else temperature

        traj = Trajectory(
            task_id=task.id, task_prompt=task.prompt, asset_version=self.asset.version
        )
        history: List[Dict[str, str]] = [{"role": "user", "content": task.prompt}]
        # 注意未分解子任务
        best_score, best_code = -1.0, ""
        final_code: Optional[str] = None

        # 290-354 真实闭环，且记录完整 actions/observations/states
        for step in range(max_steps):
            state = State(
                env_state={
                    "task_id": task.id,
                    "step": step,
                    "best_verify": max(best_score, 0.0),
                },
                history=list(history),
                step=step,
            )
            traj.states.append(state)
            action = await self.act(state, temperature)
            traj.actions.append(action)
            history.append({"role": "assistant", "content": action.content or ""})
            # 注意 history 没有压缩，长程易爆

            code = (action.tool_args or {}).get("code")
            if action.type == "answer" and code:
                final_code = code
                traj.rewards.append(0.0)
                break

            if action.type == "tool_call" and code:
                if task.tests:
                    score, detail, err = run_tests(code, task.tests, task.entry_point)
                    if score > best_score:
                        best_score, best_code = score, code
                    ok = score >= 1.0
                    passed = int(round(score * len(task.tests)))
                    text = f"执行结果：{passed}/{len(task.tests)} 个用例通过。"
                    if err:
                        text += f"\n执行错误：{err[:400]}"
                    fails = [d for d in detail if not d.get("ok")]
                    if fails:
                        text += "\n失败用例：" + json.dumps(fails[:3], ensure_ascii=False)
                    if ok:
                        text += "\n全部通过。请输出 <final> 提交。"
                    obs = Observation(
                        content=text, success=ok, metadata={"passed": passed}
                    )
                else:
                    # 自由任务：没有隐藏用例可以判定对错，但真实执行结果仍然是
                    # 模型自我修复的唯一信息来源。只报"无代码或无用例"等于让它
                    # 盲迭代 N 步，什么都学不到。
                    ok, out = run_code(code)
                    if ok:
                        # 保留最后一次跑通的代码作为兜底提交
                        best_score, best_code = 0.0, code
                    obs = Observation(
                        content=(
                            ("执行成功，输出：\n" if ok else "执行失败，输出：\n")
                            + out
                            + "\n（本任务没有隐藏测试，我无法判定对错，请自行核对后输出 <final> 提交。）"
                        ),
                        success=ok,
                    )
            else:
                obs = Observation(
                    content="未检测到合法的 <attempt> 或 <final> 代码块，请严格按协议输出。",
                    success=False,
                )

            traj.observations.append(obs)
            traj.rewards.append(0.0)
            history.append({"role": "user", "content": f"[执行反馈]\n{obs.content}"})

        # 注意没有失败恢复策略或人工上报
        if final_code is None:
            # 模型一直没提交 <final>：回退到分数最高的那次尝试，而不是直接判 0。
            # 否则"不会用协议"和"写错了代码"会被混成同一种失败，污染优势估计。
            final_code = best_code
            traj.used_fallback_code = True

        traj.final_code = final_code or ""
        if traj.final_code and task.tests:
            # 自由任务没有隐藏测试，verify_score 保持 0，奖励全部来自 judge
            score, detail, _ = run_tests(traj.final_code, task.tests, task.entry_point)
            traj.verify_score = score
            traj.verify_detail = detail
        return traj
