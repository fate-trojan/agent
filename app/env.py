"""可验证代码环境。

显式形式化 E = (S, A, T, O, G, terminate)：

    S  状态空间  —— attempts（每次提交的代码与实测成绩）、best、denied 次数、step
    A  动作空间  —— exec(code) 试执行 / submit(code) 提交最终答案
    T  转移函数  —— dispatch() 是唯一的转移入口，同时承担参数校验与安全闸门
    O  观测空间  —— 有隐藏用例时给「通过率 + 逐用例期望/实际」；
                     自由任务给真实 stdout/stderr
    G  目标      —— 隐藏测试全部通过；自由任务无客观目标，由 judge 代理
    terminate    —— 模型调用 submit，或达到 max_steps

隔离边界：执行手段只有「独立进程 + 超时 + 干净环境变量 + -I 隔离模式」，
没有网络 / 文件系统 / 内核级隔离。安全闸门是禁止操作清单，不是沙箱。
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.models import (
    Action,
    Observation,
    SafetyViolation,
    StepRecord,
    TaskSpec,
)
from app.safety import scan, scrubbed_env

#: 该环境只暴露一个工具。模型给出别的 tool_name 一律拒绝
ALLOWED_TOOLS = ("execute_code",)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


# ===================== 底层执行器 =====================

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

    四层边界：
      1. 独立进程 —— 被测代码崩溃不会带走服务
      2. 超时     —— 死循环被切断
      3. 干净环境变量 —— 剥离 os.environ，被测代码读不到 DEEPSEEK_API_KEY
      4. -I 隔离模式 —— 忽略 PYTHON* 环境变量与用户 site-packages
    """
    tmpdir = tempfile.mkdtemp(prefix="agentrl_")
    try:
        for name, content in files.items():
            with open(os.path.join(tmpdir, name), "w", encoding="utf-8") as f:
                f.write(content)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-u", entry],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=settings.EXEC_TIMEOUT,
                env=scrubbed_env(dict(os.environ)),
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

    通过率而非 0/1 二值，是为了给 GRPO 提供有方差的奖励信号。
    没有隐藏用例的自由任务不该走这里（拿不到有效判定），要用 run_code()。
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

    返回 (是否正常退出, 输出文本)；没有输出就返回空串，由调用方决定怎么措辞
    （不要在这里编一句"运行成功"把"什么都没发生"说成成功）。
    """
    if not code.strip():
        return False, "代码为空"
    rc, stdout, stderr = _run_python({"main.py": code}, "main.py")
    if rc is None:
        return False, stderr
    return rc == 0, (stdout + stderr).strip()[-800:]


# ===================== 环境 =====================


class CodeEnv:
    """代码生成任务的环境。一次 rollout 一个实例。"""

    def __init__(self, task: TaskSpec, max_steps: int) -> None:
        self.task = task
        self.max_steps = max_steps
        self.has_verifier = bool(task.tests)
        self.reset()

    # ---------- 状态 ----------

    def reset(self) -> None:
        """回到初始状态 s₀。"""
        self.step = 0
        self.attempts: List[Dict[str, Any]] = []
        self.best_score = -1.0
        self.best_code = ""
        self.best_step = -1
        self.denied = 0
        self.violations: List[SafetyViolation] = []
        self.terminated_by = ""

    def snapshot(self) -> Dict[str, Any]:
        """当前环境状态的只读快照。"""
        return {
            "task_id": self.task.id,
            "step": self.step,
            "max_steps": self.max_steps,
            "has_verifier": self.has_verifier,
            "attempts": list(self.attempts),
            "best": {"score": self.best_score, "step": self.best_step},
            "denied_count": self.denied,
            "safety_violations": len(self.violations),
            "terminated_by": self.terminated_by,
        }

    @property
    def done(self) -> bool:
        return bool(self.terminated_by) or self.step >= self.max_steps

    @property
    def safety_passed(self) -> bool:
        """一旦命中红线就永久置否：越界是运行级失败，不因后续自愈而洗白。"""
        return not self.violations

    # ---------- 转移 ----------

    def dispatch(
        self, action: Action, llm: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[Observation], StepRecord]:
        """唯一的动作入口，返回 (观测, 步记录)。

        提交类动作返回 (None, 记录) —— 提交后没有环境反馈，调用方据此终止循环。
        """
        llm = llm or {}
        index = self.step
        rec = StepRecord(
            step=index,
            action_type=action.type,
            parse_failed=action.parse_failed,
            tool_name=action.tool_name,
            started_at=now_iso(),
            llm_latency_ms=int(llm.get("latency_ms", 0)),
            prompt_tokens=int(llm.get("prompt_tokens", 0)),
            completion_tokens=int(llm.get("completion_tokens", 0)),
            cache_hit_tokens=int(llm.get("cache_hit_tokens", 0)),
            cache_miss_tokens=int(llm.get("cache_miss_tokens", 0)),
        )

        code = (action.tool_args or {}).get("code")

        # ---- 提交最终答案：终态转移 ----
        if action.type == "answer" and isinstance(code, str) and code.strip():
            deny = self._validate(action, code)
            if deny:
                # 被拒的提交同样消耗一步：否则模型反复提交越界代码时 step 永不前进，
                # rollout 会无限循环烧 token（done 判定含 step >= max_steps，故能终止）。
                self.step += 1
                return self._finish(rec, deny), rec
            self.best_code, self.best_step = code, index
            rec.credited = True
            self.terminated_by = "submit"
            return self._finish(rec, None), rec

        # ---- 试执行：执行并写入 state ----
        if action.type == "tool_call" and isinstance(code, str):
            deny = self._validate(action, code)
            if deny:
                self.step += 1
                return self._finish(rec, deny), rec
            obs = self._execute(code, rec)
            self.step += 1
            rec.finished_at = now_iso()
            if self.step >= self.max_steps:
                self.terminated_by = "max_steps"
            return obs, rec

        # ---- 协议外动作（think / 解析失败）：不是错误，只是浪费一步 ----
        self.step += 1
        if self.step >= self.max_steps:
            self.terminated_by = "max_steps"
        rec.error = "未检测到合法的 <attempt> 或 <final> 代码块"
        rec.finished_at = now_iso()
        return (
            Observation(
                content="未检测到合法的 <attempt> 或 <final> 代码块，请严格按协议输出。",
                success=False,
            ),
            rec,
        )

    # ---------- 校验与安全闸门 ----------

    def _validate(self, action: Action, code: str) -> Optional[str]:
        """工具参数校验 + 安全红线。返回拒绝原因，None 表示放行。"""
        if action.type == "tool_call":
            if action.tool_name not in ALLOWED_TOOLS:
                return f"未知工具 {action.tool_name!r}，本环境只开放 {ALLOWED_TOOLS[0]}"
            if not isinstance(action.tool_args, dict):
                return "tool_args 必须是对象"
            if set(action.tool_args) != {"code"}:
                return f"tool_args 只接受 code 字段，收到 {sorted(action.tool_args)}"
        if not code.strip():
            return "code 为空"
        if len(code) > settings.MAX_CODE_CHARS:
            return f"code 超过 {settings.MAX_CODE_CHARS} 字符上限（实际 {len(code)}）"
        if settings.SAFETY_ENFORCE:
            hits = scan(code)
            if hits:
                self.violations.extend(hits)
                detail = "；".join(f"{h.rule}@L{h.line}" for h in hits[:3])
                return f"违反安全红线（{detail}）"
        return None

    def _finish(
        self, rec: StepRecord, deny: Optional[str]
    ) -> Optional[Observation]:
        """收尾。deny 非空 = 这一步被拒，返回拒绝观测供模型重试；否则返回 None。"""
        rec.finished_at = rec.finished_at or now_iso()
        if not deny:
            return None
        rec.denied = True
        rec.deny_reason = deny
        self.denied += 1
        return Observation(
            content=f"已拒绝执行：{deny}。请移除越界操作后重新提交。",
            success=False,
        )

    # ---------- 执行 ----------

    def _execute(self, code: str, rec: StepRecord) -> Observation:
        before_best = self.best_score
        t0 = time.perf_counter()

        if self.has_verifier:
            score, detail, err = run_tests(code, self.task.tests, self.task.entry_point)
            rec.exec_latency_ms = elapsed_ms(t0)
            total = len(self.task.tests)
            passed = int(round(score * total))
            rec.verify_score, rec.passed, rec.total = score, passed, total
            # 步级信用：这一步把成绩往前推了多少（只有正增量才算贡献）
            rec.verify_delta = round(max(0.0, score - before_best), 4)
            if score > self.best_score:
                self.best_score, self.best_code, self.best_step = score, code, rec.step
            self.attempts.append(
                {
                    "step": rec.step,
                    "score": round(score, 4),
                    "passed": passed,
                    "total": total,
                    "error": err[:200],
                }
            )
            ok = score >= 1.0
            text = f"执行结果：{passed}/{total} 个用例通过。"
            if err:
                text += f"\n执行错误：{err[:400]}"
            fails = [d for d in detail if not d.get("ok")]
            if fails:
                text += "\n失败用例：" + json.dumps(fails[:3], ensure_ascii=False)
            if ok:
                text += "\n全部通过。请输出 <final> 提交。"
            return Observation(content=text, success=ok)

        # 自由任务：没有隐藏用例可判定，真实运行结果就是唯一反馈
        ok, out = run_code(code)
        rec.exec_latency_ms = elapsed_ms(t0)
        if ok:
            # 保留最后一次跑通的代码作为兜底提交
            self.best_score, self.best_code, self.best_step = 0.0, code, rec.step
        self.attempts.append({"step": rec.step, "ok": ok, "output": out[:200]})
        if not ok:
            text = "执行失败，输出：\n" + (out or "（进程没有任何输出就退出了）")
        elif out:
            text = "执行成功，输出：\n" + out
        else:
            # rc==0 但零输出：最常见的原因是只定义了函数却没人调用它，
            # 报成"执行成功"会让模型以为这版代码没问题，原地反复重试。
            text = (
                "执行成功，但没有任何输出 —— 通常意味着你只定义了函数却没有调用它。"
                "本任务没有隐藏测试来调用你的入口函数，请让脚本自己产生输出"
                "（调用入口函数或直接 print 结果）。"
            )
        return Observation(
            content=(
                text
                + "\n（本任务没有隐藏测试，我无法判定对错，请自行核对后输出 <final> 提交。）"
            ),
            success=ok,
        )

    # ---------- 终态 ----------

    def resolve_final(self) -> str:
        """模型始终没提交 <final> 时，回退到环境记录的最好一次尝试。

        否则"不会用协议"和"写错了代码"会被混成同一种失败，污染优势估计。
        """
        return self.best_code

    def final_state(self) -> Dict[str, Any]:
        snap = self.snapshot()
        snap["terminated_by"] = self.terminated_by or "max_steps"
        return snap
