"""轨迹与训练运行的落盘（可审计性）。

为什么用 JSONL 而不是数据库：#4 要求的是「可回溯」，一行一条 JSON 可以被
grep / diff / jq 直接消费，不引入依赖，也不会因为写入失败而中断训练。
所有写操作失败都吞掉并返回空路径 —— 审计是旁路，不能反噬主流程。

三类记录：
    traj-YYYYMMDD.jsonl   一条完整轨迹（含每步动作、观测、耗时、token、判定）
    run-YYYYMMDD.jsonl    一次训练运行（含前后对照、回滚、升级人工事件）
    rule-YYYYMMDD.jsonl   一次规则写入（含它来自哪条轨迹的哪一步）
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.models import Rule, TaskSpec, TrainResult, Trajectory


def tests_fingerprint(task: TaskSpec) -> str:
    """隐藏用例的指纹。

    轨迹落盘时不存用例正文（正文在 tasks.py 里由 git 版本化），只存指纹。
    这样"这次判定用的是哪一版用例"可以被验证，而不会让轨迹文件膨胀。
    """
    payload = json.dumps(
        {"entry": task.entry_point, "tests": task.tests}, ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class TraceStore:
    """append-only 审计日志。"""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = Path(root or settings.TRACE_DIR)
        self.enabled = settings.TRACE_ENABLED

    # ---------- 写 ----------

    def _append(self, kind: str, payload: Dict[str, Any]) -> str:
        if not self.enabled:
            return ""
        try:
            day = datetime.now().strftime("%Y%m%d")
            path = self.root / f"{kind}-{day}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            return str(path)
        except Exception:
            return ""

    def write_trajectory(self, traj: Trajectory, task: TaskSpec) -> str:
        return self._append(
            "traj",
            {
                "kind": "trajectory",
                "at": traj.finished_at or datetime.now().isoformat(timespec="milliseconds"),
                "trajectory_id": traj.trajectory_id,
                "run_id": traj.run_id,
                "task": {
                    "id": task.id,
                    "split": task.split,
                    "difficulty": task.difficulty,
                    "num_tests": len(task.tests),
                    "tests_fingerprint": tests_fingerprint(task),
                },
                "trajectory": traj.model_dump(mode="json"),
            },
        )

    def write_run(self, result: TrainResult) -> str:
        return self._append(
            "run",
            {
                "kind": "training_run",
                "at": datetime.now().isoformat(timespec="milliseconds"),
                "run_id": result.run_id,
                "result": result.model_dump(mode="json"),
            },
        )

    def write_rules(
        self,
        run_id: str,
        task: TaskSpec,
        rules: List[Rule],
        trajectory_ids: List[str],
        step: int,
    ) -> str:
        """记录规则写入，且把「规则 → 轨迹 → 步」的溯源链写清楚。"""
        return self._append(
            "rule",
            {
                "kind": "rule_write",
                "at": datetime.now().isoformat(timespec="milliseconds"),
                "run_id": run_id,
                "task_id": task.id,
                "trajectory_ids": trajectory_ids,
                "origin_step": step,
                "rules": [r.model_dump(mode="json") for r in rules],
            },
        )

    # ---------- 读 ----------

    def _read(self, kind: str, limit: int) -> List[Dict[str, Any]]:
        if not self.root.exists():
            return []
        files = sorted(self.root.glob(f"{kind}-*.jsonl"))
        out: List[Dict[str, Any]] = []
        for path in reversed(files):
            try:
                lines = path.read_text("utf-8").strip().splitlines()
            except Exception:
                continue
            for line in reversed(lines):
                if len(out) >= limit:
                    return out
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    def recent_trajectories(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._read("traj", limit)

    def recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self._read("run", limit)

    def recent_rules(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._read("rule", limit)

    def find_trajectory(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        for item in self.recent_trajectories(limit=500):
            if item.get("trajectory_id") == trajectory_id:
                return item
        return None
