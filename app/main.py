"""FastAPI 入口：MVP 阶段把 router 合并进了本文件。

/runs 与 /runs/{trajectory_id} 是「可审计」的落地接口：#4 要求任何一次判定都能
回溯到具体轨迹、产物与环境状态，光把轨迹返给调用方不够，必须能被事后查回来。
"""

import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException

from app.agent import Agent, detect_mode
from app.asset import AssetStore
from app.config import settings
from app.grpo import GRPOTrainer
from app.judge import Judge
from app.models import (
    AgentRequest,
    AgentResponse,
    PolicyAsset,
    TaskSpec,
    TrainingRequest,
    TrainingStatus,
    TrainResult,
    Trajectory,
)
from app.tasks import TASKS, get_task
from app.trace import TraceStore

app = FastAPI(title="Agent RL (MVP)", version="0.1.0")

_trainer: Optional[GRPOTrainer] = None
_train_task: Optional[asyncio.Task] = None
_last_result: Optional[TrainResult] = None


def _require_key() -> None:
    if not settings.llm_configured:
        raise HTTPException(
            status_code=400, detail="DEEPSEEK_API_KEY 未配置"
        )


@app.get("/health")
async def health() -> dict:
    store = AssetStore()
    return {
        "status": "ok",
        "llm_configured": settings.llm_configured,
        "model": settings.DEEPSEEK_MODEL,
        "judge_model": settings.DEEPSEEK_JUDGE_MODEL,
        "asset_version": store.version,
        "rules_count": len(store.asset.rules),
    }


@app.get("/tasks")
async def list_tasks() -> dict:
    return {
        "tasks": [
            {
                "id": t.id,
                "split": t.split,
                "difficulty": t.difficulty,
                "num_tests": len(t.tests),
                "prompt": t.prompt,
            }
            for t in TASKS
        ]
    }


@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(req: AgentRequest) -> AgentResponse:
    _require_key()
    if req.task_id:
        try:
            task = get_task(req.task_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
    else:
        # 临时任务没有隐藏用例，奖励退化为纯 judge 分，不可验证
        task = TaskSpec(id="adhoc", prompt=req.task, tests=[])

    store = AssetStore()
    agent = Agent(store.asset)
    session_id = req.session_id
    # 带上该会话上一轮的提问与产出，否则「显示一下刚才画的结果」无从指代
    prior = TraceStore().last_turn(session_id)

    # 关键词路由：命中代码词走 RL 闭环，否则走通用对话（见 app/agent.py:detect_mode）
    mode = detect_mode(req.task)
    common = dict(temperature=req.temperature, prior=prior, session_id=session_id)
    traj = await (
        agent.reply(task, **common)
        if mode == "chat"
        else agent.rollout(task, max_steps=req.max_steps, **common)
    )
    if mode == "codegen":
        await Judge().score(traj, task)

    # 落盘：独立推理的轨迹也要能事后审计，这份同时是下一轮的会话记忆来源
    traj.trace_path = TraceStore().write_trajectory(traj, task)

    memory = f"会话 {session_id}：续用 {len(prior)} 条历史" if session_id else "无会话记忆"
    head = (
        "模式=chat（通用对话，无判定）"
        if mode == "chat"
        else f"模式=codegen verify={traj.verify_score:.0%} judge={traj.judge_score:.2f}"
        f" reward={traj.total_reward:.3f} θ=v{traj.asset_version}"
    )
    # chat 只有 reply，codegen 只有 final_code —— 用 or 一次覆盖两种
    body = traj.reply or f"{traj.judge_rationale}\n\n{traj.final_code}"
    result = (
        f"{head}｜{memory}\n"
        f"tokens 输入 {traj.prompt_tokens}（缓存命中 {traj.cache_hit_tokens}"
        f" / 未命中 {traj.cache_miss_tokens}，命中率 {traj.cache_hit_rate:.0%}）"
        f"，输出 {traj.completion_tokens}，judge {traj.judge_tokens}"
        f"，共 {traj.llm_calls} 次调用 / {traj.duration_ms}ms\n"
        f"{body}"
    )
    return AgentResponse(
        task_id=req.task_id,
        result=result,
        success=traj.success,
        verifiable=bool(task.tests),
        trajectory_id=traj.trajectory_id,
        trace_path=traj.trace_path,
    )


@app.post("/training/start", response_model=TrainingStatus)
async def start_training(req: TrainingRequest) -> TrainingStatus:
    global _trainer, _train_task, _last_result
    _require_key()
    if _trainer is not None and _trainer.status.running:
        raise HTTPException(status_code=409, detail="已有训练在运行，请先 POST /training/stop")

    group_size = req.group_size or settings.GRPO_GROUP_SIZE
    if group_size < 2:
        raise HTTPException(status_code=400, detail="group_size 必须 >= 2，否则组内无方差、优势恒为 0")

    _trainer = GRPOTrainer(req.task_type)
    # 必须同步把状态置为运行中：create_task 只是把 coroutine 排入队列，
    # coroutine 体要等事件循环拿到控制权才执行。否则这里返回的是构造函数里的
    # 初始状态(running=False)，调用方会以为训练压根没启动。
    _trainer.status = TrainingStatus(
        running=True,
        task_type=req.task_type,
        total_epochs=req.epochs,
        total_rollouts=req.num_rollouts,
        asset_version=_trainer.store.version,
        rules_count=len(_trainer.store.asset.rules),
        message="已排入后台，等待执行…",
    )

    async def _runner() -> None:
        global _last_result
        try:
            _last_result = await _trainer.train(
                num_rollouts=req.num_rollouts,
                epochs=req.epochs,
                group_size=group_size,
            )
        except Exception:
            pass  # 失败细节已写入 _trainer.status.error

    _train_task = asyncio.create_task(_runner())
    return _trainer.status


@app.get("/training/status", response_model=TrainingStatus)
async def training_status() -> TrainingStatus:
    if _trainer is None:
        return TrainingStatus(message="尚未启动过训练")
    return _trainer.status


@app.post("/training/stop")
async def stop_training() -> dict:
    if _trainer is None or not _trainer.status.running:
        return {"stopped": False, "message": "当前没有运行中的训练"}
    _trainer.stop()
    return {"stopped": True, "message": "已发出停止信号，将在当前 group 结束后退出"}


@app.get("/training/result", response_model=TrainResult)
async def training_result() -> TrainResult:
    if _last_result is None:
        raise HTTPException(status_code=404, detail="还没有完成的训练结果")
    return _last_result


@app.get("/training/asset", response_model=PolicyAsset)
async def get_asset(task_type: str = "codegen") -> PolicyAsset:
    return AssetStore(task_type).asset


@app.post("/training/asset/reset")
async def reset_asset(task_type: str = "codegen") -> dict:
    """清空规则库回到 θ₀。base_prompt 是手写资产，reset 不动它。"""
    store = AssetStore(task_type)
    store.reset()
    return {"reset": True, "task_type": task_type, "asset_version": store.version}


# ===================== 审计：轨迹与运行的落盘检索 =====================


@app.get("/runs")
async def list_runs(limit: int = 20) -> dict:
    """最近的轨迹与训练运行清单（每条轨迹只回摘要，正文用 /runs/{id} 取）。

    一次训练可能产生上百条轨迹，各自带十几次 LLM 往返的历史，全量返回会把响应
    撑爆；摘要保留 id、判定结果与落盘位置，足以定位。
    """
    trace = TraceStore()
    trajs = trace.recent_trajectories(limit=limit)
    return {
        "trajectories": [
            {
                "trajectory_id": t.get("trajectory_id"),
                "run_id": t.get("run_id"),
                "task": t.get("task", {}),
                "at": t.get("at"),
                "success": (t.get("trajectory") or {}).get("success"),
                "verify_score": (t.get("trajectory") or {}).get("verify_score"),
                "total_reward": (t.get("trajectory") or {}).get("total_reward"),
                "safety_passed": (t.get("trajectory") or {}).get("safety_passed"),
                "critical_step": (t.get("trajectory") or {}).get("critical_step"),
            }
            for t in trajs
        ],
        "runs": [
            {
                "run_id": r.get("run_id"),
                "at": r.get("at"),
                "rolled_back": (r.get("result") or {}).get("rolled_back"),
                "needs_human": (r.get("result") or {}).get("needs_human"),
                "escalations": (r.get("result") or {}).get("escalations", []),
                "delta_success_rate": (r.get("result") or {}).get("delta_success_rate"),
                "trace_path": (r.get("result") or {}).get("trace_path"),
            }
            for r in trace.recent_runs(limit=10)
        ],
        "rules": trace.recent_rules(limit=limit),
    }


@app.get("/runs/{trajectory_id}", response_model=dict)
async def get_run(trajectory_id: str) -> dict:
    """按 id 取回一条完整轨迹 —— 每一步的动作、环境状态、判定依据与计量。"""
    item = TraceStore().find_trajectory(trajectory_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"未找到轨迹 {trajectory_id}")
    return {
        "task": item.get("task", {}),
        "at": item.get("at"),
        "trajectory": Trajectory.model_validate(item.get("trajectory") or {}),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT)
