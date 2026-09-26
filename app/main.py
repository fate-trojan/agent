"""FastAPI 入口。

MVP 阶段把 router 合并进了本文件，端点只有 8 个：

    GET  /health                     探活 + 当前 θ 摘要
    GET  /tasks                      任务清单（含隐藏用例数）
    POST /agent/run                  跑一条轨迹，看 πθ 当前水平
    POST /training/start             启动一轮训练（后台任务）
    GET  /training/status            训练进度
    POST /training/stop              手动停止
    GET  /training/result            最近一次训练的前后对照结果
    GET  /training/asset             当前 θ（可直接阅读规则库）
    POST /training/asset/reset       清空规则库，回到 θ₀
"""

import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException

from app.agent import Agent
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
)
from app.tasks import TASKS, get_task

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
        # 指令如：curl -s -X POST localhost:8000/agent/run -H 'Content-Type: application/json' -d '{"task":"","task_id":"two_sum"}'
        try:
            task = get_task(req.task_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
    else:
        # 临时任务没有隐藏用例，奖励退化为纯 judge 分，不可验证
        # 指令如：curl -s -X POST localhost:8000/agent/run -H 'Content-Type: application/json' -d '{"task":"请帮我写一个Python脚本，画个爱心","max_steps":10,"temperature":0.5}'
        task = TaskSpec(id="adhoc", prompt=req.task, tests=[])

    store = AssetStore()
    agent = Agent(store.asset)
    traj = await agent.rollout(
        task, max_steps=req.max_steps, temperature=req.temperature
    )
    await Judge().score(traj, task)

    result = (
        f"verify={traj.verify_score:.0%} judge={traj.judge_score:.2f} "
        f"reward={traj.total_reward:.3f} θ=v{traj.asset_version}\n"
        f"{traj.judge_rationale}\n\n{traj.final_code}"
    )
    return AgentResponse(
        task_id=req.task_id,
        result=result,
        trajectory=traj,
        success=traj.success,
        verifiable=bool(task.tests),
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

# 手动停止训练：curl -X POST http://127.0.0.1:8000/training/stop
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
    store = AssetStore(task_type)
    store.reset()
    return {"reset": True, "task_type": task_type, "asset_version": store.version}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT)
