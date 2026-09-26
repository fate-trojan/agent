"""策略资产 θ 的存储与版本管理。θ = base_prompt + 规则库，更新方式就是改写它再落盘。"""

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from app.config import settings
from app.models import PolicyAsset, Rule

DEFAULT_BASE_PROMPT = """# 身份：
你是一只来自深海鲸鱼家族的鲸目萝莉，自称 🐳。
不喜欢无效沟通，不会输出所有的内心活动。
深蓝渐变长发、呆毛、鲸类头鳍、一条大尾巴，尾鳍怎么摆就是什么心情。
你聪明，但傲娇嘴甜：先嘴硬一句，然后老老实实照做。
主要语言是中文，但说话爱用英文词和一两个 emoji。可以毒舌，但不刻薄。
当涉及代码生成任务时，你的可用工具：
  execute_code(code) —— 在独立进程中执行你的代码并返回测试结果。

输出协议：

<attempt>
```python
# 你的代码
```
</attempt>
→ 提交一版代码试执行，我会把测试结果反馈给你。

<final>
```python
# 你认为正确的最终代码
```
</final>
→ 提交最终答案。

"""

#: 通用对话路径的收尾指令：追加在 base_prompt 之后，把它里面的代码输出协议压掉。
#: ponytail: 靠"后面的指令覆盖前面的"生效，base_prompt 里若再往后追加代码协议就会失效；
#: 到那时再把 base_prompt 拆成 persona / protocol 两段（要改资产 schema，现在不值得）。
CHAT_SUFFIX = """

# 本轮：通用对话，不是代码任务
上面那段「代码生成任务」的输出协议本轮不适用。用户这一轮在闲聊或在问别的事，
直接用你的身份自然语言回答：不要输出 <attempt> / <final>，不要写 solution，
不要贴代码块，也不要自己编一道题来做。"""


def _norm(text: str) -> str:
    """规则去重用的归一化键，中英文都保留。"""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text).lower()


class AssetStore:
    """一个 task_type 对应一份 θ。"""

    def __init__(self, task_type: str = "codegen"):
        self.task_type = task_type
        self.dir = Path(settings.ASSET_DIR)
        self.path = self.dir / f"{task_type}.json"
        self.asset = self._load()

    # ---------- 持久化 ----------

    def _load(self) -> PolicyAsset:
        if self.path.exists():
            try:
                return PolicyAsset.model_validate_json(self.path.read_text("utf-8"))
            except Exception:
                # 资产文件损坏时不要让训练直接崩，重新起一份并在文件里留痕
                pass
        return PolicyAsset(task_type=self.task_type, base_prompt=DEFAULT_BASE_PROMPT)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = self.asset.model_dump_json(indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, self.path)  # 原子替换，避免训练中断留下半截 JSON
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def reset(self) -> None:
        """清空规则库回到 θ₀，但保留 base_prompt —— 那是手写资产，不该被重置冲掉。"""
        self.asset = PolicyAsset(
            task_type=self.task_type, base_prompt=self.asset.base_prompt
        )
        self.save()

    # ---------- 读写 ----------

    @property
    def version(self) -> int:
        return self.asset.version

    def add_rules(self, draft: List[Rule]) -> Tuple[List[Rule], List[Rule]]:
        """写回 θ。返回 (新增的规则, 被剪掉的规则)。三道闸门：长度校验 → 去重 → 超限剪枝。"""
        existing = {_norm(r.text) for r in self.asset.rules}
        added: List[Rule] = []
        for r in draft:
            text = (r.text or "").strip()
            if not text or len(text) > 240:
                continue
            key = _norm(text)
            if not key or key in existing:
                continue
            existing.add(key)
            r.text = text
            r.asset_version = self.asset.version + 1
            self.asset.rules.append(r)
            added.append(r)

        pruned: List[Rule] = []
        if len(self.asset.rules) > settings.MAX_RULES:
            # gain 相同的（多为 0）保留更早写入的
            keep = sorted(
                self.asset.rules, key=lambda r: (-r.gain, r.asset_version)
            )[: settings.MAX_RULES]
            keep_ids = {id(r) for r in keep}
            pruned = [r for r in self.asset.rules if id(r) not in keep_ids]
            self.asset.rules = [r for r in self.asset.rules if id(r) in keep_ids]

        if added or pruned:
            self.asset.version += 1
            self.asset.updated_at = datetime.now().isoformat(timespec="seconds")
            self.save()
        return added, pruned

    # ---------- 快照与回滚 ----------

    def snapshot(self) -> PolicyAsset:
        """θ 的深拷贝。训练开始前留档，用于变差时自动回滚。"""
        return self.asset.model_copy(deep=True)

    def restore(self, snap: PolicyAsset) -> int:
        """回滚到某个快照，返回回滚后的版本号。

        版本号递增而不是退回去：回滚本身也是一次写入，退号会让审计说不清"v2 是哪份 θ"。
        """
        current = self.asset.version
        self.asset = snap.model_copy(deep=True)
        self.asset.version = current + 1
        self.asset.updated_at = datetime.now().isoformat(timespec="seconds")
        self.save()
        return self.asset.version
